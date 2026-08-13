import os
import glob as _glob
import ctypes
import logging

# ── Preload libcusparseLt.so.0 for Jetson Orin (BEFORE importing torch) ──────
_matches = _glob.glob(os.path.expanduser(
    "~/.local/lib/python*/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"
))
if _matches:
    ctypes.CDLL(_matches[0])

os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

# ── Tắt spam H264 decode error từ FFmpeg / OpenCV / AMQTT ────────────────────
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
logging.getLogger("libav").setLevel(logging.CRITICAL)
logging.getLogger("amqtt").setLevel(logging.CRITICAL)

import torch
from ultralytics import YOLO
import cv2
from torchvision import models
import torchvision.transforms as T
import torch.nn.functional as F
import time
import threading
from queue import Queue, Empty
import signal
import requests
import numpy as np
import asyncio
from amqtt.client import MQTTClient
import json
import sqlite3
from collections import deque
from urllib.parse import urlparse
from scipy.optimize import linear_sum_assignment

# ── Suppress verbose FFmpeg output at C level ─────────────────────────────────
try:
    import ctypes as _ct
    import ctypes.util as _util
    _libav_path = _util.find_library("avcodec") or "libavcodec.so.58"
    _libav = _ct.cdll.LoadLibrary(_libav_path)
    _libav.av_log_set_level.argtypes = [_ct.c_int]
    _libav.av_log_set_level(8)   # AV_LOG_FATAL=8
except Exception:
    pass

# ── Detect GStreamer support in OpenCV (once at startup) ──────────────────────
_GST_AVAILABLE = False
try:
    _build_info = cv2.getBuildInformation()
    for _line in _build_info.splitlines():
        if "GStreamer" in _line and "YES" in _line:
            _GST_AVAILABLE = True
            break
except Exception:
    pass

print(f"{'✅' if _GST_AVAILABLE else '⚠️ '} GStreamer/NVDEC: {'khả dụng' if _GST_AVAILABLE else 'KHÔNG khả dụng → fallback FFmpeg'}")

# =============================================================================
# SMART TRACKER  — 1. IoU | 2. Center | 3. History
# =============================================================================
class _SmartTrack:
    _id_counter = 100000

    @classmethod
    def reset_counter(cls):
        cls._id_counter = 100000

    def __init__(self, box, conf):
        _SmartTrack._id_counter += 1
        self.track_id  = _SmartTrack._id_counter
        self.box       = np.array(box, dtype=float)   # [x1, y1, x2, y2]
        self.conf      = conf
        # 🔹 Bước 1: Tính center của box
        self.center    = self._get_center(self.box)
        self.age       = 1        # History: frame tồn tại
        self.miss      = 0        # History: frame không match

    def _get_center(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def update(self, box, conf):
        # 🔹 Bước 4: Update track
        self.box    = np.array(box, dtype=float)
        self.conf   = conf
        self.center = self._get_center(self.box)
        self.age   += 1
        self.miss   = 0

class SmartTracker:
    def __init__(self, iou_weight=0.3, dist_weight=0.7, dist_thresh=400, max_miss=2):
        self._tracks = []
        self.iou_weight = iou_weight
        self.dist_weight = dist_weight
        self.dist_thresh = dist_thresh
        self.max_miss = max_miss

    def reset(self):
        self._tracks = []
        _SmartTrack.reset_counter()

    def _iou(self, box_a, box_b):
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
        return iou

    def update(self, detections):
        # detections: list of [x1, y1, x2, y2, conf]
        if not detections:
            for t in self._tracks:
                t.miss += 1
                t.age += 1
            # 🔹 Bước 6: Xóa track chết
            self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]
            return self._collect()

        dets = np.array(detections, dtype=float)
        if len(self._tracks) == 0:
            # 🔹 Bước 5: Track mới
            for det in dets:
                self._tracks.append(_SmartTrack(det[:4], det[4]))
            return self._collect()

        cost_matrix = np.zeros((len(dets), len(self._tracks)))
        for d_idx, det in enumerate(dets):
            det_box = det[:4]
            det_center = np.array([(det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2])
            for t_idx, trk in enumerate(self._tracks):
                # 🔹 Bước 2: Cost function (CỰC QUAN TRỌNG)
                # 1. IoU (hình học)
                iou_val = self._iou(det_box, trk.box)
                # 2. Center distance (khoảng cách tâm)
                dist_val = np.linalg.norm(det_center - trk.center)
                dist_norm = min(dist_val / self.dist_thresh, 1.0) # Chuẩn hóa dist
                
                # Kết hợp IoU và Distance (iou cao là tốt -> cost thấp, dist nhỏ là tốt -> cost thấp)
                cost = self.iou_weight * (1.0 - iou_val) + self.dist_weight * dist_norm
                cost_matrix[d_idx, t_idx] = cost

        # 🔹 Bước 3: Matching
        ri, ci = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_dets = set()
        
        for r, c in zip(ri, ci):
            # Ngưỡng cost tối đa để coi là match (có thể điều chỉnh)
            if cost_matrix[r, c] < 0.8:
                self._tracks[c].update(dets[r][:4], dets[r][4])
                matched_tracks.add(c)
                matched_dets.add(r)
                
        # 🔹 Bước 5: Track mới
        for d_idx, det in enumerate(dets):
            if d_idx not in matched_dets:
                self._tracks.append(_SmartTrack(det[:4], det[4]))
                
        for t_idx, trk in enumerate(self._tracks):
            if t_idx not in matched_tracks:
                trk.miss += 1
                
        # 🔹 Bước 6: Xóa track chết
        self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]
        
        return self._collect()

    def _collect(self):
        out = []
        for t in self._tracks:
            # 3. History (trí nhớ theo thời gian) - Ẩn nhanh khi mất (miss <= 1)
            if t.age >= 1 and t.miss <= 1:
                x1, y1, x2, y2 = t.box.tolist()
                out.append([int(x1), int(y1), int(x2), int(y2), t.track_id, round(t.conf, 3), t.age])
        return out

# =============================================================================
# GLOBAL TRACKER  — Đấu nối ID liên camera qua ReID
# =============================================================================
reid_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GlobalTrack:
    _global_id_counter = 0

    @classmethod
    def reset_counter(cls):
        cls._global_id_counter = 0

    def __init__(self, feature, local_id_map, zone):
        GlobalTrack._global_id_counter += 1
        self.global_id = GlobalTrack._global_id_counter
        self.feature = feature # [Dim] (normalized)
        self.active_cams = local_id_map # dict {cam_id: local_id}
        self.last_seen = time.time()
        self.zone = zone
        
    def update(self, feature, cam_id, local_id, zone):
        # Update exponential moving average feature
        alpha = 0.9
        self.feature = alpha * self.feature + (1 - alpha) * feature
        self.feature /= (np.linalg.norm(self.feature) + 1e-6)
        self.active_cams[cam_id] = local_id
        self.last_seen = time.time()
        self.zone = zone

class GlobalTracker:
    def __init__(self, sim_thresh=0.55, max_lost=15.0):
        self.tracks = []
        self.sim_thresh = sim_thresh
        self.max_lost = max_lost

    def reset(self):
        self.tracks = []
        GlobalTrack.reset_counter()

    def update(self, detect_features, cam_id, local_ids):
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t.last_seen < self.max_lost]
        
        ZONE_1 = {1, 2, 4, 5}
        cam_zone = 1 if cam_id in ZONE_1 else 2

        if len(detect_features) == 0:
            return []
            
        out_global_ids = []
        if len(self.tracks) == 0:
            for f, lid in zip(detect_features, local_ids):
                nt = GlobalTrack(f, {cam_id: lid}, cam_zone)
                self.tracks.append(nt)
                out_global_ids.append(nt.global_id)
            return out_global_ids

        # Trích xuất track hợp lệ để so khớp (không conflict zone)
        track_features = []
        valid_track_indices = []
        for idx, t in enumerate(self.tracks):
            # Không cho phép chung ID nếu track đang active ở vùng khác (trong 5 giây gần đây)
            is_conflict = (t.zone != cam_zone and (now - t.last_seen < 5.0))
            if not is_conflict:
                track_features.append(t.feature)
                valid_track_indices.append(idx)
        
        if not track_features:
            for f, lid in zip(detect_features, local_ids):
                nt = GlobalTrack(f, {cam_id: lid}, cam_zone)
                self.tracks.append(nt)
                out_global_ids.append(nt.global_id)
            return out_global_ids

        track_features = np.array(track_features)
        sim_matrix = np.dot(detect_features, track_features.T) # [N, M]
        cost_matrix = 1.0 - sim_matrix
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_dets = set()
        out_global_ids = [None] * len(detect_features)

        for r, c in zip(row_ind, col_ind):
            sim = sim_matrix[r, c]
            if sim >= self.sim_thresh:
                orig_t_idx = valid_track_indices[c]
                t = self.tracks[orig_t_idx]
                t.update(detect_features[r], cam_id, local_ids[r], cam_zone)
                out_global_ids[r] = t.global_id
                matched_dets.add(r)
                
        for r in range(len(detect_features)):
            if r not in matched_dets:
                nt = GlobalTrack(detect_features[r], {cam_id: local_ids[r]}, cam_zone)
                self.tracks.append(nt)
                out_global_ids[r] = nt.global_id
                
        return out_global_ids

    def get_global_id(self, cam_id, local_id):
        for t in self.tracks:
            if cam_id in t.active_cams and t.active_cams[cam_id] == local_id:
                return t.global_id
        return None

global_tracker = GlobalTracker(sim_thresh=0.55, max_lost=30.0)
global_tracker_lock = threading.Lock()




# =============================================================================
# DATABASE
# =============================================================================
DB_PATH = "cameras.db"

def load_cameras():
    """Đọc danh sách camera đang bật từ DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            SELECT device_id, device_name, ip_address, mac_address, mqtt_topic, status
            FROM cameras
            WHERE status = 'online'
            ORDER BY device_id
        """)
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"⚠️  load_cameras error: {e}")
        return []
    return [
        {
            "id":         row[0],
            "name":       row[1],
            "rtsp":       row[2],
            "mac":        row[3],
            "mqtt_topic": row[4],
            "status":     row[5],
        }
        for row in rows
    ]

# =============================================================================
# GPU / MODEL
# =============================================================================
DEVICE     = 0
MODEL_PATH = "yolov8s.pt"
torch.backends.cudnn.benchmark = True

# =============================================================================
# CONFIG
# =============================================================================
# DETECT_FPS: fps gửi frame từ mỗi cam vào queue
#   - 6 cam  → dùng 5 fps/cam  (tổng 30 fps, YOLO ~20fps → OK với batch)
#   - 20 cam → dùng 2 fps/cam  (tổng 40 fps, YOLO ~20fps → cần batch=2+)
#   Công thức: DETECT_FPS ≤ YOLO_FPS / NUM_CAMS
#
# BATCH_SIZE = 1: TRT engine hiện tại là static batch=1
#   → Re-export với dynamic batch=8 để tăng throughput lên 4–8×
DETECT_FPS    = 3      # fps gửi frame vào queue mỗi cam (giảm cho 20 cam)
DISPLAY_FPS   = 3     # fps hiển thị cửa sổ OpenCV (riêng biệt với detect)
BATCH_SIZE    = 4           # TRT engine / PyTorch max frames cho batch computation
QUEUE_PER_CAM = 1           # realtime thật
CONF_THRESH   = 0.33       # ngưỡng confidence
PERSON_CLASS  = 0           # COCO class index "person"

# Rolling window để đo FPS chính xác
FPS_WINDOW    = 5.0         # giây (tăng lên vì FPS thấp hơn)

RTSP_RETRY_DELAY = 10       # giây

LOG_INTERVAL = 1.0
LOG_FILE     = "camera_stats.csv"

# Display layout
DISPLAY_CELL_W = 640        # chiều rộng mỗi ô trong grid hiển thị
DISPLAY_CELL_H = 360        # chiều cao mỗi ô
DISPLAY_COLS   = 2          # số cột trong grid (tự động xuống hàng theo số cam)

DB_POLL_INTERVAL = 10       # giây

# =============================================================================
# MQTT CONFIG
# =============================================================================
MQTT_BROKER       = "100.99.88.11"
MQTT_PORT         = 1883
MQTT_TOPIC        = "camera/stats"
CLIENT_ID         = "machine_a_camera_ai"
API_SEND_INTERVAL = 5
MQTT_URI          = f"mqtt://{MQTT_BROKER}:{MQTT_PORT}/"

# =============================================================================
# SHARED STATE
# =============================================================================
state_lock   = threading.Lock()

CAMERAS      = load_cameras()
CAM_IDS      = [c["id"] for c in CAMERAS]
TOTAL_VIDEO  = len(CAM_IDS)

init_time    = int(time.time())
cam_topic_map: dict[int, str] = {c["id"]: c["mqtt_topic"] for c in CAMERAS}

# Ánh xạ cam_id (DB) → IP camera
cam_ip_map: dict[int, str] = {c["id"]: (urlparse(c["rtsp"]).hostname or str(c["id"])) for c in CAMERAS}

camera_state = {
    cid: {"timestamp": init_time, "fps": 0.0, "person_ids": [], "is_night": "0", "boxes": []}
    for cid in CAM_IDS
}
frame_queues = {cid: Queue(maxsize=QUEUE_PER_CAM) for cid in CAM_IDS}

# Queue riêng cho display: lưu frame gần nhất để vẽ bbox lên
display_frames: dict[int, np.ndarray] = {}   # cam_id → frame mới nhất
display_lock = threading.Lock()

cam_stop_events: dict[int, threading.Event] = {}

detect_timestamps: dict[int, list] = {cid: [] for cid in CAM_IDS}

last_detect_time: dict[int, float] = {}

# Mỗi cam có 1 SmartTracker riêng biệt
cam_trackers: dict[int, SmartTracker] = {
    cid: SmartTracker() for cid in CAM_IDS
}

STALE_TIMEOUT = 5.0

# =============================================================================
# INIT LOG FILE
# =============================================================================
with open(LOG_FILE, "w") as f:
    f.write("timestamp,cam_ip,person_ids,is_night\n")

# =============================================================================
# DAY / NIGHT
# =============================================================================
def get_brightness(frame) -> str:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hsv[:, :, 1].mean() < 20:
        return "1"   # IR / tối

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    b = p50 * 0.5 + (p10 + p90) * 0.25

    if b < 25.5: return "1"
    elif b < 51: return "2"
    elif b < 76.5: return "3"
    elif b < 102: return "4"
    elif b < 127.5: return "5"
    elif b < 153: return "6"
    elif b < 178.5: return "7"
    elif b < 204: return "8"
    elif b < 229.5: return "9"
    else: return "10"

def _is_gray_frame(frame, threshold=2.0):
    return frame.std() < threshold

def enhance_for_ai(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_frame

def _make_gst_pipeline(url: str) -> str:
    """
    GStreamer pipeline dùng decodebin (tự chọn decoder tốt nhất).
    NOTE đã test: nvv4l2decoder block vô hạn, avdec_h264 trả frame xám.
    decodebin sẽ tự chọn; nếu cưng không work thì fallback FFmpeg/TCP.
    """
    return (
        f"rtspsrc location={url} latency=300 protocols=tcp+udp "
        "! decodebin "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=1 max-buffers=4 sync=false emit-signals=false"
    )

def rtsp_worker(cam: dict, stop_event: threading.Event):
    """
    Đọc RTSP stream qua GStreamer/NVDEC (ưu tiên) hoặc FFmpeg (fallback).
    Gửi frame vào queue đúng nhịp DETECT_FPS.
    """
    cam_id        = cam["id"]
    url           = cam["rtsp"]
    cam_ip        = urlparse(url).hostname or ""

    while not stop_event.is_set():
        # ── Mở capture ────────────────────────────────────────────────────────
        # NOTE: GStreamer đã test (nvv4l2decoder, avdec_h264, decodebin):
        #   - nvv4l2decoder: block vô hạn với stream portrait 480x852
        #   - avdec_h264: trả frame xám (mean=128, std≈0)
        #   - decodebin: quá chậm negotiate, không kịp trong thời gian hợp lý
        # → Bắt buộc dùng FFmpeg/TCP làm primary decode.
        use_gst = False
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|analyzeduration;100000|probesize;100000"
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


        if not cap.isOpened():
            print(f"❌ Cam {cam_id}: không kết nối — thử lại sau {RTSP_RETRY_DELAY}s")
            with state_lock:
                if cam_id in camera_state:
                    camera_state[cam_id]["fps"] = -1.0
            cap.release()
            stop_event.wait(timeout=RTSP_RETRY_DELAY)
            continue

        backend = "GStreamer/NVDEC" if use_gst else "FFmpeg"
        print(f"✅ Cam {cam_id}: kết nối RTSP OK [{backend}]")
        fail_count  = 0
        first_frame = True   # tránh GStreamer-CRITICAL caps warning lúc đầu

        while not stop_event.is_set():
            ret, frame = cap.read()

            # GStreamer appsink đôi khi trả về ret=True nhưng frame=None
            # hoặc frame xám (mean≈78-130, std<2) khi pipeline flush/EOS
            if ret and (frame is None or (use_gst and _is_gray_frame(frame))):
                fail_count += 1
                if fail_count == 1 and use_gst:
                    print(f"⚠️  Cam {cam_id}: GStreamer trả frame xám/NULL → fallback FFmpeg")
                    cap.release()
                    use_gst = False
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        "rtsp_transport;tcp|analyzeduration;100000|probesize;100000"
                    )
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if cap.isOpened():
                        print(f"✅ Cam {cam_id}: reconnect FFmpeg/TCP OK")
                        fail_count = 0
                    else:
                        fail_count = 10  # force reconnect loop
                    continue
                if fail_count >= 10:
                    print(f"⚠️  Cam {cam_id}: frame NULL/xám liên tục — reconnect sau {RTSP_RETRY_DELAY}s")
                    with state_lock:
                        if cam_id in camera_state:
                            camera_state[cam_id]["fps"] = -1.0
                    cap.release()
                    stop_event.wait(timeout=RTSP_RETRY_DELAY)
                    break
                time.sleep(0.02)
                continue

            if not ret:
                fail_count += 1
                if fail_count >= 5:
                    print(f"⚠️  Cam {cam_id}: mất kết nối — reconnect sau {RTSP_RETRY_DELAY}s")
                    with state_lock:
                        if cam_id in camera_state:
                            camera_state[cam_id]["fps"] = -1.0
                    cap.release()
                    stop_event.wait(timeout=RTSP_RETRY_DELAY)
                    break
                time.sleep(0.01)
                continue

            if first_frame:
                first_frame = False   # caps đã negotiated, warnings sẽ dừng

            fail_count = 0
            # frame_resized = frame
            # Resize frame về 1280x720
            frame_resized = cv2.resize(frame, (1920, 1080))

            # Cập nhật frame mới nhất cho display worker
            with display_lock:
                display_frames[cam_id] = frame_resized.copy()

            q: Queue = frame_queues.get(cam_id)
            if q is not None:
                qsize_before = q.qsize()
                if q.full():
                    try: q.get_nowait()   # drop oldest → luôn có frame mới
                    except Empty: pass
                q.put_nowait(frame_resized)
                # Debug: log khi queue fill cao bất thường
                if qsize_before >= QUEUE_PER_CAM - 1:
                    # print(f"[Q] Cam {cam_id}: queue={qsize_before}/{QUEUE_PER_CAM} (FULL → dropped old)")
                    continue
        cap.release()

    print(f"🔴 Cam {cam_id}: thread dừng")

# =============================================================================
# YOLO WORKER  — 1 thread duy nhất + smart batch collection
# =============================================================================
model_init_lock = threading.Lock()

def _yolo_thread_logic():
    """
    1 thread duy nhất: smart round-robin + batch collection.
    - Ưu tiên cam nào có qsize cao nhất để giảm latency
    - Gom tối đa BATCH_SIZE frames rồi inference 1 lần
    - Idle sleep cực ngắn (0.0005s) để phản ứng nhanh khi có frame
    """
    print("[YOLO] Waiting for lock to load TensorRT...")
    with model_init_lock:
        print(f"[YOLO] Loading {MODEL_PATH}...")
        local_model = YOLO(MODEL_PATH, task="detect")
        try:
            local_model.predict(
                torch.zeros(1, 3, 640, 640, device=DEVICE),
                verbose=False
            )
            print("[YOLO] Warmup complete.")
        except Exception as e:
            print(f"[YOLO] Warmup error (ignored): {e}")

        import torchreid
        print("[ReID] Loading OSNet pretrained...")
        reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            loss='softmax',
            pretrained=True
        ).to(DEVICE).eval()
        print("[ReID] Loaded.")

    loop_count = 0
    while True:
        loop_count += 1
        if loop_count % 30 == 0:
            torch.cuda.empty_cache()

        # ── Lấy snapshot cam ids ──────────────────────────────────────────────
        with state_lock:
            cur_ids = list(CAM_IDS)

        n = len(cur_ids)
        if n == 0:
            time.sleep(0.1)
            continue

        # ── Smart scan: chỉ chọn cam có frame, sort theo qsize giảm dần ──────
        cams_with_frames = []
        for cid in cur_ids:
            q = frame_queues.get(cid)
            if q is not None:
                qs = q.qsize()
                if qs > 0:
                    cams_with_frames.append((cid, qs))

        if not cams_with_frames:
            time.sleep(0.0005)   # idle ngắn → phản ứng nhanh khi có frame
            continue

        # Ưu tiên cam bị "bỏ đói" lâu nhất (phân bổ FPS đều nhau)
        # Nếu chưa từng xử lý, get() trả về 0.0 → được ưu tiên cao nhất.
        # Ở cùng một thời điểm chờ, ưu tiên cam có queue dài hơn (-x[1]).
        with state_lock:
            ldt = dict(last_detect_time)
        cams_with_frames.sort(key=lambda x: (ldt.get(x[0], 0.0), -x[1]))

        # ── Thu thập batch ────────────────────────────────────────────────────
        batch_frames  = []
        batch_cam_ids = []

        for cid, _ in cams_with_frames:
            if len(batch_frames) >= BATCH_SIZE:
                break
            q = frame_queues.get(cid)
            if q is None:
                continue
            try:
                frame = q.get_nowait()
                batch_frames.append(frame)
                batch_cam_ids.append(cid)
            except Empty:
                continue

        if not batch_frames:
            time.sleep(0.0005)
            continue

        # ── GPU Batch Inference (Siêu tối ưu Inference Gộp) ─────────────────────
        with torch.no_grad():
            # 🔹 2. Enhance riêng cho AI bằng CLAHE (tối ưu chỉ dùng khi thiếu sáng)
            batch_enhanced = []
            for f in batch_frames:
                if get_brightness(f) <= "4":
                    batch_enhanced.append(enhance_for_ai(f))
                else:
                    batch_enhanced.append(f)

            # 🔹 3. Cắt mỗi frame làm 2 nửa (Trái/Phải) để Detect với Margin
            batch_crops = []
            crop_offsets = [] # (x_offset, y_offset)
            margin = 140
            
            for f in batch_enhanced:
                h, w = f.shape[:2]
                cw = w // 2
                
                # Nửa trái (kèm margin)
                batch_crops.append(f[:, :min(w, cw + margin)])
                crop_offsets.append((0, 0))
                
                # Nửa phải (kèm margin)
                batch_crops.append(f[:, max(0, cw - margin):])
                crop_offsets.append((max(0, cw - margin), 0))

            # Bước 1. Inference YOLO lần lượt từng ảnh (batch=1) y hệt như scale_video_2 để an toàn tuyệt đối, chống CUDACachingAllocator Error 12
            results = []
            for crop in batch_crops:
                r = local_model.predict(
                    source=crop,
                    device=DEVICE,
                    imgsz=640,
                    classes=[PERSON_CLASS],
                    conf=CONF_THRESH,
                    verbose=False,
                    stream=False,
                )[0]
                results.append(r)
            
            MAX_REID = 128
            all_reid_tensors = []
            all_valid_boxes = [] # tuple (idx_in_batch, box_data)
            all_tracked_boxes_list = []
            
            # Bước 2. Xử lý Tracking Cục Bộ (Local Tracker) cho từng camera
            for i in range(len(batch_frames)):
                cid = batch_cam_ids[i]
                frame = batch_frames[i]      # ảnh gốc
                frame_enhanced = batch_enhanced[i] # ảnh đã enhance
                
                # Gom kết quả từ 2 nửa
                res_left = results[2*i]
                res_right = results[2*i + 1]
                
                temp_boxes = []
                for idx, res in enumerate([res_left, res_right]):
                    offset_x, offset_y = crop_offsets[2*i + idx]
                    if res.boxes is not None and len(res.boxes) > 0:
                        for box in res.boxes:
                            coords = box.xyxy[0].cpu().numpy()
                            cx1, cy1, cx2, cy2 = coords
                            
                            # Cộng thêm offset để đưa box về tọa độ gốc của frame lớn
                            cx1 += offset_x
                            cx2 += offset_x
                            cy1 += offset_y
                            cy2 += offset_y
                            # Lọc rác (quá nhỏ, false positive)
                            if (cx2 - cx1) * (cy2 - cy1) < 2000:
                                continue
                            conf_val = float(box.conf)
                            temp_boxes.append([int(cx1), int(cy1), int(cx2), int(cy2), conf_val])
                            
                # NMS loại bỏ trùng lặp vùng margin
                final_temp_boxes = []
                if len(temp_boxes) > 0:
                    temp_boxes = sorted(temp_boxes, key=lambda x: x[4], reverse=True)
                    for b_new in temp_boxes:
                        is_duplicate = False
                        for k_box in final_temp_boxes:
                            x1_inter = max(b_new[0], k_box[0])
                            y1_inter = max(b_new[1], k_box[1])
                            x2_inter = min(b_new[2], k_box[2])
                            y2_inter = min(b_new[3], k_box[3])
                            
                            if x2_inter > x1_inter and y2_inter > y1_inter: # Có giao nhau
                                inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                                b_area = (b_new[2] - b_new[0]) * (b_new[3] - b_new[1])
                                k_area = (k_box[2] - k_box[0]) * (k_box[3] - k_box[1])
                                
                                union_area = b_area + k_area - inter_area
                                iou = inter_area / union_area if union_area > 0 else 0
                                ioa = inter_area / min(b_area, k_area) if min(b_area, k_area) > 0 else 0
                                
                                if iou > 0.40 or ioa > 0.75:
                                    is_duplicate = True
                                    break
                                    
                        if not is_duplicate:
                            final_temp_boxes.append(b_new)
                temp_boxes = final_temp_boxes
                        
                tracker = cam_trackers.get(cid)
                if tracker is None:
                    tracker = SmartTracker()
                    cam_trackers[cid] = tracker
                
                tracked_boxes = tracker.update(temp_boxes)
                all_tracked_boxes_list.append(tracked_boxes)
                
                # Cắt Crop chuẩn bị Feature Extraction
                h_img, w_img = frame_enhanced.shape[:2]
                for box in tracked_boxes:
                    bx1, by1, bx2, by2, tid, conf, age = box
                    x1, y1 = max(0, int(bx1)), max(0, int(by1))
                    x2, y2 = min(w_img, int(bx2)), min(h_img, int(by2))
                    
                    if x2 - x1 >= 10 and y2 - y1 >= 10:
                        # Dùng ảnh đã enhance cho ReID. Chỉ chạy ReID cho box mới, hoặc mỗi 5 frame
                        if age % 5 == 0 or age == 1:
                            if len(all_reid_tensors) < MAX_REID:
                                crop = frame_enhanced[y1:y2, x1:x2]
                                t = reid_transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                all_reid_tensors.append(t)
                                all_valid_boxes.append((i, box))

            # Bước 3. Trích xuất đặc trưng ReID GỘP của mọi camera TẠI 1 LẦN GỌI DUY NHẤT
            if all_reid_tensors:
                batch_t = torch.stack(all_reid_tensors).to(DEVICE)
                batch_features = reid_model(batch_t).cpu().numpy()
                batch_features = batch_features / (np.linalg.norm(batch_features, axis=1, keepdims=True) + 1e-6)
            else:
                batch_features = []

            # Bước 4. Phân phối Global ID và lưu ngược về từng Camera Object
            feature_idx = 0
            for i in range(len(batch_frames)):
                cid = batch_cam_ids[i]
                frame = batch_frames[i]
                tracked_boxes = all_tracked_boxes_list[i]
                now = time.time()
                
                cam_features = []
                cam_local_ids = []
                cam_valid_boxes = []
                
                while feature_idx < len(all_valid_boxes) and all_valid_boxes[feature_idx][0] == i:
                    box = all_valid_boxes[feature_idx][1]
                    cam_valid_boxes.append(box)
                    cam_local_ids.append(box[4]) # id cục bộ
                    cam_features.append(batch_features[feature_idx])
                    feature_idx += 1
                    
                final_global_boxes = []
                
                if cam_features:
                    with global_tracker_lock:
                        g_ids = global_tracker.update(np.array(cam_features), cid, cam_local_ids)
                    
                    for box, gid in zip(cam_valid_boxes, g_ids):
                        b_new = list(box)
                        b_new[4] = gid
                        final_global_boxes.append(b_new)
                
                # Xử lý các object không chạy ReID ở frame hiện tại để gán ID cũ hoặc lưu ID local fallback
                reid_local_ids = set(cam_local_ids)
                for box in tracked_boxes:
                    local_id = box[4]
                    if local_id not in reid_local_ids:
                        with global_tracker_lock:
                            gid = global_tracker.get_global_id(cid, local_id)
                        if gid is None:
                            gid = local_id # fallback về local track id
                        
                        b_new = list(box)
                        b_new[4] = gid
                        final_global_boxes.append(b_new)

                # Bảo vệ độc quyền ID trong cùng 1 camera (đã cắt đôi nhưng vẫn phải Unique ID)
                seen_gids = set()
                sanitized_boxes = []
                for b in final_global_boxes:
                    gid = b[4]
                    if gid in seen_gids:
                        # Tránh ID G_1 xuất hiện 2 lần trên cùng 1 camera
                        gid = gid + 200000 
                        b[4] = gid
                    seen_gids.add(gid)
                    sanitized_boxes.append(b)
                final_global_boxes = sanitized_boxes

                # Đồng bộ trạng thái lưu DB & Visualization
                with state_lock:
                    ts = detect_timestamps.setdefault(cid, [])
                    ts.append(now)
                    cutoff = now - FPS_WINDOW
                    detect_timestamps[cid] = [t for t in ts if t > cutoff]
                    fps_val = round(len(detect_timestamps[cid]) / FPS_WINDOW, 2)

                    person_ids = [f"G_{box[4]}" for box in final_global_boxes]

                    last_detect_time[cid] = now
                    brightness = get_brightness(frame)

                    if cid in camera_state:
                        camera_state[cid] = {
                            "timestamp":  int(now),
                            "fps":        fps_val,
                            "person_ids": person_ids,
                            "is_night":   brightness,
                            "boxes":      final_global_boxes,
                        }

def yolo_worker():
    """Entry point: chạy 1 thread YOLO duy nhất."""
    t = threading.Thread(target=_yolo_thread_logic, name="yolo_t0", daemon=True)
    t.start()
    # t.join()

# =============================================================================
# DISPLAY WORKER  — hiển thị video stream với bbox + số người theo DISPLAY_FPS
# =============================================================================
def display_worker():
    """
    Hiển thị tất cả camera thành grid (DISPLAY_COLS cột).
    Mỗi ô vẽ:
      - bounding box các person (màu xanh lá)
      - label 'Px conf%' trên mỗi box
      - góc trên trái: tên cam + FPS detect
      - góc trên phải: tổng số người (lớn + nổi bật)
    Chạy theo nhịp DISPLAY_FPS (độc lập với DETECT_FPS).
    """
    interval = 1.0 / DISPLAY_FPS
    win_name = "Camera AI - Person Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    CELL_W = DISPLAY_CELL_W
    CELL_H = DISPLAY_CELL_H
    COLS   = DISPLAY_COLS

    # Màu sắc box
    BOX_COLOR   = (0, 220, 0)       # xanh lá
    TEXT_COLOR  = (255, 255, 255)   # trắng
    SHADOW_COLOR= (0, 0, 0)         # viền đen
    COUNT_BG    = (0, 180, 0)       # nền xanh cho số người

    while True:
        t0 = time.perf_counter()

        with state_lock:
            cur_ids = list(CAM_IDS)

        n = len(cur_ids)
        if n == 0:
            blank = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            cv2.putText(blank, "No cameras", (20, CELL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)
            cv2.imshow(win_name, blank)
            if cv2.waitKey(max(1, int(interval * 1000))) & 0xFF == ord('q'):
                break
            continue

        rows = (n + COLS - 1) // COLS
        grid = np.zeros((rows * CELL_H, COLS * CELL_W, 3), dtype=np.uint8)

        with state_lock:
            snap_state = {cid: dict(camera_state[cid]) for cid in cur_ids if cid in camera_state}
        with display_lock:
            snap_frames = dict(display_frames)

        for idx, cid in enumerate(sorted(cur_ids)):
            row_i = idx // COLS
            col_i = idx % COLS
            y_off = row_i * CELL_H
            x_off = col_i * CELL_W

            frame_src = snap_frames.get(cid)
            if frame_src is None:
                cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
                cv2.putText(cell, f"Cam {cid}: connecting...",
                            (10, CELL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 100, 100), 1)
            else:
                # Resize frame gốc về kích thước ô
                orig_h, orig_w = frame_src.shape[:2]
                cell = cv2.resize(frame_src, (CELL_W, CELL_H))
                scale_x = CELL_W / orig_w
                scale_y = CELL_H / orig_h

                # Vẽ bounding box + Track ID (SmartTracker)
                s = snap_state.get(cid, {})
                boxes = s.get("boxes", [])
                for box in boxes:
                    # format: [x1, y1, x2, y2, track_id, conf]
                    bx1 = int(box[0] * scale_x)
                    by1 = int(box[1] * scale_y)
                    bx2 = int(box[2] * scale_x)
                    by2 = int(box[3] * scale_y)
                    tid      = int(box[4]) if len(box) > 4 else 0
                    conf_pct = int(box[5] * 100) if len(box) > 5 else int(box[4] * 100)
                    
                    # Màu box khác nhau cho các Track ID
                    hue = (tid * 47) % 180
                    hsv_color = np.uint8([[[hue, 220, 220]]])
                    bgr_color  = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    cv2.rectangle(cell, (bx1, by1), (bx2, by2), bgr_color, 2)
                    
                    # Label: G_7 87%
                    label = f"G_{tid} {conf_pct}%"
                    lx, ly = bx1, max(by1 - 6, 14)
                    cv2.putText(cell, label, (lx + 1, ly + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SHADOW_COLOR, 2)
                    cv2.putText(cell, label, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)

                # Vẽ đường phân cách giữa 2 nửa (Trái/Phải) bằng màu vàng
                cv2.line(cell, (CELL_W // 2, 0), (CELL_W // 2, CELL_H), (0, 255, 255), 1)

                # ── Thông tin góc trên trái: cam name + detect FPS ────────────
                cam_ip  = cam_ip_map.get(cid, str(cid))
                fps_val = s.get("fps", 0.0)
                info_txt = f"Cam {cam_ip}  |  {fps_val:.1f} fps"
                cv2.putText(cell, info_txt, (11, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, SHADOW_COLOR, 3)
                cv2.putText(cell, info_txt, (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)

                # ── Số người: nền màu + chữ lớn góc trên phải ────────────────
                n_people = len(s.get("person_ids", []))
                count_txt = f"{n_people} nguoi"
                (tw, th), _ = cv2.getTextSize(
                    count_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cx = CELL_W - tw - 12
                cy = 32
                # nền nhỏ phía sau
                cv2.rectangle(cell,
                              (cx - 6, cy - th - 6),
                              (cx + tw + 6, cy + 4),
                              COUNT_BG, -1)
                cv2.putText(cell, count_txt, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

            grid[y_off:y_off + CELL_H, x_off:x_off + CELL_W] = cell

        cv2.imshow(win_name, grid)

        elapsed = time.perf_counter() - t0
        wait_ms = max(1, int((interval - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# =============================================================================
# LOG WRITER
# =============================================================================
def log_writer_worker():
    while True:
        time.sleep(LOG_INTERVAL)
        now    = time.time()
        cutoff = now - FPS_WINDOW
        with state_lock:
            cur_ids  = list(CAM_IDS)
            snapshot = {cid: dict(camera_state[cid]) for cid in cur_ids if cid in camera_state}
            fps_snap = {
                cid: round(
                    len([t for t in detect_timestamps.get(cid, []) if t > cutoff]) / FPS_WINDOW, 2
                )
                for cid in cur_ids
            }
            last_det_snap = {cid: last_detect_time.get(cid) for cid in cur_ids}

        lines = ["timestamp,cam_ip,FPS,person_ids,is_night\n"]
        for cid in sorted(cur_ids):
            s    = snapshot.get(cid)
            last = last_det_snap.get(cid)
            if not s:
                continue
            stale      = (last is None) or (now - last > STALE_TIMEOUT)
            cam_ip     = cam_ip_map.get(cid, str(cid))
            fps_val    = fps_snap.get(cid, 0.0)
            ids_str    = "|".join(s.get("person_ids", [])) if not stale else ""
            lines.append(
                f"{s['timestamp']},{cam_ip},{fps_val},{ids_str},{s['is_night']}\n"
            )

        try:
            with open(LOG_FILE, "w", buffering=1) as f:
                f.writelines(lines)
        except Exception as e:
            print(f"❌ log_writer error: {e}")

# =============================================================================
# MQTT SENDER
# =============================================================================
async def _async_mqtt_sender():
    _cfg = {"reconnect_retries": 0, "reconnect_max_interval": 5}
    while True:
        client = MQTTClient(client_id=CLIENT_ID, config=_cfg)
        try:
            await client.connect(MQTT_URI)
            print(f"✅ Đã kết nối MQTT broker: {MQTT_BROKER}")
        except Exception as e:
            print(f"⚠️  MQTT chưa kết nối: {e} — thử lại sau 5s")
            await asyncio.sleep(5)
            continue

        try:
            while True:
                await asyncio.sleep(API_SEND_INTERVAL)
                now2    = time.time()
                cutoff2 = now2 - FPS_WINDOW
                with state_lock:
                    cur_ids    = list(CAM_IDS)
                    snap_state = {cid: dict(camera_state[cid]) for cid in cur_ids if cid in camera_state}
                    snap_fps   = {
                        cid: round(
                            len([t for t in detect_timestamps.get(cid, []) if t > cutoff2]) / FPS_WINDOW, 2
                        )
                        for cid in cur_ids
                    }
                    snap_topics = dict(cam_topic_map)

                sent = 0
                for cid in cur_ids:
                    if cid not in snap_state:
                        continue
                    topic   = snap_topics.get(cid, MQTT_TOPIC)
                    s       = snap_state[cid]
                    
                    payload = {
                        
                        "person_ids":  s.get("person_ids", []),
                        "light_level": int(s["is_night"]),
                    }
                    await client.publish(topic, json.dumps(payload).encode(), qos=0x01)
                    sent += 1

                print(f"📤 Đã gửi MQTT: {sent} cameras")
        except Exception as e:
            print(f"❌ MQTT send error: {e} — reconnect sau 3s")
            await asyncio.sleep(3)
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

def mqtt_sender_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_async_mqtt_sender())

# =============================================================================
# DB WATCHER  — hot-reload cameras.db mỗi DB_POLL_INTERVAL giây
# =============================================================================
def db_watcher_worker():
    global CAM_IDS, TOTAL_VIDEO

    while True:
        time.sleep(DB_POLL_INTERVAL)

        new_cameras = load_cameras()
        new_ids     = {c["id"] for c in new_cameras}
        new_cam_map = {c["id"]: c for c in new_cameras}

        with state_lock:
            old_ids = set(CAM_IDS)

        added   = new_ids - old_ids
        removed = old_ids - new_ids

        for cid in removed:
            evt = cam_stop_events.pop(cid, None)
            if evt:
                evt.set()
            cam_trackers.pop(cid, None)
            with state_lock:
                camera_state.pop(cid, None)
                frame_queues.pop(cid, None)
                detect_timestamps.pop(cid, None)
                last_detect_time.pop(cid, None)
                cam_topic_map.pop(cid, None)
                cam_ip_map.pop(cid, None)
            print(f"🔴 DB watcher: cam {cid} bị xóa/disable")

        for cid in added:
            cam = new_cam_map[cid]
            with state_lock:
                frame_queues[cid]      = Queue(maxsize=QUEUE_PER_CAM)
                detect_timestamps[cid] = []
                camera_state[cid]      = {
                    "timestamp": int(time.time()),
                    "fps": 0.0, "person_ids": [], "is_night": "0"
                }
                cam_topic_map[cid]     = cam.get("mqtt_topic", MQTT_TOPIC)
                cam_ip_map[cid]        = urlparse(cam.get("rtsp", "")).hostname or str(cid)
            cam_trackers[cid] = SmartTracker()
            stop_evt = threading.Event()
            cam_stop_events[cid] = stop_evt
            threading.Thread(
                target=rtsp_worker, args=(cam, stop_evt), daemon=True
            ).start()
            print(f"🟢 DB watcher: cam {cid} mới → khởi thread (topic: {cam.get('mqtt_topic', MQTT_TOPIC)})")

        if added or removed:
            with state_lock:
                CAM_IDS         = sorted(new_ids)
                TOTAL_VIDEO     = len(CAM_IDS)
            print(f"📋 DB watcher: tổng {TOTAL_VIDEO} cameras đang chạy")

# =============================================================================
# DAILY RESET WORKER
# =============================================================================
def daily_reset_worker():
    """Tự động reset bộ đếm người vào 00:00 mỗi ngày"""
    last_reset_day = time.localtime().tm_mday
    while True:
        time.sleep(10)
        now_time = time.localtime()
        if now_time.tm_hour == 0 and now_time.tm_min == 0 and now_time.tm_mday != last_reset_day:
            print("🕒 Đang tiến hành reset số đếm người lúc nửa đêm...")
            with global_tracker_lock:
                global_tracker.reset()
                for cid, tracker in cam_trackers.items():
                    tracker.reset()
            last_reset_day = now_time.tm_mday
            print("✅ Đã reset thành công!")

# =============================================================================
# KHỞI ĐỘNG
# =============================================================================
for cam in CAMERAS:
    evt = threading.Event()
    cam_stop_events[cam["id"]] = evt
    threading.Thread(target=rtsp_worker, args=(cam, evt), daemon=True).start()

threading.Thread(target=yolo_worker,        daemon=True).start()
threading.Thread(target=log_writer_worker,  daemon=True).start()
threading.Thread(target=mqtt_sender_worker, daemon=True).start()
threading.Thread(target=db_watcher_worker,  daemon=True).start()
threading.Thread(target=display_worker,     daemon=True).start()
threading.Thread(target=daily_reset_worker, daemon=True).start()

print("✅ Camera AI pipeline started")
print(f"📹 Tổng số camera: {TOTAL_VIDEO}  |  Batch: {BATCH_SIZE}  |  Detect FPS/cam: {DETECT_FPS}")
print(f"🖥️  Display FPS: {DISPLAY_FPS}  |  Grid: {DISPLAY_COLS} cột  |  Ô: {DISPLAY_CELL_W}x{DISPLAY_CELL_H}")
print(f"🔄 DB hot-reload mỗi {DB_POLL_INTERVAL}s  |  Log: {LOG_INTERVAL}s")
print(f"🎥 GStreamer/NVDEC: {'✅ bật' if _GST_AVAILABLE else '⚠️  tắt (dùng FFmpeg)'}")
print("💡 Nhấn 'q' trong cửa sổ display để thoát")

# =============================================================================
# SIGNAL HANDLER
# =============================================================================
running = True

def signal_handler(sig, frame):
    global running
    print("\n🛑 Đang tắt chương trình...")
    running = False

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

while running:
    time.sleep(1)

print("👋 Đã tắt chương trình an toàn.")
time.sleep(0.5)
os._exit(0)
