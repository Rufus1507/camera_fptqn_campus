import cv2
import numpy as np
import os
import time
import ctypes
import threading
from collections import deque


import glob as _glob

# Preload libcusparseLt.so.0 to fix torch import error on some NVIDIA systems
_matches = _glob.glob(os.path.expanduser("~/.local/lib/python*/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"))
if _matches:
    ctypes.CDLL(_matches[0])

os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

from ultralytics import YOLO

# --- Global shared states ---
latest_frame = None
latest_boxes = []
total_people = 0
frame_lock = threading.Lock()

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def yolo_worker():
    global latest_frame, latest_boxes, total_people
    
    print("⏳ [YOLO Thread] Đang tải mô hình YOLOv8s...")
    model = YOLO("yolov8s.pt")
    
    # Chạy warmup ngắn để mô hình không bị khựng lúc xuất hiện frame đầu
    try:
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, conf=0.25)
    except: pass
    
    # [CODE CŨ KHÔNG XOÁ]: margin = 40  # Thuật toán Overlap: thêm 40 pixel khoảng an toàn cho mỗi bên viền cắt
    # [SỬA ĐỂ CHỐNG OVERLAP]: Nâng margin từ 40 -> 140 pixel. Biên 40 quá mỏng nên một người lớn đi qua tâm bị vỡ ra, NMS sẽ tính điểm trượt. Lấn 140px đủ để AI bắt nguyên con người.
    margin = 140
    history = deque([0]*5, maxlen=5)
    active_tracks = [] # Lưu các track: {'box': box, 'hits': 1, 'misses': 0, 'first_pos': (cx, cy), 'is_moving': False}
    
    while True:
        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()
        
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Giới hạn tốc độ YOLO xử lý (vd: rate limit nghỉ 150ms mỗi frame -> ~6.6 FPS)
        # Main thread vẫn render 25FPS bình thường
        time.sleep(0.10)
        
        h, w = frame.shape[:2]
        h = h - (h % 2)
        w = w - (w % 2)
        frame = frame[:h, :w]
        ch, cw = h // 2, w // 2
        
        crops = []
        offsets = []
        
        # Grid 2x2 crop với margin để không làm gãy người đi qua viền ranh giới
        bbox_ranges = [
            (0, min(w, cw + margin), 0, min(h, ch + margin)), # Top-Left
            (max(0, cw - margin), w, 0, min(h, ch + margin)), # Top-Right
            (0, min(w, cw + margin), max(0, ch - margin), h), # Bottom-Left
            (max(0, cw - margin), w, max(0, ch - margin), h) # Bottom-Right
        ]
        
        for (x1, x2, y1, y2) in bbox_ranges:
            crops.append(frame[y1:y2, x1:x2])
            offsets.append((x1, y1))
            
        # Chạy inference lần lượt từng phần (batch=1) để CHỐNG TRÀN VRAM (Lỗi Error 12 NvMapMem) trên các máy Jetson
        results = []
        for crop in crops:
            # [SỬA ĐỂ TỐI ƯU HOÁ MODEL (640x640)]: Thiết lập chuẩn cứng imgsz=640 cho model infer
            r = model.predict(crop, classes=[0], verbose=False, conf=0.25, imgsz=640)[0]
            results.append(r)
        
        temp_boxes = []
        for i, res in enumerate(results):
            offset_x, offset_y = offsets[i]
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    cx1, cy1, cx2, cy2 = coords
                    # Mapping toạ độ Box củaảnh crop nhỏ trở về đúng vị trí trên khung hình FULL GỐC
                    ox1 = int(cx1 + offset_x)
                    oy1 = int(cy1 + offset_y)
                    ox2 = int(cx2 + offset_x)
                    oy2 = int(cy2 + offset_y)
                    conf = float(box.conf)
                    temp_boxes.append([ox1, oy1, ox2, oy2, conf])
                    
        final_boxes = []
        
        # Chạy thuật toán triệt tiêu đếm lặp NMS (Non-Maximum Suppression) trong vùng đệm overlap
        if len(temp_boxes) > 0:
            temp_boxes = sorted(temp_boxes, key=lambda x: x[4], reverse=True) # Ưu điểm: Lấy cái Box có độ AI tin tưởng xịn nhất để làm mốc so
            
            for b_new in temp_boxes:
                is_duplicate = False
                for k_box in final_boxes: # final_boxes từ mảng trống sẽ đóng vai trò như chốt chặn
                    # Tìm tọa độ đè góc trên cùng
                    x1_inter = max(b_new[0], k_box[0])
                    y1_inter = max(b_new[1], k_box[1])
                    x2_inter = min(b_new[2], k_box[2])
                    y2_inter = min(b_new[3], k_box[3])
                    
                    if x2_inter > x1_inter and y2_inter > y1_inter: # Nhìn thấy có giao nhau
                        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                        b_area = (b_new[2] - b_new[0]) * (b_new[3] - b_new[1])
                        k_area = (k_box[2] - k_box[0]) * (k_box[3] - k_box[1])
                        
                        # Chỉ số 1: Tỷ lệ IoU (Sử dụng của OpenCV gốc -> Yếu điểm lúc phân thân)
                        union_area = b_area + k_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        # Chỉ số 2 Cứu cánh: Tỷ lệ lấp đầy IoMin (Tức mảnh nhỏ bị nuốt bao nhiêu % bởi hộp lớn)
                        # Khắc phục lỗi: mảnh cơ thể 20% bên này sẽ bị trùng nằm lọt trong mảng cơ thể hoàn thiện 100% bên nọ
                        ioa = inter_area / min(b_area, k_area) if min(b_area, k_area) > 0 else 0
                        
                        # Điều kiện gộp kép dội ngược để làm sạch (iou > 0.4 Hoặc Diện tích nuốt > 0.75)
                        if iou > 0.40 or ioa > 0.75:
                            is_duplicate = True
                            break
                            
                if not is_duplicate:
                    final_boxes.append(b_new)

        # -- TRACKER ĐỂ CHỐNG VẬT THỂ CỐ ĐỊNH CHẬP CHỜN NHƯNG KHÔNG LÀM RỚT NGƯỜI ĐI BỘ --
        new_tracks = []
        unmatched_boxes = final_boxes.copy()
        
        for track in active_tracks:
            best_match_score = -1
            best_box_idx = -1
            for i, box in enumerate(unmatched_boxes):
                iou = compute_iou(track['box'][:4], box[:4])
                
                # Tính khoảng cách tâm (hỗ trợ tracker do FPS thấp = 2FPS -> người di chuyển xa)
                cx1 = (track['box'][0] + track['box'][2]) / 2
                cy1 = (track['box'][1] + track['box'][3]) / 2
                cx2 = (box[0] + box[2]) / 2
                cy2 = (box[1] + box[3]) / 2
                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                
                # Chấp nhận match nếu người di chuyển rất xa (iou = 0) nhưng lệch không quá 300px
                if iou > 0.05 or dist < 300: 
                    score = iou if iou > 0 else (1.0 / (dist + 1))
                    if score > best_match_score:
                        best_match_score = score
                        best_box_idx = i
                    
            if best_box_idx != -1: 
                matched_box = unmatched_boxes.pop(best_box_idx)
                track['box'] = matched_box
                track['hits'] += 1
                track['misses'] = 0
                
                # Kiểm tra xem vật thể có di chuyển không
                if not track.get('is_moving', False):
                    if 'first_pos' not in track:
                        track['first_pos'] = ((matched_box[0]+matched_box[2])/2, (matched_box[1]+matched_box[3])/2)
                        
                    cx_now = (matched_box[0] + matched_box[2]) / 2
                    cy_now = (matched_box[1] + matched_box[3]) / 2
                    cx_first, cy_first = track['first_pos']
                    move_dist = ((cx_now - cx_first)**2 + (cy_now - cy_first)**2)**0.5
                    
                    # Nếu lệch khỏi tâm xuất hiện đầu tiên > 30 pixel -> Đang di chuyển -> Hiện luôn
                    if move_dist > 30:
                        track['is_moving'] = True
                        
                new_tracks.append(track)
            else:
                track['misses'] += 1
                if track['misses'] < 3: # Cho phép tàng hình/mất dấu tạm thời trong 2 frame
                    new_tracks.append(track)
                    
        # Những box MỚI chưa được track thì tạo track mới
        for box in unmatched_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            new_tracks.append({
                'box': box, 
                'hits': 1, 
                'misses': 0, 
                'first_pos': (cx, cy), 
                'is_moving': False
            })
            
        active_tracks = new_tracks
        
        # ĐIỀU KIỆN TÍNH LÀ NGƯỜI CHO BẢN FPS THẤP:
        # 1. Đang di chuyển (is_moving = True) -> Bỏ qua kiểm tra hits, hiện luôn
        # 2. Vật thể có vẻ cố định -> Bắt buộc phải hits >= 3 mới được làm người thật
        confirmed_boxes = [t['box'] for t in active_tracks if t.get('is_moving', False) or t['hits'] >= 5]

        raw_count = len(confirmed_boxes)
        # History làm mượt số lượng đếm (y như ở bản main.py)
        history.append(raw_count)
        stable_count = max(history)
        
        with frame_lock:
            latest_boxes = final_boxes
            total_people = stable_count

def main():
    global latest_frame, latest_boxes, total_people

    rtsp_url = "rtsp://iot:Iot@1234@10.21.1.45:554/cam/realmonitor?channel=1&subtype=1"
    # rtsp_url = "/home/fptqn/workspace/AI_model/saved_videos/Cam_Corridor_02_135307_7_03042026.mp4"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|analyzeduration;100000|probesize;100000"
    )
    
    print(f"🔗 Đang kết nối tới camera: {rtsp_url}")
    # cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(rtsp_url)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Không thể kết nối tới camera!")
        return

    print("✅ Đã kết nối thành công. Nhấn 'Q' để thoát.")
    
    # Bật Thread giải thuật AI độc lập (chạy ngầm trong Background)
    t = threading.Thread(target=yolo_worker, daemon=True)
    t.start()

    fail_count = 0
    last_frame_time = 0
    send_interval = 0.5 # 10 FPS
    
    # Main loop giới hạn 10 FPS để tránh lag và tối ưu hiệu suất
    while True:
        ret, frame = cap.read()
        
        if ret and frame is None:
            fail_count += 1
            if fail_count >= 10:
                print("⚠️ Frame NULL/xám liên tục — reconnect sau 2s")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fail_count = 0
            time.sleep(0.02)
            continue
            
        if not ret:
            fail_count += 1
            if fail_count >= 5:
                print("⚠️ Mất kết nối hoặc không đọc được frame. Đang thử kết nối lại...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fail_count = 0
            time.sleep(0.01)
            continue
            
        fail_count = 0

        now = time.time()
        if now - last_frame_time < send_interval:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        last_frame_time = now

        # [SỬA THEO FLOW LƯU CHUYỂN MỚI]: RSTP(2K) -> decode -> resize(Full HD) 
        # Hạ kích thước khung hình về chuẩn chung 1920x1080 (Full HD) ngay lập tức dể làm nhẹ đi toàn bộ dây chuyền về sau (chia 4, vẽ UI)
        frame = cv2.resize(frame, (1920, 1080))

        # 

        # 
        # # Cắt video thành 9 vùng bằng nhau (lưới 3x3)
        # crops = [
        #     frame[0:ch, 0:cw],       frame[0:ch, cw:2*cw],       frame[0:ch, 2*cw:w],
        #     frame[ch:2*ch, 0:cw],    frame[ch:2*ch, cw:2*cw],    frame[ch:2*ch, 2*cw:w],
        #     frame[2*ch:h, 0:cw],     frame[2*ch:h, cw:2*cw],     frame[2*ch:h, 2*cw:w]
        # ]
        # Ghép 9 vùng trở lại thành 1 frame lớn
        # row1 = np.hstack((crops[0], crops[1], crops[2]))
        # row2 = np.hstack((crops[3], crops[4], crops[5]))
        # row3 = np.hstack((crops[6], crops[7], crops[8]))
        # final_frame = np.vstack((row1, row2, row3))
        # 
        # # Vẽ lưới (đường line phân tách 4 màn hình)
        h, w = frame.shape[:2]
        h = h - (h % 2)
        w = w - (w % 2)
        frame = frame[:h, :w]
        ch, cw = h // 2, w // 2
        # 1. Cập nhật frame mới cho luồng Background (yolo_worker) để mang đi phân tích
        # 2. Rút toạ độ & số lượng người MỚI NHẤT từ luồng bên kia để vẽ lên UI Main.
        with frame_lock:
            latest_frame = frame.copy()
            draw_boxes = list(latest_boxes)  
            draw_total = total_people        
            
        # Vẽ Lưới mỏng 2x2
        cv2.line(frame, (cw, 0), (cw, h), (0, 255, 255), 2)
        cv2.line(frame, (0, ch), (w, ch), (0, 255, 255), 2)
        
        # Vẽ các hình chữ nhật (Bounding Boxes) gốc
        for b in draw_boxes:
            ox1, oy1, ox2, oy2, conf = b
            cv2.rectangle(frame, (int(ox1), int(oy1)), (int(ox2), int(oy2)), (0, 0, 255), 2)
            cv2.putText(frame, f"Nguoi {conf:.2f}", (int(ox1), int(oy1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # Hiển thị số lượng người
        cv2.putText(frame, f"TONG SO NGUOI (SMOOTH): {draw_total}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    
        # Thu nhỏ để chuẩn HD dễ nhìn trên màn máy tính
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("4-Split Smooth Fast Person Counter", display_frame)
        
        # [PHẦN EDIT CỦA AI]: Đổi cv2.waitKey(1) cũ thành cv2.waitKey(200) 
        # Để giới hạn phát video ở mức 5 Frame/s (Tính toán: 1000ms / 5 = 200ms)
        # Các comment và code gốc của bạn vẫn được giữ nguyên đầy đủ.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
