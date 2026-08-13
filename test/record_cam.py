import os
import time
import datetime
import sqlite3
import threading
import cv2
import signal
from urllib.parse import urlparse

DB_PATH = "cameras.db"
OUTPUT_DIR = "recorded_1fps"
CAPTURE_INTERVAL = 1.0  # 1 frame per second
OUTPUT_FPS = 1.0        # FPS của file video đầu ra
VIDEO_DURATION = 24 * 3600  # 24 giờ (được tính bằng giây)

os.makedirs(OUTPUT_DIR, exist_ok=True)
stop_event = threading.Event()

def load_cameras():
    # Sử dụng danh sách 6 IP tĩnh cố định thay vì đọc từ cameras.db
    return [
        {"id": 1, "name": "Cam_Lobby_01", "rtsp": "rtsp://iot:Iot@1234@10.21.1.45:554/cam/realmonitor?channel=1&subtype=1"},
        {"id": 2, "name": "Cam_Corridor_01", "rtsp": "rtsp://iot:Iot@1234@10.21.1.20:554/cam/realmonitor?channel=1&subtype=1"},
        {"id": 3, "name": "Cam_Corridor_02", "rtsp": "rtsp://iot:Iot@1234@10.21.1.21:554/cam/realmonitor?channel=1&subtype=1"},
        {"id": 4, "name": "Cam_Meeting_01", "rtsp": "rtsp://iot:Iot@1234@10.21.1.46:554/cam/realmonitor?channel=1&subtype=1"},
        {"id": 5, "name": "Cam_Corridor_03", "rtsp": "rtsp://iot:Iot@1234@10.21.1.22:554/cam/realmonitor?channel=1&subtype=1"},
        {"id": 6, "name": "Cam_Corridor_04", "rtsp": "rtsp://iot:Iot@1234@10.21.1.23:554/cam/realmonitor?channel=1&subtype=1"}
    ]

def safe_name(name):
    # Loại bỏ ký tự đặc biệt để làm tên file hợp lệ
    return "".join([c for c in name if c.isalnum() or c in "-_ "]).strip()

def camera_worker(cam):
    cam_id = cam["id"]
    cam_name = cam["name"]
    rtsp_url = cam["rtsp"]

    print(f"[{cam_name}] Bắt đầu tiến trình ghi hình...")
    
    # Ưu tiên giao thức TCP cho RTSP để ổn định hơn
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    
    while not stop_event.is_set():
        start_time = time.time()
        now_dt = datetime.datetime.now()
        date_str = now_dt.strftime("%Y%m%d_%H%M%S")
        filename = f"cam{cam_id}_{safe_name(cam_name)}_{date_str}.mp4"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        out = None
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        last_write_time = 0
        
        while time.time() - start_time < VIDEO_DURATION and not stop_event.is_set():
            if not cap.isOpened():
                print(f"[{cam_name}] Mất kết nối, đang thử kết nối lại...")
                cap.release()
                for _ in range(5):
                    if stop_event.is_set(): break
                    time.sleep(1)
                
                if stop_event.is_set():
                    break
                
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                continue
                
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                time.sleep(1)
                continue
                
            if out is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filepath, fourcc, OUTPUT_FPS, (w, h))
                print(f"[{cam_name}] Bắt đầu xuất video 24h: {filepath}")
                
            now_t = time.time()
            if now_t - last_write_time >= CAPTURE_INTERVAL:
                out.write(frame)
                last_write_time = now_t
                
        # Kết thúc 1 đoạn block 24h (hoặc app đang bị kill)
        if out is not None:
            out.release()
        cap.release()
        
        if stop_event.is_set():
            print(f"[{cam_name}] Đã đóng file video do dừng chương trình.")
            break
        else:
            print(f"[{cam_name}] Hoàn tất phiên 24h, sẽ mở file mới tiếp theo.")

def signal_handler(sig, frame):
    print("\nNhận lệnh tắt, đang đóng an toàn các file video (có thể mất vài giây)...")
    stop_event.set()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    cameras = load_cameras()
    if not cameras:
        print("Không tìm thấy camera online nào trong DB!")
        exit()
        
    print(f"Đã tìm thấy {len(cameras)} camera online. Đang khởi tạo ghi hình...")
    
    threads = []
    for cam in cameras:
        t = threading.Thread(target=camera_worker, args=(cam,), daemon=False)
        t.start()
        threads.append(t)
        
    print("Hệ thống đang chạy. Nhấn Ctrl+C để dừng.")
    
    # Wait cho các thread con đóng
    for t in threads:
        t.join()
        
    print("Hoàn tất đóng chương trình an toàn.")
