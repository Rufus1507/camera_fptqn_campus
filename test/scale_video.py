import cv2
import numpy as np
import os
import time
import ctypes

# Preload libcusparseLt.so.0 to fix torch import error on some NVIDIA systems
nvidia_lib = os.path.expanduser("~/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0")
if os.path.exists(nvidia_lib):
    ctypes.CDLL(nvidia_lib)

from ultralytics import YOLO
def main():
    # Khởi tạo mô hình YOLO
    # Sử dụng file weights yolov8s.pt (sẽ tự động tải nếu chưa có)
    print("⏳ Đang tải mô hình YOLOv8s...")
    model = YOLO("yolov8s.pt")
    
    rtsp_url = "rtsp://iot:Iot@1234@10.21.1.21:554/cam/realmonitor?channel=1&subtype=1"
    
    # Thiết lập biến môi trường ưu tiên dùng FFmpeg TCP để stream RTSP mượt hơn
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|analyzeduration;100000|probesize;100000"
    )
    
    print(f"🔗 Đang kết nối tới camera: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Không thể kết nối tới camera!")
        return

    print("✅ Đã kết nối thành công. Nhấn 'Q' để thoát.")

    fail_count = 0
    last_sent = 0.0
    send_interval = 1.0 / 5.0  # Lấy 10 frame/s để xử lý độ trễ

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
        
        # Bỏ qua frame thừa, chỉ lấy tối đa 10 frame/s để tránh trễ video
        now = time.time()
        if now - last_sent < send_interval:
            continue
        last_sent = now
            
        # Resize frame về 640x640 trước khi cắt
        # frame = cv2.resize(frame, (640, 640))
        # h, w = 640, 640
        # cy, cx = 320, 320
        # h, w = frame.shape[:2]
        # # Bỏ phần dư để kích thước chia hết cho 3
        # h = h - (h % 3)
        # w = w - (w % 3)
        # frame = frame[:h, :w]
        # 
        # ch, cw = h // 3, w // 3
        # 
        # # Cắt video thành 9 vùng bằng nhau (lưới 3x3)
        # crops = [
        #     frame[0:ch, 0:cw],       frame[0:ch, cw:2*cw],       frame[0:ch, 2*cw:w],
        #     frame[ch:2*ch, 0:cw],    frame[ch:2*ch, cw:2*cw],    frame[ch:2*ch, 2*cw:w],
        #     frame[2*ch:h, 0:cw],     frame[2*ch:h, cw:2*cw],     frame[2*ch:h, 2*cw:w]
        # ]

        h, w = frame.shape[:2]
        # Bỏ phần dư để kích thước chia hết cho 2
        h = h - (h % 2)
        w = w - (w % 2)
        frame = frame[:h, :w]
        
        ch, cw = h // 2, w // 2
        
        # Cắt video thành 4 vùng bằng nhau (lưới 2x2)
        crops = [
            frame[0:ch, 0:cw],       frame[0:ch, cw:w],
            frame[ch:h, 0:cw],       frame[ch:h, cw:w]
        ]
        
        # Chạy inference trên 4 vùng cùng một lúc (batch size = 4)
        # Lọc class=0 (chỉ lấy "person" - con người)
        results = model.predict(crops, classes=[0], verbose=False, conf=0.25)
        
        # Xử lý kết quả và vẽ bounding box lên từng vùng
        counts = []
        for i, res in enumerate(results):
            # Hàm plot() của Ultralytics sẽ tự động vẽ boxes có sẵn
            annotated_crop = res.plot()
            count = len(res.boxes)
            counts.append(count)
            
            # Ghi số lượng người phát hiện được ở từng vùng góc trái
            cv2.putText(annotated_crop, f"Người: {count}", (5, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            crops[i] = annotated_crop
            
        # # Ghép 9 vùng trở lại thành 1 frame lớn
        # row1 = np.hstack((crops[0], crops[1], crops[2]))
        # row2 = np.hstack((crops[3], crops[4], crops[5]))
        # row3 = np.hstack((crops[6], crops[7], crops[8]))
        # final_frame = np.vstack((row1, row2, row3))
        # 
        # # Vẽ lưới (đường line phân tách 9 màn hình)
        # cv2.line(final_frame, (cw, 0), (cw, h), (0, 255, 255), 2)
        # cv2.line(final_frame, (2*cw, 0), (2*cw, h), (0, 255, 255), 2)
        # cv2.line(final_frame, (0, ch), (w, ch), (0, 255, 255), 2)
        # cv2.line(final_frame, (0, 2*ch), (w, 2*ch), (0, 255, 255), 2)

        # Ghép 4 vùng trở lại thành 1 frame lớn
        row1 = np.hstack((crops[0], crops[1]))
        row2 = np.hstack((crops[2], crops[3]))
        final_frame = np.vstack((row1, row2))
        
        # Vẽ lưới (đường line phân tách 4 màn hình)
        cv2.line(final_frame, (cw, 0), (cw, h), (0, 255, 255), 2)
        cv2.line(final_frame, (0, ch), (w, ch), (0, 255, 255), 2)
        
        # Tổng hợp số người và in lên trên cùng của Frame
        total_people = sum(counts)
        # Khung nền đen mờ để số dễ đọc hơn

        cv2.putText(final_frame, f"TỔNG SỐ NGƯỜI: {total_people}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        # Thu nhỏ khung hình để hiển thị vừa vặn trên màn hình máy tính (giả sử resize về HD 720p)
        display_frame = cv2.resize(final_frame, (1280, 720))

        # # Hiển thị
        # cv2.imshow("9-Split - YOLOv8 Person Counter", display_frame)
        
        # Hiển thị
        cv2.imshow("4-Split - YOLOv8 Person Counter", display_frame)
        
        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
