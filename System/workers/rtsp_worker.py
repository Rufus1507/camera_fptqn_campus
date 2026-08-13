import os
import cv2
import time
import threading
from urllib.parse import urlparse
from queue import Empty
import shared.state as state
from config.settings import RTSP_RETRY_DELAY, QUEUE_PER_CAM, DETECT_FPS, MAX_FRAME_W, MAX_FRAME_H
from utils.brightness import _is_gray_frame
from utils.logger import get_cam_logger

def rtsp_worker(cam: dict, stop_event: threading.Event):
    cam_id = cam["id"]
    url    = cam["rtsp"]
    log    = get_cam_logger(cam_id)

    while not stop_event.is_set():
        use_gst = False
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|analyzeduration;100000|probesize;100000"
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            log.warning(f"RTSP connect failed, retry in {RTSP_RETRY_DELAY}s | url={url}")
            with state.state_lock:
                if cam_id in state.camera_state:
                    state.camera_state[cam_id]["fps"] = -1.0
            cap.release()
            stop_event.wait(timeout=RTSP_RETRY_DELAY)
            continue

        backend = "GStreamer/NVDEC" if use_gst else "FFmpeg"
        log.info(f"RTSP connected [{backend}]")
        fail_count  = 0
        first_frame = True
        last_put_time = 0.0

        while not stop_event.is_set():
            ret, frame = cap.read()

            if ret and (frame is None or (use_gst and _is_gray_frame(frame))):
                fail_count += 1
                if fail_count == 1 and use_gst:
                    log.warning("GStreamer returned gray/NULL frame, fallback FFmpeg")
                    cap.release()
                    use_gst = False
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        "rtsp_transport;tcp|analyzeduration;100000|probesize;100000"
                    )
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if cap.isOpened():
                        log.info("Reconnected via FFmpeg/TCP")
                        fail_count = 0
                    else:
                        fail_count = 10
                    continue
                if fail_count >= 10:
                    log.warning(f"Repeated NULL/gray frames, reconnect in {RTSP_RETRY_DELAY}s")
                    with state.state_lock:
                        if cam_id in state.camera_state:
                            state.camera_state[cam_id]["fps"] = -1.0
                    cap.release()
                    stop_event.wait(timeout=RTSP_RETRY_DELAY)
                    break
                time.sleep(0.02)
                continue

            if not ret:
                fail_count += 1
                if fail_count >= 5:
                    log.warning(f"RTSP stream lost, reconnect in {RTSP_RETRY_DELAY}s")
                    with state.state_lock:
                        if cam_id in state.camera_state:
                            state.camera_state[cam_id]["fps"] = -1.0
                    cap.release()
                    stop_event.wait(timeout=RTSP_RETRY_DELAY)
                    break
                time.sleep(0.01)
                continue

            if first_frame:
                first_frame = False

            fail_count = 0

            now = time.time()
            elapsed = now - last_put_time
            if elapsed < 1.0 / DETECT_FPS:
                if not str(url).startswith(("rtsp://", "http://")):
                    time.sleep((1.0 / DETECT_FPS) - elapsed)
                    last_put_time = time.time()
                else:
                    continue
            else:
                last_put_time = now

            # Cap to a max working size instead of forcing an exact 1920x1080:
            # a fixed-size resize both upscales cameras with a smaller native
            # resolution for no benefit (both display_worker's grid cells and
            # YOLOEngine's imgsz=640 letterboxing resize this back down anyway)
            # and distorts aspect ratio for any camera that isn't exactly 16:9
            # (e.g. a 704x576 stream forced to 1920x1080 gets stretched 2.73x
            # horizontally vs 1.88x vertically). Downscale only if needed, and
            # always preserve aspect ratio.
            fh, fw = frame.shape[:2]
            if fw > MAX_FRAME_W or fh > MAX_FRAME_H:
                scale = min(MAX_FRAME_W / fw, MAX_FRAME_H / fh)
                frame_resized = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))))
            else:
                frame_resized = frame

            with state.display_lock:
                state.display_frames[cam_id] = frame_resized.copy()

            q = state.frame_queues.get(cam_id)
            if q is not None:
                if q.full():
                    try: q.get_nowait()
                    except Empty: pass
                q.put_nowait(frame_resized)

        cap.release()
    log.info("RTSP thread stopped")
