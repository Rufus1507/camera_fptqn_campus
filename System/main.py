import os
import sys
import signal
import time
import threading
import datetime
import traceback

# ── Preload libcusparseLt.so.0 for Jetson Orin (BEFORE importing torch) ──────
from utils.gst_utils import preload_libraries, check_gstreamer_support, suppress_ffmpeg_output
preload_libraries()
suppress_ffmpeg_output()

from database.db import load_cameras, load_zone_map, load_adjacent_zones, load_crop_configs
from config.settings import (
    DB_PATH, DB_POLL_INTERVAL, DETECT_FPS, DISPLAY_FPS, BATCH_SIZE,
    DISPLAY_COLS, DISPLAY_CELL_W, DISPLAY_CELL_H, DEVICE, MODEL_PATH,
    MODEL_MAX_BATCH, API_URL, API_SYNC_INTERVAL, LOG_DIR, STARTUP_WARNINGS
)
from utils.logger import init_logging, get_system_logger
import shared.state as state
from tracking.global_tracker import GlobalTracker
from tracking.smart_tracker import SmartTracker

from core.yolo_engine import YOLOEngine
from core.reid_engine import ReIDEngine

from workers.rtsp_worker import rtsp_worker
from workers.yolo_worker import yolo_worker
from workers.display_worker import display_worker
from workers.mqtt_worker import mqtt_sender_worker
from workers.db_watcher import db_watcher_worker, daily_reset_worker
from workers.api_sync_worker import api_sync_worker
from queue import Queue
from config.settings import QUEUE_PER_CAM


def _log_uncaught(exc_type, exc_value, exc_tb, thread_name: str) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        return  # SIGINT is handled via signal_handler; this is defensive only
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    get_system_logger().critical(f"Uncaught exception in {thread_name}:\n{tb_text}")


def _sys_excepthook(exc_type, exc_value, exc_tb) -> None:
    _log_uncaught(exc_type, exc_value, exc_tb, "MainThread")


def _threading_excepthook(args) -> None:
    # threading.excepthook already ignores SystemExit before calling us —
    # this repeats the check defensively, it's not load-bearing.
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread else "UnknownThread"
    _log_uncaught(args.exc_type, args.exc_value, args.exc_traceback, name)


def main():
    init_logging(LOG_DIR)
    sys.excepthook = _sys_excepthook
    threading.excepthook = _threading_excepthook
    sys_log = get_system_logger()

    for level, msg in STARTUP_WARNINGS:
        sys_log.log(level, msg)

    sys_log.info("AI System initializing")
    
    _GST_AVAILABLE = check_gstreamer_support()
    sys_log.info(f"GStreamer/NVDEC: {'available' if _GST_AVAILABLE else 'unavailable, fallback FFmpeg'}")

    # Sync DB with API before loading cameras
    from database.db import sync_cameras_from_api
    sys_log.info(f"Initial DB sync from API: {API_URL}")
    sync_cameras_from_api(API_URL, DB_PATH)

    # Initial DB load
    state.CAMERAS        = load_cameras(DB_PATH)
    state.CAM_IDS        = [c["id"] for c in state.CAMERAS]
    state.TOTAL_VIDEO    = len(state.CAM_IDS)
    state.cam_zone_map   = load_zone_map(DB_PATH)
    state.adjacent_zones = load_adjacent_zones(DB_PATH)
    state.cam_crop_configs = load_crop_configs(DB_PATH)
    sys_log.info(f"Zone map: {state.cam_zone_map}")
    sys_log.info(f"Adjacent zones: {state.adjacent_zones}")
    sys_log.info(f"Crop configs: { {k: (v.grid_rows, v.grid_cols, v.overlap_ratio) for k, v in state.cam_crop_configs.items()} }")

    state.global_tracker = GlobalTracker(sim_thresh=0.55, max_lost=30.0)
    
    from urllib.parse import urlparse
    state.cam_topic_map = {c["id"]: c["mqtt_topic"] for c in state.CAMERAS}
    state.cam_ip_map = {c["id"]: (urlparse(c["rtsp"]).hostname or str(c["id"])) for c in state.CAMERAS}

    state.camera_state = {
        cid: {"timestamp": state.init_time, "fps": 0.0, "person_ids": [], "is_night": "0", "boxes": []}
        for cid in state.CAM_IDS
    }
    state.frame_queues = {cid: Queue(maxsize=QUEUE_PER_CAM) for cid in state.CAM_IDS}
    for cid in state.CAM_IDS:
        state.cam_trackers[cid] = SmartTracker()
        evt = threading.Event()
        state.cam_stop_events[cid] = evt
        cam_info = next(c for c in state.CAMERAS if c["id"] == cid)
        threading.Thread(target=rtsp_worker, args=(cam_info, evt), daemon=True).start()

    with state.model_init_lock:
        yolo_engine = YOLOEngine(MODEL_PATH, DEVICE, max_batch=MODEL_MAX_BATCH)
        reid_engine = ReIDEngine(DEVICE)

    # Start Workers
    yolo_worker(yolo_engine, reid_engine)
    threading.Thread(target=mqtt_sender_worker, daemon=True).start()
    threading.Thread(target=db_watcher_worker,  daemon=True).start()
    threading.Thread(target=display_worker,     daemon=True).start()
    threading.Thread(target=daily_reset_worker, daemon=True).start()
    threading.Thread(target=api_sync_worker,    daemon=True).start()

    sys_log.info("Camera AI pipeline started")
    sys_log.info(f"Cameras: {state.TOTAL_VIDEO} | Batch: {BATCH_SIZE} | Detect FPS: {DETECT_FPS}")
    sys_log.info(f"Display FPS: {DISPLAY_FPS} | Grid cols: {DISPLAY_COLS} | Cell: {DISPLAY_CELL_W}x{DISPLAY_CELL_H}")
    sys_log.info(f"DB hot-reload every {DB_POLL_INTERVAL}s")

    # ──────────────────────────────────────────────────────────────────────────
    # 🛡️  LớP 2: GPU Health Watchdog (kiểm tra GPU sống mỗi 2 giây)
    # ──────────────────────────────────────────────────────────────────────────
    import torch

    def gpu_healthy() -> bool:
        """Trả về True nếu GPU hoạt động bình thường."""
        try:
            ok = torch.cuda.is_available() and torch.cuda.memory_allocated() >= 0
            return ok
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # 🔍 YOLO Hang Detector: không có detect trong YOLO_HANG_TIMEOUT giây
    # ──────────────────────────────────────────────────────────────────────────
    YOLO_HANG_TIMEOUT = 30   # giây — điều chỉnh nếu cần
    GPU_FAIL_MAX      = 3    # số lần liên tiếp GPU lỗi trước khi restart

    # ──────────────────────────────────────────────────────────────────────────
    # 🧹 Graceful Shutdown
    # ──────────────────────────────────────────────────────────────────────────
    def graceful_shutdown(exit_code: int, reason: str) -> None:
        """Safe cleanup: stop RTSP threads, purge GPU, flush logs, then exit."""
        _sl = get_system_logger()
        _sl.info(f"[{reason}] Graceful shutdown (exit_code={exit_code})")

        # Step 1: Signal all RTSP threads to stop
        try:
            with state.state_lock:
                stop_events = list(state.cam_stop_events.values())
            for evt in stop_events:
                evt.set()
            _sl.info(f"Sent stop event to {len(stop_events)} RTSP thread(s)")
        except Exception as e:
            _sl.warning(f"Error stopping RTSP threads: {e}")

        # Step 2: Purge GPU memory cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _sl.info("GPU cache purged")
        except Exception as e:
            _sl.warning(f"Failed to purge GPU cache: {e}")

        # Step 3: Flush
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        _sl.info(f"Shutdown complete, exit code {exit_code}")

        # Step 4: Safety-net timer: force os._exit() if sys.exit hangs after 5s
        def _force_exit():
            _sl.warning("sys.exit stuck after 5s, forcing os._exit()")
            os._exit(exit_code)

        timer = threading.Timer(5.0, _force_exit)
        timer.daemon = True
        timer.start()
        sys.exit(exit_code)

    # ──────────────────────────────────────────────────────────────────────────
    # 🕐 Midnight Restart: thoát lúc 0h để watchdog (run.sh/systemd) khởi lại
    # ──────────────────────────────────────────────────────────────────────────
    def seconds_until_midnight() -> float:
        now_dt  = datetime.datetime.now()
        midnight = now_dt.replace(hour=0, minute=0, second=0, microsecond=0) \
                   + datetime.timedelta(days=1)
        return (midnight - now_dt).total_seconds()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        get_system_logger().info("SIGINT/SIGTERM received, shutting down")
        running = False
        state.shutdown_exit_code = 0
        state.shutdown_event.set()

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    gpu_fail_count   = 0
    secs_to_midnight = seconds_until_midnight()
    sys_log.info(f"Midnight restart in {secs_to_midnight / 3600:.1f} hours")

    while running:
        # ─ Đợi tối đa 2s, sớ thức sớm nếu worker yêu cầu shutdown ────────
        if state.shutdown_event.wait(timeout=2):
            reason = "Worker-requested shutdown"
            graceful_shutdown(state.shutdown_exit_code, reason)
            return  # không bao giờ tới đây (graceful_shutdown luôn thoát)

        secs_to_midnight -= 2

        # ─ [1] GPU Watchdog ──────────────────────────────────────────────
        if not gpu_healthy():
            gpu_fail_count += 1
            sys_log.warning(f"GPU unhealthy (attempt {gpu_fail_count}/{GPU_FAIL_MAX})")
            if gpu_fail_count >= GPU_FAIL_MAX:
                sys_log.error("GPU died, initiating graceful shutdown")
                graceful_shutdown(1, "GPU Watchdog")
                return
        else:
            gpu_fail_count = 0

        # ─ [2] YOLO Hang Detector ────────────────────────────────────
        with state.state_lock:
            last_times = list(state.last_detect_time.values())

        if last_times:
            latest = max(last_times)
            if time.time() - latest > YOLO_HANG_TIMEOUT:
                sys_log.error(f"YOLO hung >{YOLO_HANG_TIMEOUT}s, initiating graceful shutdown")
                graceful_shutdown(1, "YOLO Hang Detector")
                return

        # ─ [3] Midnight Restart ───────────────────────────────────────
        if secs_to_midnight <= 0:
            sys_log.info("Midnight restart, watchdog will restart process")
            graceful_shutdown(0, "Midnight Restart")
            return

    # SIGINT/SIGTERM đã set shutdown_event rồi, nhưng không chạy quá shutdown_event.wait
    # (vì running = False trước). Xử lý ở đây để đảm bảo clean exit.
    graceful_shutdown(state.shutdown_exit_code, "Signal Handler")

if __name__ == "__main__":
    main()
