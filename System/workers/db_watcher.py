import time
import threading
from urllib.parse import urlparse
from queue import Queue
import shared.state as state
from config.settings import DB_POLL_INTERVAL, DB_PATH, MQTT_TOPIC, QUEUE_PER_CAM
from database.db import load_cameras, load_crop_configs
from workers.rtsp_worker import rtsp_worker
from tracking.smart_tracker import SmartTracker
from utils.logger import get_system_logger, get_cam_logger, remove_cam_logger


def db_watcher_worker():
    while True:
        time.sleep(DB_POLL_INTERVAL)
        sys_log = get_system_logger()

        new_cameras   = load_cameras(DB_PATH)
        new_ids       = {c["id"] for c in new_cameras}
        new_cam_map   = {c["id"]: c for c in new_cameras}
        new_crop_cfgs = load_crop_configs(DB_PATH)

        changed_cfgs: dict = {}
        invalidate_roi: list = []
        for cid, new_cfg in new_crop_cfgs.items():
            old_cfg = state.cam_crop_configs.get(cid)
            if old_cfg is None or old_cfg != new_cfg:
                changed_cfgs[cid] = new_cfg
                if old_cfg is not None and old_cfg.roi_polygons != new_cfg.roi_polygons:
                    invalidate_roi.append(cid)

        with state.state_lock:
            old_ids = set(state.CAM_IDS)

        added   = new_ids - old_ids
        removed = old_ids - new_ids

        need_lock = bool(changed_cfgs) or bool(new_cam_map)
        if need_lock:
            with state.state_lock:
                if changed_cfgs:
                    for cid in invalidate_roi:
                        state.cam_roi_mask_shape.pop(cid, None)
                        state.cam_roi_masks.pop(cid, None)
                        get_cam_logger(cid).info("ROI polygon changed, rebuilding mask")
                    state.cam_crop_configs.update(changed_cfgs)

                for cid, cam in new_cam_map.items():
                    state.cam_display_enabled[cid] = cam.get("enable_cam", 1)

                if added or removed:
                    state.CAM_IDS     = sorted(new_ids)
                    state.TOTAL_VIDEO = len(state.CAM_IDS)

        # Handle removed cameras
        for cid in removed:
            evt = state.cam_stop_events.pop(cid, None)
            if evt:
                evt.set()
            state.cam_trackers.pop(cid, None)
            with state.state_lock:
                state.camera_state.pop(cid, None)
                state.frame_queues.pop(cid, None)
                state.detect_timestamps.pop(cid, None)
                state.last_detect_time.pop(cid, None)
                state.cam_topic_map.pop(cid, None)
                state.cam_ip_map.pop(cid, None)
                state.cam_display_enabled.pop(cid, None)
                state.cam_crop_configs.pop(cid, None)
                state.cam_roi_masks.pop(cid, None)
                state.cam_roi_mask_shape.pop(cid, None)
            with state.display_lock:
                state.display_frames.pop(cid, None)
            remove_cam_logger(cid)
            sys_log.info(f"Camera {cid} removed/disabled")

        # Handle newly added cameras
        for cid in added:
            cam = new_cam_map[cid]
            with state.state_lock:
                state.frame_queues[cid]      = Queue(maxsize=QUEUE_PER_CAM)
                state.detect_timestamps[cid] = []
                state.camera_state[cid]      = {
                    "timestamp": int(time.time()),
                    "fps": 0.0, "person_ids": [], "is_night": "0"
                }
                state.cam_topic_map[cid] = cam.get("mqtt_topic", MQTT_TOPIC)
                state.cam_ip_map[cid]   = urlparse(cam.get("rtsp", "")).hostname or str(cid)

            state.cam_trackers[cid] = SmartTracker()
            stop_evt = threading.Event()
            state.cam_stop_events[cid] = stop_evt
            threading.Thread(target=rtsp_worker, args=(cam, stop_evt), daemon=True).start()
            sys_log.info(f"Camera {cid} added, RTSP thread started (topic={cam.get('mqtt_topic', MQTT_TOPIC)})")

        if added or removed:
            sys_log.info(f"Active cameras: {len(new_ids)}")

        # Detect IP / RTSP URL changes for existing cameras
        existing_ids = new_ids - added
        for cid in existing_ids:
            cam = new_cam_map.get(cid)
            if cam is None:
                continue
            new_rtsp = cam.get("rtsp", "")
            new_ip   = urlparse(new_rtsp).hostname or str(cid)
            old_ip   = state.cam_ip_map.get(cid)

            if old_ip is not None and old_ip != new_ip:
                sys_log.info(f"Camera {cid} IP changed ({old_ip} -> {new_ip}), restarting RTSP thread")

                old_evt = state.cam_stop_events.pop(cid, None)
                if old_evt:
                    old_evt.set()

                with state.state_lock:
                    state.frame_queues[cid]  = Queue(maxsize=QUEUE_PER_CAM)
                    state.cam_ip_map[cid]    = new_ip
                    state.cam_topic_map[cid] = cam.get("mqtt_topic", MQTT_TOPIC)

                new_evt = threading.Event()
                state.cam_stop_events[cid] = new_evt
                threading.Thread(
                    target=rtsp_worker,
                    args=(cam, new_evt),
                    daemon=True,
                ).start()


def daily_reset_worker():
    """Reset person counters every 2 hours."""
    last_reset_time = time.time()
    while True:
        time.sleep(10)
        if time.time() - last_reset_time >= 2 * 3600:
            sys_log = get_system_logger()
            sys_log.info("Resetting person counters (2-hour cycle)")
            from shared.state import global_tracker_lock, global_tracker
            with global_tracker_lock:
                if global_tracker is not None:
                    global_tracker.reset()
                for cid, tracker in state.cam_trackers.items():
                    tracker.reset()
            last_reset_time = time.time()
            sys_log.info("Person counters reset done")
