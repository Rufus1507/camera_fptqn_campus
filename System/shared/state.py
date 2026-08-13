import time
import threading
from queue import Queue
from collections import defaultdict
from typing import Dict, List, Any, Set

# ── Graceful Shutdown ──────────────────────────────────────────────────────────
# Worker threads set shutdown_event (+ ghi shutdown_exit_code) thay vì os._exit().
# Main loop phát hiện, gọi graceful_shutdown(), rồi sys.exit(exit_code).
shutdown_event: threading.Event = threading.Event()
shutdown_exit_code: int = 0          # 0 = clean restart, 1 = error restart

# Sync primitives
state_lock = threading.Lock()
display_lock = threading.Lock()
model_init_lock = threading.Lock()
global_tracker_lock = threading.Lock()

# App State variables (Initialized in main.py or db_watcher.py)
CAMERAS: List[Dict[str, Any]] = []
CAM_IDS: List[int] = []
TOTAL_VIDEO: int = 0
init_time: int = int(time.time())

# Zone mappings (loaded from DB at startup)
cam_zone_map: Dict[int, int] = {}    # {camera_id: zone_id}
adjacent_zones: Dict[int, Set[int]] = {}  # {zone_id: set of adjacent zone_ids}

# Mappings & Queues
cam_topic_map: Dict[int, str] = {}
cam_ip_map: Dict[int, str] = {}
camera_state: Dict[int, Dict[str, Any]] = {}
frame_queues: Dict[int, Queue] = {}
cam_display_enabled: Dict[int, int] = {}

display_frames: Dict[int, Any] = {}
cam_stop_events: Dict[int, threading.Event] = {}
detect_timestamps: Dict[int, List[float]] = defaultdict(list)
last_detect_time: Dict[int, float] = {}

# Tracking instances (SmartTracker for each cam, GlobalTracker instance)
cam_trackers: Dict[int, Any] = {}
global_tracker: Any = None  # Will be assigned an instance of GlobalTracker

# Crop config cache (loaded from DB, keyed by cam_id)
cam_crop_configs: Dict[int, Any] = {}  # {cam_id: CropConfig}

# ROI mask cache (lazy-built per camera, keyed by cam_id)
cam_roi_masks: Dict[int, Any] = {}       # {cam_id: np.ndarray uint8 (H,W) | None}
cam_roi_mask_shape: Dict[int, tuple] = {} # {cam_id: (H, W)} — dùng để detect resolution change
