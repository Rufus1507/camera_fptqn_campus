import os
import logging

# MQTT Config
MQTT_BROKER       = "localhost"         # Broker amqtt chạy ngầm trên cùng máy (amqtt.service)
MQTT_PORT         = 1883
MQTT_TOPIC        = "camera/stats"
CLIENT_ID         = "machine_a_camera_ai"
API_SEND_INTERVAL = 5
MQTT_URI          = f"mqtt://{MQTT_BROKER}:{MQTT_PORT}/"

_SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logging Config
LOG_DIR = os.path.join(_SYSTEM_DIR, "logs")

# DB Config
DB_PATH = os.path.join(_SYSTEM_DIR, "database", "cameras.db")
DB_POLL_INTERVAL = 10       # seconds

# API Sync Config
API_URL           = "http://0.0.0.0:8001/api/cameras"
API_SYNC_INTERVAL = 30      # seconds — đồng bộ DB với API mỗi 30 giây

# Model Config
DEVICE = 0

# Đặt USE_CUSTOM_MODEL = True để dùng yolov8s_custom, False để dùng yolov8s gốc
USE_CUSTOM_MODEL = True

# TEMP OVERRIDE (2026-08-12): both yolov8s_custom.engine and .onnx were exported
# with a STATIC batch of 1 (see models/export_custom.log), so they can't serve
# the multi-camera batched predict() call in core/yolo_engine.py — only .pt
# (native PyTorch, no fixed input shape) can. Forcing .pt until the engine is
# rebuilt with a larger/dynamic batch.
# TODO: set back to False once yolov8s_custom.engine is rebuilt for batch >= BATCH_SIZE.
FORCE_PT_MODEL = False

# Model-selection warnings collected at import time (a real logger doesn't
# exist yet -- init_logging() needs LOG_DIR from this module, so this module
# can't depend on the logger). Flushed by main.py right after init_logging().
STARTUP_WARNINGS: list = []  # list[tuple[int, str]]

def _warn(msg: str) -> None:
    STARTUP_WARNINGS.append((logging.WARNING, msg))

if USE_CUSTOM_MODEL:
    # Custom model: ưu tiên .engine (TensorRT) → .onnx → fallback .pt
    _custom_engine = os.path.join(_SYSTEM_DIR, "models", "yolov8s_custom.engine")
    _custom_onnx   = os.path.join(_SYSTEM_DIR, "models", "yolov8s_custom.onnx")
    _custom_pt     = os.path.join(_SYSTEM_DIR, "models", "yolov8s_custom.pt")
    if FORCE_PT_MODEL and os.path.exists(_custom_pt):
        MODEL_PATH = _custom_pt
        _warn("[settings] FORCE_PT_MODEL=True — using yolov8s_custom.pt instead of "
              "the TensorRT engine/ONNX (both static batch=1, can't serve the "
              "multi-camera batched inference call). Uses more memory than the FP16 "
              "engine on Jetson. Set FORCE_PT_MODEL=False once yolov8s_custom.engine "
              "is rebuilt for a larger/dynamic batch.")
    elif os.path.exists(_custom_engine):
        MODEL_PATH = _custom_engine
    elif os.path.exists(_custom_onnx):
        MODEL_PATH = _custom_onnx
        _warn("[settings] TensorRT engine not found — falling back to ONNX "
              "(slower, uses more memory than the FP16 TensorRT engine on Jetson)")
    else:
        MODEL_PATH = _custom_pt
        _warn("[settings] TensorRT engine/ONNX not found — falling back to FP32 .pt. "
              "This roughly DOUBLES model memory use on Jetson vs the FP16 TensorRT "
              "engine. Rebuild yolov8s_custom.engine as soon as possible.")
else:
    # Standard model: ưu tiên .engine (TensorRT) → fallback .pt
    _engine_path = os.path.join(_SYSTEM_DIR, "models", "yolov8s.engine")
    _pt_path     = os.path.join(_SYSTEM_DIR, "models", "yolov8s.pt")
    if FORCE_PT_MODEL and os.path.exists(_pt_path):
        MODEL_PATH = _pt_path
        _warn("[settings] FORCE_PT_MODEL=True — using yolov8s.pt instead of the "
              "TensorRT engine (static batch=1, can't serve the multi-camera batched "
              "inference call). Set FORCE_PT_MODEL=False once yolov8s.engine is "
              "rebuilt for a larger/dynamic batch.")
    elif os.path.exists(_engine_path):
        MODEL_PATH = _engine_path
    else:
        MODEL_PATH = _pt_path
        _warn("[settings] TensorRT engine not found — falling back to FP32 .pt. "
              "This roughly DOUBLES model memory use on Jetson vs the FP16 TensorRT "
              "engine. Rebuild yolov8s.engine as soon as possible.")

CONF_THRESH  = 0.33
PERSON_CLASS = 0
BATCH_SIZE   = 4

# Max batch dimension the loaded model file actually supports. .engine/.onnx
# here are exported with a STATIC batch of 1 (see models/export_custom.log) —
# sending more crashes YOLOEngine.predict(). .pt is native PyTorch and has no
# fixed input shape, so it can take a real batch — capped at BATCH_SIZE since
# that's the most crops/cycle the current 1×1 crop-grid config produces.
# Derived from MODEL_PATH so this tracks FORCE_PT_MODEL automatically —
# no manual edit needed when that flag is flipped back.
MODEL_MAX_BATCH = BATCH_SIZE if MODEL_PATH.endswith(".pt") else 1

# FPS & Queue Config
DETECT_FPS    = 1.2
DISPLAY_FPS   = 3
QUEUE_PER_CAM = 1
FPS_WINDOW    = 5.0
RTSP_RETRY_DELAY = 10
STALE_TIMEOUT = 5.0

# Display Config
DISPLAY_CELL_W = 640
DISPLAY_CELL_H = 360
DISPLAY_COLS   = 2

# Max working frame size for queued/display frames (aspect-ratio preserved,
# only downscales if the source exceeds this — never upscales). Both the
# display grid (resizes to DISPLAY_CELL_W x DISPLAY_CELL_H per camera) and
# YOLOEngine (letterboxes to imgsz=640) resize frames down again regardless,
# so this only needs to be "big enough", not a fixed target size.
MAX_FRAME_W = 1280
MAX_FRAME_H = 720


# Env vars setup that was at module level
os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
