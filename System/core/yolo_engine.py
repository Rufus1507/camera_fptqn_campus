import torch
from ultralytics import YOLO
from utils.logger import get_system_logger

class YOLOEngine:
    def __init__(self, model_path: str, device: int, max_batch: int = 1):
        get_system_logger().info(f"[YOLO] Loading {model_path}")
        self.device = device
        self.max_batch = max(1, max_batch)
        self.model = YOLO(model_path, task="detect")
        try:
            self.model.predict(
                torch.zeros(1, 3, 640, 640, device=device),
                verbose=False
            )
            get_system_logger().info("[YOLO] Warmup complete")
        except Exception as e:
            get_system_logger().warning(f"[YOLO] Warmup error (ignored): {e}")

    def predict(self, crops: list, conf_thresh: float, person_class: int):
        if not crops:
            return []
        # Batched call per chunk instead of one .predict() per crop: fewer
        # Python-level preprocessing/allocator round-trips, same per-image
        # NMS/filtering semantics, same list[Results] return (one per crop,
        # same order) — callers are unaffected.
        #
        # Chunked to self.max_batch because the loaded engine may be a
        # TensorRT .engine built with a STATIC batch dimension (e.g.
        # yolov8s_custom.engine == batch 1) — sending more than that in one
        # call raises "input size ... not equal to max model size". With
        # max_batch=1 this is equivalent to one .predict() call per crop.
        results = []
        for start in range(0, len(crops), self.max_batch):
            chunk = crops[start:start + self.max_batch]
            results.extend(self.model.predict(
                source=chunk,
                device=self.device,
                imgsz=640,
                classes=[person_class],
                conf=conf_thresh,
                verbose=False,
                stream=False,
                batch=len(chunk),
            ))
        return results
