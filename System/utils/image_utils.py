import cv2
import numpy as np

# Reused across calls instead of constructing a new CLAHE object (and its
# internal LUT buffers) on every low-light frame — only the single YOLO
# worker thread calls enhance_for_ai(), so this is thread-safe here.
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def enhance_for_ai(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Chỉ tăng nhẹ mức giới hạn phơi sáng (clipLimit) từ 2.0 lên 3.0
    # để ảnh sáng hơn một chút mà không bị quá chói/gắt
    cl = _clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_frame
