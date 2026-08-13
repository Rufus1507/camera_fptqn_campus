import cv2
import numpy as np

def get_brightness(frame: np.ndarray) -> str:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hsv[:, :, 1].mean() < 20:
        return "1"   # IR / tối

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    b = p50 * 0.5 + (p10 + p90) * 0.25

    if b < 25.5: return "1"
    elif b < 51: return "2"
    elif b < 76.5: return "3"
    elif b < 102: return "4"
    elif b < 127.5: return "5"
    elif b < 153: return "6"
    elif b < 178.5: return "7"
    elif b < 204: return "8"
    elif b < 229.5: return "9"
    else: return "10"

def _is_gray_frame(frame: np.ndarray, threshold: float = 2.0) -> bool:
    return frame.std() < threshold
