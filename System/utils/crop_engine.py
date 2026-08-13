"""
crop_engine.py
--------------
Engine tạo crop từ frame theo config lưới (grid_rows × grid_cols) với overlap.
Hỗ trợ ROI polygon (chuẩn hóa 0.0–1.0) để lọc vùng không cần detect.

Pipeline:
    DB config → CropConfig → build_roi_mask() → generate_crops(frame, roi_mask)
                                                    → (valid_crops, offsets, coverages)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
import numpy as np
import cv2


@dataclass
class CropConfig:
    """Config crop của 1 camera, map trực tiếp từ bảng camera_crop_config."""
    crop_mode: int          # 0 = không crop (toàn frame), 1 = crop lưới
    grid_rows: int          # số hàng
    grid_cols: int          # số cột
    overlap_ratio: float    # tỉ lệ overlap giữa các ô (0.0 – 0.5)
    roi_polygons: list = field(default_factory=list)
    # list of polygons, mỗi polygon = list of [x_norm, y_norm] (0.0–1.0)
    # Ví dụ: [[[0.0,0.5],[1.0,0.5],[1.0,1.0],[0.0,1.0]]]
    # Rỗng [] = toàn frame (không áp ROI)

    @classmethod
    def default(cls) -> "CropConfig":
        """Hành vi tương đương code cũ: 1×2, overlap 10%, toàn frame."""
        return cls(crop_mode=1, grid_rows=1, grid_cols=2, overlap_ratio=0.1, roi_polygons=[])

    @classmethod
    def from_db_row(cls, crop_mode, grid_rows, grid_cols, overlap_ratio, roi_polygon_json: str) -> "CropConfig":
        """Parse từ row DB. roi_polygon_json là TEXT, ví dụ: '[[[0.0,0.5],[1.0,0.5],...]]'"""
        try:
            raw = json.loads(roi_polygon_json or "[]")
            # Chuẩn hoá: raw có thể là [] hoặc [[[x,y],...]] tuỳ format
            polygons = raw if isinstance(raw, list) else []
        except Exception:
            polygons = []
        return cls(
            crop_mode=crop_mode,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            overlap_ratio=overlap_ratio,
            roi_polygons=polygons,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROI Mask Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_roi_mask(h: int, w: int, roi_polygons: list) -> Optional[np.ndarray]:
    """
    Tạo mask uint8 (H×W) từ danh sách polygon chuẩn hóa.
    Pixel trong ROI = 255, pixel ngoài = 0.

    Parameters
    ----------
    roi_polygons : list of polygons, mỗi polygon = [[x_norm, y_norm], ...]
                   Tọa độ chuẩn hóa 0.0–1.0.
                   Rỗng [] → trả về None (không áp mask).

    Returns
    -------
    mask : np.ndarray uint8 (H, W) hoặc None nếu không có ROI.
    """
    if not roi_polygons:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    for polygon_norm in roi_polygons:
        if not polygon_norm:
            continue
        # Denormalize sang pixel coords
        pts = np.array(
            [[int(x * w), int(y * h)] for x, y in polygon_norm],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_roi_mask(frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Áp ROI mask lên frame. Pixel ngoài ROI sẽ bị đặt về 0 (đen).
    Nếu mask=None → trả về frame gốc không xử lý (zero overhead).
    """
    if mask is None:
        return frame
    return cv2.bitwise_and(frame, frame, mask=mask)


def _cell_roi_coverage(mask: Optional[np.ndarray], cy0: int, cy1: int, cx0: int, cx1: int) -> float:
    """Tính tỉ lệ pixel ROI (mask > 0) trong vùng cell [cy0:cy1, cx0:cx1]. Trả về 0.0–1.0."""
    if mask is None:
        return 1.0  # Không có mask = toàn bộ là ROI
    cell_mask = mask[cy0:cy1, cx0:cx1]
    total = cell_mask.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(cell_mask)) / total


# ─────────────────────────────────────────────────────────────────────────────
# Core Crop Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_crops(
    frame: np.ndarray,
    config: CropConfig,
    roi_mask: Optional[np.ndarray] = None,
    min_coverage: float = 0.05,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[float]]:
    """
    Tạo danh sách crop từ frame theo config lưới, có lọc theo ROI.

    Parameters
    ----------
    frame        : Frame đã được apply_roi_mask() (nếu có ROI).
    config       : CropConfig.
    roi_mask     : Mask uint8 (H×W) hoặc None.
    min_coverage : Ngưỡng tỉ lệ ROI tối thiểu. Cell nào < ngưỡng sẽ bị bỏ qua.

    Returns
    -------
    valid_crops   : list[ndarray]       – Chỉ crop có coverage >= min_coverage.
    valid_offsets : list[(off_x, off_y)]– Offset tương ứng.
    valid_covs    : list[float]         – Coverage ratio (dùng để debug / display).
    """
    if config.crop_mode == 0 or (config.grid_rows == 1 and config.grid_cols == 1):
        return [frame], [(0, 0)], [1.0]

    h, w = frame.shape[:2]
    rows, cols = config.grid_rows, config.grid_cols
    overlap = config.overlap_ratio

    cell_h = h / rows
    cell_w = w / cols
    ov_x = int(cell_w * overlap)
    ov_y = int(cell_h * overlap)

    valid_crops: List[np.ndarray] = []
    valid_offsets: List[Tuple[int, int]] = []
    valid_covs: List[float] = []

    for r in range(rows):
        for c in range(cols):
            # Tọa độ ô chuẩn (không overlap)
            x0 = int(c * cell_w)
            y0 = int(r * cell_h)
            x1 = int((c + 1) * cell_w)
            y1 = int((r + 1) * cell_h)

            # Kiểm tra coverage trên vùng ô cơ bản (không overlap)
            cov = _cell_roi_coverage(roi_mask, y0, y1, x0, x1)
            if cov < min_coverage:
                continue  # Cell này không đủ ROI → bỏ qua YOLO

            # Mở rộng với overlap
            cx0 = max(0, x0 - ov_x)
            cy0 = max(0, y0 - ov_y)
            cx1 = min(w, x1 + ov_x)
            cy1 = min(h, y1 + ov_y)

            valid_crops.append(frame[cy0:cy1, cx0:cx1])
            valid_offsets.append((cx0, cy0))
            valid_covs.append(cov)

    # Nếu tất cả cell đều bị lọc (ROI quá nhỏ / sai config) → trả về toàn frame để tránh crash
    if not valid_crops:
        return [frame], [(0, 0)], [1.0]

    return valid_crops, valid_offsets, valid_covs


# ─────────────────────────────────────────────────────────────────────────────
# ROI Bounding-Box Crop (thay thế grid khi ROI được định nghĩa)
# ─────────────────────────────────────────────────────────────────────────────

def get_roi_bbox_crop(
    frame: np.ndarray,
    roi_mask: np.ndarray,
    roi_polygons: list,
) -> Tuple[np.ndarray, int, int]:
    """
    Crop ketat theo bounding rect của tất cả ROI polygon, rồi apply mask bên trong.

    Pipeline mới:
        Frame → Crop ROI (bounding box) → apply mask → YOLO (engine tự resize 640)

    Parameters
    ----------
    frame        : Frame gốc (đã enhanced).
    roi_mask     : Mask uint8 (H×W) từ build_roi_mask().
    roi_polygons : List of polygons (normalized coords).

    Returns
    -------
    crop     : np.ndarray — vùng ảnh khớp với bounding rect của ROI, đã mask.
    offset_x : int — tọa độ x gốc trong frame gốc (dùng để map box về frame).
    offset_y : int — tọa độ y gốc trong frame gốc.
    """
    h, w = frame.shape[:2]

    # Tính bounding rect của union tất cả polygons
    all_pts: List[Tuple[int, int]] = []
    for polygon_norm in roi_polygons:
        for x_n, y_n in polygon_norm:
            all_pts.append((int(x_n * w), int(y_n * h)))

    if not all_pts:
        # Không có polygon → trả về toàn frame
        return apply_roi_mask(frame, roi_mask), 0, 0

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(w, max(xs))
    y1 = min(h, max(ys))

    if x1 <= x0 or y1 <= y0:
        return apply_roi_mask(frame, roi_mask), 0, 0

    # Apply mask lên toàn frame rồi crop vùng bounding rect
    masked_frame = apply_roi_mask(frame, roi_mask)
    crop = masked_frame[y0:y1, x0:x1]

    return crop, x0, y0
