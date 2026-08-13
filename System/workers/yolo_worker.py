import os
import time
import logging
import threading
from queue import Empty
import torch
import numpy as np

import shared.state as state
from config.settings import BATCH_SIZE, FPS_WINDOW, CONF_THRESH, PERSON_CLASS
from utils.brightness import get_brightness
from utils.image_utils import enhance_for_ai
from utils.crop_engine import CropConfig, generate_crops, build_roi_mask, apply_roi_mask, get_roi_bbox_crop
from core.tracker import TrackingOrchestrator
from tracking.smart_tracker import SmartTracker
from utils.logger import get_system_logger, log_throttled

def _yolo_thread_logic(yolo_engine, reid_engine):
    """
    1 thread duy nhất: smart round-robin + batch collection.
    """
    detect_cycle_count = 0
    round_robin_idx = 0
    while True:
        with state.state_lock:
            cur_ids = list(state.CAM_IDS)

        n = len(cur_ids)
        if n == 0:
            time.sleep(0.1)
            continue

        cams_with_frames = []
        for cid in cur_ids:
            q = state.frame_queues.get(cid)
            if q is not None:
                qs = q.qsize()
                if qs > 0:
                    cams_with_frames.append((cid, qs))

        if not cams_with_frames:
            time.sleep(0.0005)
            continue

        batch_frames  = []
        batch_cam_ids = []
        
        available_cids = [x[0] for x in cams_with_frames]
        start_idx = round_robin_idx % n
        ordered_search = cur_ids[start_idx:] + cur_ids[:start_idx]
        
        for cid in ordered_search:
            if len(batch_frames) >= BATCH_SIZE:
                break
            if cid in available_cids:
                q = state.frame_queues.get(cid)
                if q is None: continue
                try:
                    frame = q.get_nowait()
                    batch_frames.append(frame)
                    batch_cam_ids.append(cid)
                except Empty:
                    continue

        if batch_cam_ids:
            last_picked_cid = batch_cam_ids[-1]
            try:
                last_idx = cur_ids.index(last_picked_cid)
                round_robin_idx = last_idx + 1
            except ValueError:
                round_robin_idx = 0

        if not batch_frames:
            time.sleep(0.0005)
            continue

        # Purge CUDA cache every 30 real detection cycles (NOT raw loop
        # iterations — idle spins used to inflate this counter and trigger
        # empty_cache() dozens of times/sec, causing allocator fragmentation
        # instead of preventing it).
        detect_cycle_count += 1
        if detect_cycle_count % 30 == 0:
            torch.cuda.empty_cache()

        # ──────────────────────────────────────────────────────────────────────
        # 🛡️  LớP 1: Bắt CUDA exception → yêu cầu shutdown graceful
        # ──────────────────────────────────────────────────────────────────────
        try:
            with torch.no_grad():
                batch_enhanced = []
                for f in batch_frames:
                    if int(get_brightness(f)) <= 4:
                        batch_enhanced.append(enhance_for_ai(f))
                    else:
                        batch_enhanced.append(f)

                # ── Bước 1: Generate crops theo config từng camera ──────────────────
                # flat_crops  : tất cả crop gộp chung, gửi vào YOLO 1 batch
                # per_frame_k : số crop mỗi frame (dùng để unpack kết quả sau)
                # per_frame_offsets: offset tương ứng (off_x, off_y) mỗi crop
                flat_crops: list = []
                per_frame_k: list = []            # [k0, k1, ...]
                per_frame_offsets: list = []      # [(off0,off1,...), ...]

                for i, f in enumerate(batch_enhanced):
                    cid_i = batch_cam_ids[i]
                    cfg: CropConfig = state.cam_crop_configs.get(cid_i) or CropConfig.default()

                    # ── Lazy build / invalidate ROI mask khi resolution thay đổi ──
                    h, w = f.shape[:2]
                    if state.cam_roi_mask_shape.get(cid_i) != (h, w):
                        mask = build_roi_mask(h, w, cfg.roi_polygons)
                        state.cam_roi_masks[cid_i]      = mask
                        state.cam_roi_mask_shape[cid_i] = (h, w)

                    roi_mask = state.cam_roi_masks.get(cid_i)

                    if roi_mask is not None and cfg.roi_polygons:
                        # ══════════════════════════════════════════════════════
                        # Pipeline chuẩn: ROI bbox crop → Grid → YOLO
                        # ══════════════════════════════════════════════════════

                        # Bước 1: Crop ketat bounding rect của ROI polygon
                        roi_crop, roi_ox, roi_oy = get_roi_bbox_crop(f, roi_mask, cfg.roi_polygons)

                        # Bước 2: Cắt phần roi_mask tương ứng với roi_crop
                        # (để generate_crops tính coverage đúng trên roi_crop)
                        h_rc, w_rc = roi_crop.shape[:2]
                        roi_crop_mask = roi_mask[roi_oy:roi_oy + h_rc, roi_ox:roi_ox + w_rc]

                        # Bước 3: Grid crop bên trong roi_crop
                        cells, cell_offsets, _ = generate_crops(roi_crop, cfg, roi_crop_mask)

                        # Bước 4: Cộng dồn offset
                        # frame_offset = (roi_ox + cell_ox, roi_oy + cell_oy)
                        combined = [(roi_ox + cox, roi_oy + coy) for cox, coy in cell_offsets]

                        flat_crops.extend(cells)
                        per_frame_k.append(len(cells))
                        per_frame_offsets.append(combined)

                    else:
                        # ══ Fallback: Grid crop trên toàn frame (khi không có ROI) ══
                        crops_i, offsets_i, _ = generate_crops(f, cfg, roi_mask)
                        flat_crops.extend(crops_i)
                        per_frame_k.append(len(crops_i))
                        per_frame_offsets.append(offsets_i)

                # ── Bước 2: YOLO inference trên flat batch (1 lần gọi) ─────────────
                results = yolo_engine.predict(flat_crops, conf_thresh=CONF_THRESH, person_class=PERSON_CLASS)

                # ── Bước 3: Unpack kết quả và xử lý từng camera ──────────────────
                MAX_REID = 128
                all_reid_tensors = []
                all_valid_boxes = []   # tuple (idx_in_batch, box_data)
                all_tracked_boxes_list = []

                result_cursor = 0  # con trỏ duyệt flat results
                for i in range(len(batch_frames)):
                    cid = batch_cam_ids[i]
                    frame_enhanced = batch_enhanced[i]
                    k = per_frame_k[i]
                    offsets_i = per_frame_offsets[i]

                    results_for_frame = results[result_cursor: result_cursor + k]
                    result_cursor += k

                    temp_boxes = TrackingOrchestrator.process_camera_boxes(
                        cid, results_for_frame, offsets_i
                    )

                    tracker = state.cam_trackers.get(cid)
                    if tracker is None:
                        tracker = SmartTracker()
                        state.cam_trackers[cid] = tracker

                    tracked_boxes = tracker.update(temp_boxes)
                    all_tracked_boxes_list.append(tracked_boxes)

                    h_img, w_img = frame_enhanced.shape[:2]

                    # Tách danh sách candidate ReID và sắp xếp ưu tiên:
                    # 1. Object mới (age == 1)
                    # 2. Object đến kỳ ReID định kỳ (age % 5 == 0)
                    reid_candidates = []
                    for box in tracked_boxes:
                        bx1, by1, bx2, by2, tid, conf, age = box
                        if age == 1 or age % 5 == 0:
                            reid_candidates.append(box)

                    # Sắp xếp: Ưu tiên age=1 lên trước
                    reid_candidates.sort(key=lambda x: 0 if x[6] == 1 else 1)

                    for box in reid_candidates:
                        bx1, by1, bx2, by2, tid, conf, age = box
                        x1, y1 = max(0, int(bx1)), max(0, int(by1))
                        x2, y2 = min(w_img, int(bx2)), min(h_img, int(by2))

                        if x2 - x1 >= 10 and y2 - y1 >= 10:
                            if len(all_reid_tensors) < MAX_REID:
                                crop = frame_enhanced[y1:y2, x1:x2]
                                t = reid_engine.preprocess(crop)
                                all_reid_tensors.append(t)
                                all_valid_boxes.append((i, box))

                batch_features = reid_engine.extract_features(all_reid_tensors)

                all_final_boxes = TrackingOrchestrator.assign_global_ids(
                    batch_cam_ids, all_valid_boxes, batch_features, all_tracked_boxes_list
                )

        except RuntimeError as e:
            err_str = str(e)
            if "CUDA" in err_str or "NVML" in err_str or "cudnn" in err_str.lower():
                get_system_logger().error(f"CUDA error in YOLO thread, requesting graceful shutdown: {err_str[:200]}")
                state.shutdown_exit_code = 1
                state.shutdown_event.set()
                return
            else:
                raise
        except Exception as e:
            log_throttled(
                get_system_logger(), logging.ERROR,
                "yolo_thread_unexpected_error",
                f"Unexpected error in YOLO thread: {e}",
                interval=60.0,
            )
            continue

        now = time.time()
        for i in range(len(batch_frames)):
            cid = batch_cam_ids[i]
            frame = batch_frames[i]
            final_global_boxes = all_final_boxes[i]

            # ── Post-tracking ROI filter ────────────────────────────────
            # Lọc bỏ box nào overlap với ROI mask < 30%.
            # Giải quyết 3 trường hợp lọt sót:
            #   1. Overlap crop tràn ra ngoài ROI
            #   2. Tracker giữ track sau khi người ra khỏi ROI
            #   3. Person đứng sát ranh giới (box nửa trong nửa ngoài)
            roi_mask = state.cam_roi_masks.get(cid)
            if roi_mask is not None:
                filtered_boxes = []
                for box in final_global_boxes:
                    bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    h_m, w_m = roi_mask.shape[:2]
                    bx1c = max(0, bx1); by1c = max(0, by1)
                    bx2c = min(w_m, bx2); by2c = min(h_m, by2)
                    if bx2c <= bx1c or by2c <= by1c:
                        continue  # Box hoàn toàn ngoài frame
                    box_area = (bx2c - bx1c) * (by2c - by1c)
                    if box_area == 0:
                        continue
                    roi_overlap = int(np.count_nonzero(roi_mask[by1c:by2c, bx1c:bx2c]))
                    overlap_ratio = roi_overlap / box_area
                    if overlap_ratio >= 0.30:
                        filtered_boxes.append(box)
                final_global_boxes = filtered_boxes

            with state.state_lock:
                ts = state.detect_timestamps.setdefault(cid, [])
                ts.append(now)
                cutoff = now - FPS_WINDOW
                state.detect_timestamps[cid] = [t for t in ts if t > cutoff]
                fps_val = round(len(state.detect_timestamps[cid]) / FPS_WINDOW, 2)

                person_ids = [f"G_{box[4]}" for box in final_global_boxes]
                state.last_detect_time[cid] = now
                brightness = get_brightness(frame)

                if cid in state.camera_state:
                    state.camera_state[cid] = {
                        "timestamp":  int(now),
                        "fps":        fps_val,
                        "person_ids": person_ids,
                        "is_night":   brightness,
                        "boxes":      final_global_boxes,
                    }

def yolo_worker(yolo_engine, reid_engine):
    t = threading.Thread(target=_yolo_thread_logic, args=(yolo_engine, reid_engine), name="yolo_t0", daemon=True)
    t.start()
