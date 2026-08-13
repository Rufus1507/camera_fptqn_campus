import time
import numpy as np
import shared.state as state
from tracking.smart_tracker import SmartTracker

class TrackingOrchestrator:
    @staticmethod
    def apply_nms(temp_boxes):
        final_temp_boxes = []
        if len(temp_boxes) > 0:
            temp_boxes = sorted(temp_boxes, key=lambda x: x[4], reverse=True)
            for b_new in temp_boxes:
                is_duplicate = False
                for k_box in final_temp_boxes:
                    x1_inter = max(b_new[0], k_box[0])
                    y1_inter = max(b_new[1], k_box[1])
                    x2_inter = min(b_new[2], k_box[2])
                    y2_inter = min(b_new[3], k_box[3])
                    
                    if x2_inter > x1_inter and y2_inter > y1_inter: # Giao nhau
                        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                        b_area = (b_new[2] - b_new[0]) * (b_new[3] - b_new[1])
                        k_area = (k_box[2] - k_box[0]) * (k_box[3] - k_box[1])
                        
                        union_area = b_area + k_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        ioa = inter_area / min(b_area, k_area) if min(b_area, k_area) > 0 else 0
                        
                        if iou > 0.40 or ioa > 0.75:
                            is_duplicate = True
                            break
                            
                if not is_duplicate:
                    final_temp_boxes.append(b_new)
        return final_temp_boxes

    @staticmethod
    def process_camera_boxes(cid, results_for_frame, offsets_for_frame):
        """Thu thập boxes từ K crops rồi ứng dụng NMS.

        Parameters
        ----------
        results_for_frame : list  – K YOLO result objects (khớp với offsets)
        offsets_for_frame : list  – K tuple (off_x, off_y) map crop → frame gốc
        """
        temp_boxes = []
        for res, (offset_x, offset_y) in zip(results_for_frame, offsets_for_frame):
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    cx1, cy1, cx2, cy2 = coords
                    cx1 += offset_x
                    cx2 += offset_x
                    cy1 += offset_y
                    cy2 += offset_y
                    # Lọc rác (quá nhỏ)
                    if (cx2 - cx1) * (cy2 - cy1) < 2000:
                        continue
                    temp_boxes.append([int(cx1), int(cy1), int(cx2), int(cy2), float(box.conf)])
        return TrackingOrchestrator.apply_nms(temp_boxes)
        
    @staticmethod
    def assign_global_ids(batch_cam_ids, all_valid_boxes, batch_features, all_tracked_boxes_list):
        from shared.state import global_tracker, global_tracker_lock
        feature_idx = 0
        all_final_boxes = [] # per camera
        for i in range(len(batch_cam_ids)):
            cid = batch_cam_ids[i]
            tracked_boxes = all_tracked_boxes_list[i]
            cam_features = []
            cam_local_ids = []
            cam_valid_boxes = []
            
            while feature_idx < len(all_valid_boxes) and all_valid_boxes[feature_idx][0] == i:
                box = all_valid_boxes[feature_idx][1]
                cam_valid_boxes.append(box)
                cam_local_ids.append(box[4]) # id cục bộ
                cam_features.append(batch_features[feature_idx])
                feature_idx += 1
                
            final_global_boxes = []
            
            if cam_features:
                with global_tracker_lock:
                    g_ids = global_tracker.update(np.array(cam_features), cid, cam_local_ids)
                
                for box, gid in zip(cam_valid_boxes, g_ids):
                    b_new = list(box)
                    b_new[4] = gid
                    final_global_boxes.append(b_new)
            
            # Xử lý các object không chạy ReID ở frame hiện tại (hoặc ReID bị bỏ qua/thất bại)
            reid_local_ids = set(cam_local_ids)
            for box in tracked_boxes:
                local_id = box[4]
                if local_id not in reid_local_ids:
                    with global_tracker_lock:
                        gid = global_tracker.get_global_id(cid, local_id)
                    
                    if gid is None:
                        # Fallback về local ID (hiện đã là dải 1000+) 
                        # Sẽ được thay bằng Global ID ở lần ReID kế tiếp (mỗi 5 frame hoặc khi rảnh)
                        gid = local_id 
                    
                    b_new = list(box)
                    b_new[4] = gid
                    final_global_boxes.append(b_new)

            all_final_boxes.append(final_global_boxes)
            
        return all_final_boxes
