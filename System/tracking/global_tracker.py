import time
import numpy as np
from scipy.optimize import linear_sum_assignment

class GlobalTrack:
    _global_id_counter = 0

    @classmethod
    def reset_counter(cls):
        cls._global_id_counter = 0

    def __init__(self, feature, local_id_map, zone):
        GlobalTrack._global_id_counter += 1
        self.global_id = GlobalTrack._global_id_counter
        self.features = [feature] # Store history for fusion
        self.feature = feature # [Dim] (normalized)
        self.active_cams = local_id_map # dict {cam_id: local_id}
        self.last_seen = time.time()
        self.zone = zone
        
    def update(self, feature, cam_id, local_id, zone):
        self.features.append(feature)
        self.features = self.features[-10:] # Keep the last 10 features
        
        avg_feature = np.mean(self.features, axis=0)
        self.feature = avg_feature / (np.linalg.norm(avg_feature) + 1e-6)
        
        self.active_cams[cam_id] = local_id
        self.last_seen = time.time()
        self.zone = zone

class GlobalTracker:
    def __init__(self, sim_thresh=0.55, max_lost=30.0):
        self.tracks = []
        self.sim_thresh = sim_thresh
        self.max_lost = max_lost
        
        # Ngưỡng (Thresholds) cho Logic Split Track và Force Match
        self.high_thres = 0.75
        self.low_thres = 0.60
        self.conflict_sim_thres = 0.50

    def reset(self):
        self.tracks = []
        GlobalTrack.reset_counter()

    def update(self, detect_features, cam_id, local_ids):
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t.last_seen < self.max_lost]
        
        # Lấy zone của camera hiện tại từ state (load từ DB, không hardcode)
        import shared.state as state
        cam_zone = state.cam_zone_map.get(cam_id, 0)  # 0 = zone không xác định

        if len(detect_features) == 0:
            return []
            
        out_global_ids = [None] * len(detect_features)
        unmatched_dets = list(range(len(detect_features)))
        
        def match_pass(det_indices, track_indices, is_inter_zone=False):
            if not det_indices or not track_indices:
                return []
            
            pass_track_features = np.array([self.tracks[i].feature for i in track_indices])
            pass_det_features = np.array([detect_features[i] for i in det_indices])
            
            sim_matrix = np.dot(pass_det_features, pass_track_features.T) # [N, M]
            
            for r in range(len(det_indices)):
                for c in range(len(track_indices)):
                    t = self.tracks[track_indices[c]]
                    time_diff = now - t.last_seen
                    orig_sim = sim_matrix[r, c]
                    
                    # 1. Phát hiện trùng ID (Conflict) - Thời gian gần nhưng Feature khác xa -> Đánh rớt để Split Track
                    if time_diff < 1.0 and orig_sim < self.conflict_sim_thres:
                        sim_matrix[r, c] = 0.0
                        continue
                        
                    # 2. Logic Scoring cho khác Zone (Inter-zone)
                    if is_inter_zone:
                        # Bypass phạt nếu chắc chắn giống người cũ (High Thres)
                        if orig_sim >= self.high_thres or (time_diff < 1.0 and orig_sim >= 0.70):
                            pass 
                        # Nằm trong vùng nghi ngờ hoặc khoảng cách điểm yếu -> áp dụng trừ điểm theo cooldown
                        else:
                            if time_diff < 2.0:
                                sim_matrix[r, c] *= 0.6
                            elif time_diff < 5.0:
                                sim_matrix[r, c] *= 0.8
                                
            cost_matrix = 1.0 - sim_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matches = []
            for r, c in zip(row_ind, col_ind):
                if sim_matrix[r, c] >= self.sim_thresh:
                    matches.append((det_indices[r], track_indices[c]))
                    
            return matches

        # Pass 1: Intra-zone (Ưu tiên)
        intra_track_indices = [i for i, t in enumerate(self.tracks) if t.zone == cam_zone]
        intra_matches = match_pass(unmatched_dets, intra_track_indices, is_inter_zone=False)
        
        for det_idx, trk_idx in intra_matches:
            t = self.tracks[trk_idx]
            t.update(detect_features[det_idx], cam_id, local_ids[det_idx], cam_zone)
            out_global_ids[det_idx] = t.global_id
            unmatched_dets.remove(det_idx)
            
        # Pass 2: Inter-zone (chỉ xét zone LANg giềng theo zone_relations trong DB)
        adjacent = state.adjacent_zones.get(cam_zone, set())
        if adjacent:
            # Chỉ so sánh với track trong các zone tiếp giáp
            inter_track_indices = [i for i, t in enumerate(self.tracks)
                                   if t.zone in adjacent]
        else:
            # Fallback: nếu không có dữ liệu zone_relations, xét tất cả zone khác
            inter_track_indices = [i for i, t in enumerate(self.tracks)
                                   if t.zone != cam_zone and t.zone != 0]
        inter_matches = match_pass(unmatched_dets, inter_track_indices, is_inter_zone=True)
        
        for det_idx, trk_idx in inter_matches:
            t = self.tracks[trk_idx]
            t.update(detect_features[det_idx], cam_id, local_ids[det_idx], cam_zone)
            out_global_ids[det_idx] = t.global_id
            unmatched_dets.remove(det_idx)
            
        # Pass 3: Create new tracks for unmatched detections (Tự nhiên hoạt động như Split Track)
        for det_idx in unmatched_dets:
            nt = GlobalTrack(detect_features[det_idx], {cam_id: local_ids[det_idx]}, cam_zone)
            self.tracks.append(nt)
            out_global_ids[det_idx] = nt.global_id
            
        return out_global_ids

    def get_global_id(self, cam_id, local_id):
        for t in self.tracks:
            if cam_id in t.active_cams and t.active_cams[cam_id] == local_id:
                return t.global_id
        return None
