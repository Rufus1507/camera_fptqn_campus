import numpy as np
from scipy.optimize import linear_sum_assignment

class _SmartTrack:
    _id_counter = 100000

    @classmethod
    def reset_counter(cls):
        cls._id_counter = 100000

    def __init__(self, box, conf):
        _SmartTrack._id_counter += 1
        self.track_id  = _SmartTrack._id_counter
        self.box       = np.array(box, dtype=float)   # [x1, y1, x2, y2]
        self.conf      = conf
        self.center    = self._get_center(self.box)
        self.age       = 1        # History: frame tồn tại
        self.miss      = 0        # History: frame không match

    def _get_center(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def update(self, box, conf):
        self.box    = np.array(box, dtype=float)
        self.conf   = conf
        self.center = self._get_center(self.box)
        self.age   += 1
        self.miss   = 0

class SmartTracker:
    def __init__(self, iou_weight=0.3, dist_weight=0.7, dist_thresh=400, max_miss=3):
        self._tracks = []
        self.iou_weight = iou_weight
        self.dist_weight = dist_weight
        self.dist_thresh = dist_thresh
        self.max_miss = max_miss

    def reset(self):
        self._tracks = []
        _SmartTrack.reset_counter()

    def _iou(self, box_a, box_b):
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
        return iou

    def update(self, detections):
        if not detections:
            for t in self._tracks:
                t.miss += 1
                t.age += 1
            self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]
            return self._collect()

        dets = np.array(detections, dtype=float)
        if len(self._tracks) == 0:
            for det in dets:
                self._tracks.append(_SmartTrack(det[:4], det[4]))
            return self._collect()

        cost_matrix = np.zeros((len(dets), len(self._tracks)))
        for d_idx, det in enumerate(dets):
            det_box = det[:4]
            det_center = np.array([(det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2])
            for t_idx, trk in enumerate(self._tracks):
                iou_val = self._iou(det_box, trk.box)
                dist_val = np.linalg.norm(det_center - trk.center)
                dist_norm = min(dist_val / self.dist_thresh, 1.0)
                
                cost = self.iou_weight * (1.0 - iou_val) + self.dist_weight * dist_norm
                cost_matrix[d_idx, t_idx] = cost

        ri, ci = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_dets = set()
        
        for r, c in zip(ri, ci):
            if cost_matrix[r, c] < 0.8:
                self._tracks[c].update(dets[r][:4], dets[r][4])
                matched_tracks.add(c)
                matched_dets.add(r)
                
        for d_idx, det in enumerate(dets):
            if d_idx not in matched_dets:
                self._tracks.append(_SmartTrack(det[:4], det[4]))
                
        for t_idx, trk in enumerate(self._tracks):
            if t_idx not in matched_tracks:
                trk.miss += 1
                
        self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]
        
        return self._collect()

    def _collect(self):
        out = []
        for t in self._tracks:
            if t.age >= 1 and t.miss <= 1:
                x1, y1, x2, y2 = t.box.tolist()
                out.append([int(x1), int(y1), int(x2), int(y2), t.track_id, round(t.conf, 3), t.age])
        return out
