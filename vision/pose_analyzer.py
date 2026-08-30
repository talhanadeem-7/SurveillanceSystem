import logging
import threading

import cv2
import numpy as np
import torch
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MiDaS process-wide singleton
# ---------------------------------------------------------------------------
# The depth model and its transform are stateless and identical for every run,
# but PoseAnalyzer is constructed once per run_surveillance() call because it
# holds PER-RUN state (the ORB reference frame, the Kalman filter dicts). That
# meant a full torch.hub.load on every single run.
#
# Loaded lazily on first use and cached here, keyed by (model type, device), so
# switching config.DEPTH_MODEL_TYPE still works. PoseAnalyzer stays constructible
# per run; it just borrows the shared model instead of reloading it.
_MIDAS_CACHE: dict = {}
_MIDAS_LOCK = threading.Lock()      # Streamlit reruns can overlap in threads


def get_midas(model_type: str | None = None):
    """
    Return (device, model, transform) for the requested MiDaS variant.

    First call loads and caches; later calls are a dict lookup. Safe to call
    from multiple threads — the lock prevents two concurrent torch.hub.load
    calls for the same key.
    """
    if model_type is None:
        model_type = config.DEPTH_MODEL_TYPE
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    key = (model_type, str(device))

    cached = _MIDAS_CACHE.get(key)
    if cached is not None:
        return cached

    with _MIDAS_LOCK:
        cached = _MIDAS_CACHE.get(key)      # re-check inside the lock
        if cached is not None:
            return cached

        logger.info("Loading MiDaS depth model %s on %s (first use)", model_type, device)
        print("Loading MiDaS Depth Model...")
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        model.to(device)
        model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform

        print("MiDaS Model Loaded Successfully.")
        _MIDAS_CACHE[key] = (device, model, transform)
        return _MIDAS_CACHE[key]


class KalmanSmoother:  # stop the "3D boxes" from flickering or jumping around.
    def __init__(self, process_noise=0.005, measurement_noise=0.3):
        self.initialized = False
        self.x = 0.0 
        self.P = 1.0 
        self.Q = process_noise      
        self.R = measurement_noise  

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return measurement
        
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        
        return self.x
# ------------------------

class PoseAnalyzer:
    def __init__(self, reference_frame):  # Prepares the 3D brain and the camera-tracking system.
        self.ref_frame = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        
        # ORB Fingerprinting: It takes a "fingerprint" of the room's background. It remembers where corners and edges are./HEATMAP
        self.orb = cv2.ORB_create(
            nfeatures=int(getattr(config, "ZONE_ALIGN_NFEATURES", 1000))
        )
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_frame, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # --- zone-alignment gating state (see align_zones) ---
        self._align_calls = 0            # processed frames seen by align_zones
        self._last_homography = None     # last successfully computed M
        self.align_consecutive_failures = 0
        self.align_failures_total = 0
        self.align_recomputes = 0

        # MiDaS: borrowed from the process-wide singleton above, NOT reloaded.
        # First PoseAnalyzer in the process pays the torch.hub.load cost; every
        # later run reuses the same model and transform.
        self.device, self.depth_model, self.depth_transform = get_midas()

        # 3. Kalman State Memory
        # NOTE: both dicts grow without bound. person_depth_filters is keyed
        # f"{person_id}_{zone_type}_{kp_idx}", so every distinct track id that
        # reaches a zone adds entries that are never pruned. Per-run only (a new
        # PoseAnalyzer is built each run), so it is bounded by one run's track
        # count — but a long live-camera run in Phase 2 would grow indefinitely.
        self.person_depth_filters = {}
        self.zone_depth_filters = {}

    def get_depth_map(self, frame):  # Converts a flat photo into a 3D map.
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.depth_transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        return normalized_depth

    # It checks if someone is reaching into a "3D Bubble" around an object.
    def check_trespassing(self, keypoints, bbox, zones, depth_map=None, person_id=None):  
        """
        Checks if specific keypoints enter the 3D Voxel of the zone.
        - Restricted: Checks Hands (9, 10)
        - Passive: Checks Feet (15, 16)
        """
        if keypoints is None: return False
        
        pts = keypoints.xy[0].cpu().numpy()
        conf = keypoints.conf[0].cpu().numpy()

        for i, z in enumerate(zones):
            # Choose keypoints based on zone type
            check_indices = [9, 10] if z['type'] == 'restricted' else [15, 16]
            zone_id = f"{z['type']}_{z['name']}_{i}"
            
            # 1. CALCULATE ZONE DEPTH
            if depth_map is not None:
                h, w = depth_map.shape
                
                if z['type'] == 'restricted':
                    zx1, zy1, zx2, zy2 = z['coords']
                    safe_zx1, safe_zy1 = max(0, zx1), max(0, zy1)
                    safe_zx2, safe_zy2 = min(w, zx2), min(h, zy2)
                    zone_roi = depth_map[safe_zy1:safe_zy2, safe_zx1:safe_zx2]
                else:
                    # PASSIVE: Use a mask to get depth from the Polygon area
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [z['polygon']], 255)
                    zone_roi = depth_map[mask == 255]

                if zone_roi.size == 0: continue
                
                # Dynamic Depth Spread
                obj_min_raw = np.percentile(zone_roi, 15)
                obj_max_raw = np.percentile(zone_roi, 85)
                structural_spread = (obj_max_raw - obj_min_raw) / 2.0
                raw_obj_z = np.median(zone_roi)
                
                if zone_id not in self.zone_depth_filters:
                    self.zone_depth_filters[zone_id] = KalmanSmoother(process_noise=0.001, measurement_noise=0.1)
                
                object_z = self.zone_depth_filters[zone_id].update(raw_obj_z)
                effective_tolerance = config.DEPTH_TOLERANCE + structural_spread
                z_min, z_max = object_z - effective_tolerance, object_z + effective_tolerance
            else:
                z_min, z_max = -1, 2

            # 2. CHECK KEYPOINTS AGAINST 3D VOXEL
            # It looks at the depth of the object (e.g., a laptop at 5 meters away). It creates a "bubble" (Voxel) around that 5-meter mark
            for idx in check_indices:
                if idx >= len(pts) or conf[idx] < config.MIN_KEYPOINT_CONFIDENCE: continue
                px, py = int(pts[idx][0]), int(pts[idx][1])
                
                # 2D Check
                inside_2d = False
                if z['type'] == 'restricted':
                    zx1, zy1, zx2, zy2 = z['coords']
                    inside_2d = (zx1 <= px <= zx2 and zy1 <= py <= zy2)
                else:
                    inside_2d = (cv2.pointPolygonTest(z['polygon'], (px, py), False) >= 0)

                if inside_2d:
                    if depth_map is not None:
                        if 0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1]:
                            raw_hand_z = depth_map[py, px]
                            k_key = f"{person_id}_{z['type']}_{idx}"
                            if k_key not in self.person_depth_filters:
                                self.person_depth_filters[k_key] = KalmanSmoother(process_noise=0.005, measurement_noise=0.3)
                            hand_z = self.person_depth_filters[k_key].update(raw_hand_z)

                            if z_min <= hand_z <= z_max:
                                return True
                    else:
                        return True
        return False

    # The Comparison: It checks if the hand's 3D depth matches the laptop's 3D depth. If they overlap in 3D space, it says "True" (Someone is touching it!).
    def check_2d_overlap(self, bbox, zones):
        bx1, by1, bx2, by2 = bbox
        for z in zones:
            zx1, zy1, zx2, zy2 = z['coords']
            dx = min(bx2, zx2) - max(bx1, zx1)
            dy = min(by2, zy2) - max(by1, zy1)
            if dx > 0 and dy > 0:
                return True
        return False

    # Fixes the boxes if the camera is bumped or moved.
    def _compute_homography(self, frame):
        """
        ORB + BFMatcher + RANSAC against the reference frame.

        Returns (M, reason). M is None on failure and `reason` says why, so the
        caller can log a camera that has drifted out of match range instead of
        failing silently.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(curr_gray, None)

        if des is None:
            return None, "no descriptors in current frame"
        if self.ref_des is None:
            return None, "no descriptors in reference frame"

        matches = self.bf.match(self.ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) <= 10:
            return None, f"only {len(matches)} matches (need >10)"

        src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None, f"RANSAC found no homography from {len(matches)} matches"
        return M, None

    def align_zones(self, frame, zones):
        """
        Compares current frame to the STARTING frame.
        Ensures the box/polygon stays exactly where it was drawn on the background.

        Gated by config.ZONE_ALIGN_ENABLED / ZONE_ALIGN_INTERVAL: the homography
        is recomputed only every Nth processed frame and reused in between, since
        this is the most expensive stage in the pipeline. The interval counter
        lives here rather than in the caller because this object already owns the
        ORB reference state.
        """
        if not getattr(config, "ZONE_ALIGN_ENABLED", True):
            return zones

        self._align_calls += 1
        interval = max(1, int(getattr(config, "ZONE_ALIGN_INTERVAL", 15)))

        # Recompute on the first call and every Nth thereafter; also recompute
        # if we have never yet succeeded, so a failed first frame keeps retrying.
        due = ((self._align_calls - 1) % interval == 0) or self._last_homography is None
        if due:
            self.align_recomputes += 1
            M, reason = self._compute_homography(frame)
            if M is None:
                self.align_failures_total += 1
                self.align_consecutive_failures += 1
                logger.warning(
                    "align_zones: homography recompute failed at processed frame "
                    "%d (%s); reusing previous alignment "
                    "(consecutive failures=%d, total=%d)",
                    self._align_calls, reason,
                    self.align_consecutive_failures, self.align_failures_total,
                )
            else:
                self._last_homography = M
                self.align_consecutive_failures = 0

        M = self._last_homography
        if M is not None:
            for z in zones:
                # --- CASE A: RESTRICTED (RECTANGLES) ---
                if z['type'] == 'restricted':
                    x1, y1, x2, y2 = z['orig_coords']
                    pts_z = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts_z, M)
                        
                    z['coords'] = (int(dst[:,0,0].min()), int(dst[:,0,1].min()), int(dst[:,0,0].max()), int(dst[:,0,1].max()))
                    
                # --- CASE B: PASSIVE (POLYGONS) ---
                elif z['type'] == 'passive' and 'polygon' in z:
                        
                    if 'orig_polygon' not in z:
                        z['orig_polygon'] = np.array(z['polygon'], dtype=np.float32).reshape(-1, 1, 2)
                        
                    dst_poly = cv2.perspectiveTransform(z['orig_polygon'], M)
                    z['polygon'] = dst_poly.astype(np.int32)

        return zones

    def detect_theft(self, frame, zone):
        x1, y1, x2, y2 = zone['coords']
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        width, height = x2 - x1, y2 - y1
        if width < 15 or height < 15: return False

        curr_gray = cv2.GaussianBlur(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (3,3), 0)
        ref_gray = cv2.resize(zone['reference_patch'], (width, height))
        curr_edges = cv2.Canny(curr_gray, 70, 200)
        ref_edges = cv2.Canny(ref_gray, 70, 200)
        
        grid_h, grid_w = height // config.GRID_SIZE, width // config.GRID_SIZE
        cells_showing_loss = 0
        
        for r in range(config.GRID_SIZE):
            for c in range(config.GRID_SIZE):
                cy, cx = r * grid_h, c * grid_w
                c_cell = curr_edges[cy:cy+grid_h, cx:cx+grid_w]
                r_cell = ref_edges[cy:cy+grid_h, cx:cx+grid_w]
                
                curr_density = np.count_nonzero(c_cell)
                ref_density = np.count_nonzero(r_cell)
                
                if ref_density > 10: 
                    loss_percent = (ref_density - curr_density) / ref_density * 100
                    if loss_percent > config.THEFT_THRESHOLD:
                        cells_showing_loss += 1


        if cells_showing_loss >= 2:
            zone['missing_counter'] += 1
        else:
            zone['missing_counter'] = 0
            
        return zone['missing_counter'] > config.THEFT_FRAME_PERSISTENCE
    