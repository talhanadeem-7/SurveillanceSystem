# action_recognizer.py

import numpy as np
import collections
import config

class ActionRecognizer:
    def __init__(self, model_path=None):
        self.window_size = 30
        self.buffer = {} # {id: deque}
        self.states = {} # {id: {"is_bending": False}}

    def update(self, person_id, keypoints):
        kpts_x = keypoints.xyn[0][:, 0].cpu().numpy()
        kpts_y = keypoints.xyn[0][:, 1].cpu().numpy()
        conf = keypoints.conf[0].cpu().numpy()

        # Filter out zero-confidence points
        valid_y = kpts_y[conf > 0.3]
        valid_x = kpts_x[conf > 0.3]
        if len(valid_y) < 5: return None

        min_x, max_x = np.min(valid_x), np.max(valid_x)
        min_y, max_y = np.min(valid_y), np.max(valid_y)
        width = max_x - min_x
        height = max_y - min_y

        # Keypoints for advanced math
        # 9,10: Wrists | 11,12: Hips | 13,14: Knees | 15,16: Ankles
        avg_hip_y = (kpts_y[11] + kpts_y[12]) / 2
        avg_knee_y = (kpts_y[13] + kpts_y[14]) / 2
        avg_wrist_y = (kpts_y[9] + kpts_y[10]) / 2
        avg_ankle_y = (kpts_y[15] + kpts_y[16]) / 2

        current_state = {
            'center_y': avg_hip_y,
            'center_x': (kpts_x[11] + kpts_x[12]) / 2,
            'height': height,
            'aspect_ratio': height / (width + 1e-6),
            'hip_knee_dist': abs(avg_hip_y - avg_knee_y),
            'wrist_knee_diff': avg_wrist_y - avg_knee_y # Positive means hand is lower than knee
        }

        if person_id not in self.buffer:
            self.buffer[person_id] = collections.deque(maxlen=self.window_size)
            self.states[person_id] = {"was_bending": False}

        self.buffer[person_id].append(current_state)
        if len(self.buffer[person_id]) < 5: return None

        return self._classify(person_id)

    def _classify(self, person_id):
        history = self.buffer[person_id]
        curr = history[-1]
        prev = history[-5] # Look back 0.15s

        # 1. VELOCITY CALCULATION (Running vs Walking)
        dx = curr['center_x'] - prev['center_x']
        dy = curr['center_y'] - prev['center_y']
        velocity = np.sqrt(dx**2 + dy**2)

        # 2. PICKING UP (Sequence Logic)
        # If hands go below knees, person is 'bending'
        is_currently_bending = curr['wrist_knee_diff'] > config.PICKUP_HAND_KNEE_RELATION

        if self.states[person_id]["was_bending"] and not is_currently_bending:
            # They were bending and just stood up
            self.states[person_id]["was_bending"] = False
            return "Picking Up"

        if is_currently_bending:
            self.states[person_id]["was_bending"] = True
            return "Bending"

        # 3. STATIC POSES (Sitting / Lying)
        # Lying: Very flat aspect ratio
        if curr['aspect_ratio'] < 0.6:
            return "Lying Down"

        # Sitting: Vertical Hip-to-Knee distance is compressed
        # In standing, this is ~20-25% of height. In sitting, it's <15%
        if curr['hip_knee_dist'] < (curr['height'] * 0.15):
            return "Sitting"

        # 4. DYNAMIC MOVEMENT (Running vs Walking)
        if velocity > config.RUNNING_VELOCITY_THRESHOLD:
            return "Running"

        if velocity > 0.002: # Threshold for micro-movements
            return "Walking"

        return "Standing"