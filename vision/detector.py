import os

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

import config
from vision.tracker import StrongSortTracker
from vision.identity_guardian import IdentityGuardian


class PoseDetector:
    """
    YOLO pose detector + BotSort tracker + IdentityGuardian ReID correction.

    Pipeline per frame
    ------------------
    1. YOLO detects all persons and their keypoints.
    2. BotSort assigns track ids (may switch ids on occlusion/reappearance).
    3. IdentityGuardian compares appearance embeddings and reverts any
       switched ids back to the original one — in the same frame.
    """

    def __init__(self):
        self.model   = YOLO(config.MODEL_PATH)
        self.tracker = StrongSortTracker()

        # Snapshot folder — sits next to config.py
        snapshot_root = getattr(
            config, "REID_SNAPSHOT_ROOT",
            os.path.join(os.path.dirname(os.path.abspath(config.__file__)),
                         "data", "reid_snapshots")
        )
        self.guardian = IdentityGuardian(snapshot_root=snapshot_root)

    def track_and_detect(self, frame):
        """
        Args:
            frame: BGR numpy array (original or inference-resolution).

        Returns:
            Ultralytics Results object with corrected, stable track ids.
        """
        results = self.model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False,
            classes=[0],  # person only
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            self.guardian.update(frame, np.empty((0, 4), dtype=np.float32),
                                 np.empty((0,), dtype=np.int32))
            return result

        xyxy_np = result.boxes.xyxy.detach().cpu().numpy()
        conf_np = result.boxes.conf.detach().cpu().numpy()

        # Step 1 — BotSort raw ids (may contain switches)
        raw_ids = self.tracker.update(
            xyxy_np, conf_np, frame, keypoints=result.keypoints
        )

        # Step 2 — IdentityGuardian corrects switches using appearance
        corrected_ids = self.guardian.update(frame, xyxy_np, raw_ids)

        # Step 3 — patch negative ids with temporary display ids
        display_ids = corrected_ids.copy()
        for idx, tid in enumerate(display_ids):
            if tid < 0:
                display_ids[idx] = -(idx + 1)

        # Step 4 — rebuild Ultralytics Boxes with corrected ids
        valid_indices  = list(range(len(corrected_ids)))
        tracked_result = result[valid_indices]

        id_tensor = torch.as_tensor(
            display_ids[valid_indices],
            device=tracked_result.boxes.xyxy.device,
            dtype=tracked_result.boxes.xyxy.dtype,
        ).unsqueeze(1)

        tracked_data = torch.cat(
            [
                tracked_result.boxes.xyxy,
                id_tensor,
                tracked_result.boxes.conf.unsqueeze(1),
                tracked_result.boxes.cls.unsqueeze(1),
            ],
            dim=1,
        )
        tracked_result.boxes = Boxes(tracked_data, tracked_result.orig_shape)

        return tracked_result