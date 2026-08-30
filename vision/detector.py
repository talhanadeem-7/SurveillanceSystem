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

        # Monotonic counter for detections the tracker could not match.
        # Ephemeral ids are negative and are NEVER reused for the lifetime of
        # this detector, so two different people can no longer end up sharing
        # an id. The previous scheme, -(idx + 1), was positional: the person at
        # detection index 0 was always -1, so unrelated people collided across
        # frames in every per-id dict downstream.
        self._next_ephemeral_id = -1

    @staticmethod
    def _guardian_enabled() -> bool:
        """Master switch for the IdentityGuardian ReID layer (default off)."""
        return bool(getattr(config, "IDENTITY_GUARDIAN_ENABLED", False))

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
            # Still drive the tracker with an empty detection set. This branch
            # is the ACTUAL empty-scene path (YOLO found nobody), and skipping
            # the tracker here froze its frame_count, so lost tracks never aged
            # and were re-matched to whoever appeared next. The matching guard
            # in StrongSortTracker.update is not enough on its own, because
            # this early return means it is never reached.
            self.tracker.update(np.empty((0, 4), dtype=np.float32),
                                np.empty((0,), dtype=np.float32), frame)
            if self._guardian_enabled():
                self.guardian.update(frame, np.empty((0, 4), dtype=np.float32),
                                     np.empty((0,), dtype=np.int32))
            return result

        xyxy_np = result.boxes.xyxy.detach().cpu().numpy()
        conf_np = result.boxes.conf.detach().cpu().numpy()

        # Step 1 — BotSort raw ids (may contain switches)
        raw_ids = self.tracker.update(
            xyxy_np, conf_np, frame, keypoints=result.keypoints
        )

        # Step 2 — IdentityGuardian corrects switches using appearance.
        # Gated by config.IDENTITY_GUARDIAN_ENABLED: when disabled, raw BotSort
        # ids pass straight through so BotSort's own OSNet ReID can be measured
        # on its own. Read per call so an A/B harness can toggle it between runs.
        if self._guardian_enabled():
            corrected_ids = self.guardian.update(frame, xyxy_np, raw_ids)
        else:
            corrected_ids = raw_ids.copy()

        # Step 3 — give every unmatched detection a unique, never-reused
        # negative id. Callers must treat negative ids as EPHEMERAL: draw them
        # and evaluate them for the current frame, but never accumulate
        # cross-frame state against them (see run_surveillance).
        display_ids = corrected_ids.copy()
        for idx, tid in enumerate(display_ids):
            if tid < 0:
                display_ids[idx] = self._next_ephemeral_id
                self._next_ephemeral_id -= 1

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