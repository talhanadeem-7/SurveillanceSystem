from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config


logger = logging.getLogger(__name__)


def _get_reid_weights() -> Path:
    """
    Return ReID weights path for BotSort.
    Checks TRACKER_REID_WEIGHTS first, then the legacy STRONGSORT_REID_WEIGHTS
    key so existing config.py files keep working without changes.
    Falls back to osnet_x0_25_msmt17.pt which boxmot auto-downloads.
    """
    model_weights = (
        getattr(config, "TRACKER_REID_WEIGHTS", None)
        or getattr(config, "STRONGSORT_REID_WEIGHTS", None)
    )
    if model_weights:
        return Path(model_weights)
    return Path("osnet_x0_25_msmt17.pt")


class StrongSortTracker:
    """
    BotSort wrapper for boxmot >= 18.x.

    BoTrack no longer exists as a class in modern boxmot — it was merged into
    BotSort, which is the direct successor with identical capabilities:
      - ByteTrack two-stage motion matching (IoU first, ReID as tiebreaker)
      - OSNet ReID appearance to prevent wrong re-association in crowds
      - Kalman filter keeps tracks alive through long occlusions

    The class name is kept as StrongSortTracker so detector.py needs no changes.
    """

    def __init__(
        self,
        max_age: int = 1200,
        n_init: int = 1,
        max_iou_distance: float = 0.9,
        max_cosine_distance: float = 0.55,
        nn_budget: int = 150,
    ):
        try:
            from boxmot.trackers.botsort.botsort import BotSort
        except ImportError as exc:
            raise RuntimeError(
                "BotSort (boxmot) could not be imported.\n"
                "Fix: pip install boxmot --upgrade\n"
                f"Original error: {exc}"
            ) from exc

        device = (
            getattr(config, "TRACKER_DEVICE", None)
            or getattr(config, "STRONGSORT_DEVICE", None)
            or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        fp16 = bool(
            getattr(config, "TRACKER_FP16", None)
            if getattr(config, "TRACKER_FP16", None) is not None
            else getattr(config, "STRONGSORT_FP16", torch.cuda.is_available())
        )
        model_weights = _get_reid_weights()

        # BotSort constructor in boxmot 18.x
        # Confirmed params: reid_weights, device, half, det_thresh,
        #                   max_age, min_hits, iou_threshold, max_obs
        try:
            self.tracker = BotSort(
                reid_weights=model_weights,
                device=device,
                half=fp16,
                det_thresh=0.3,
                max_age=max_age,
                min_hits=n_init,
                iou_threshold=max_iou_distance,
                max_obs=max_age + 5,  # must be > max_age per boxmot docs
            )
            logger.info(
                "BotSort initialised — device=%s fp16=%s weights=%s",
                device, fp16, model_weights,
            )
        except TypeError as exc:
            # Fallback: minimal constructor if param names changed again
            logger.warning("BotSort full init failed (%s), trying minimal init.", exc)
            self.tracker = BotSort(
                reid_weights=model_weights,
                device=device,
                half=fp16,
            )

        self.match_iou_threshold = float(
            getattr(config, "TRACKER_MATCH_IOU_THRESHOLD", 0.2)
        )

    # ------------------------------------------------------------------
    # Public API used by detector.py
    # ------------------------------------------------------------------

    def update(
        self,
        xyxy: np.ndarray,
        confidences: np.ndarray,
        frame: np.ndarray,
        keypoints: object | None = None,
    ) -> np.ndarray:
        """
        Args:
            xyxy:        shape (N, 4) bounding boxes [x1, y1, x2, y2].
            confidences: shape (N,)   detection confidence scores.
            frame:       current BGR frame (H, W, 3).
            keypoints:   unused; kept for API compatibility with detector.py.

        Returns:
            np.ndarray of shape (N,) with track IDs for each detection.
            Detections that could not be matched get ID = -1.
        """
        del keypoints  # not used by BotSort

        if xyxy is None or len(xyxy) == 0:
            return np.array([], dtype=np.int32)

        if confidences is None or len(confidences) != len(xyxy):
            confidences = np.ones((len(xyxy),), dtype=np.float32)

        # boxmot expects: [x1, y1, x2, y2, confidence, class_id]
        dets = np.concatenate(
            [
                xyxy.astype(np.float32),
                confidences.reshape(-1, 1).astype(np.float32),
                np.zeros((len(xyxy), 1), dtype=np.float32),  # class 0 = person
            ],
            axis=1,
        )

        raw_tracks = self._tracker_update(dets, frame)
        track_boxes, track_ids = self._parse_tracks(raw_tracks)
        return self._assign_tracks_to_detections(
            xyxy.astype(np.float32), track_boxes, track_ids
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tracker_update(self, dets: np.ndarray, frame: np.ndarray):
        """Try the common boxmot update signatures in order."""
        methods = [
            lambda: self.tracker.update(dets, frame),
            lambda: self.tracker.update(dets=dets, img=frame),
            lambda: self.tracker.update(dets=dets, im=frame),
        ]
        last_exc: Exception | None = None
        for method in methods:
            try:
                return method()
            except TypeError as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            f"Could not call BotSort.update() with any known signature. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _parse_tracks(raw_tracks: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract boxes and IDs from tracker output.

        boxmot 18.x output shape: (N, 8)
        columns: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        """
        if raw_tracks is None:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)

        if isinstance(raw_tracks, np.ndarray):
            if raw_tracks.ndim != 2 or raw_tracks.shape[0] == 0 or raw_tracks.shape[1] < 5:
                return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)
            boxes = raw_tracks[:, :4].astype(np.float32)
            ids   = raw_tracks[:, 4].astype(np.int32)
            return boxes, ids

        # Fallback: object-based output (older boxmot versions)
        boxes: list[np.ndarray] = []
        ids:   list[int]        = []
        if isinstance(raw_tracks, (list, tuple)):
            for track in raw_tracks:
                track_id = getattr(track, "track_id", None)
                if track_id is None:
                    continue
                box = None
                if hasattr(track, "to_ltrb"):
                    box = track.to_ltrb()
                elif hasattr(track, "tlbr"):
                    box = track.tlbr
                if box is None:
                    continue
                box_arr = np.asarray(box, dtype=np.float32).reshape(-1)
                if box_arr.size < 4:
                    continue
                boxes.append(box_arr[:4])
                ids.append(int(track_id))

        if not boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)

        return np.stack(boxes, axis=0), np.asarray(ids, dtype=np.int32)

    def _assign_tracks_to_detections(
        self,
        det_boxes:   np.ndarray,
        track_boxes: np.ndarray,
        track_ids:   np.ndarray,
    ) -> np.ndarray:
        """
        Greedy IoU matching so output IDs are aligned with detector box order.
        Each detection gets at most one track; unmatched detections get ID = -1.
        """
        out = np.full((len(det_boxes),), -1, dtype=np.int32)
        if len(track_boxes) == 0:
            return out

        iou_matrix = self._pairwise_iou(det_boxes, track_boxes)
        used_det:   set[int] = set()
        used_track: set[int] = set()

        while True:
            det_idx, trk_idx = np.unravel_index(
                np.argmax(iou_matrix), iou_matrix.shape
            )
            best_iou = float(iou_matrix[det_idx, trk_idx])
            if best_iou < self.match_iou_threshold:
                break
            if det_idx in used_det or trk_idx in used_track:
                iou_matrix[det_idx, trk_idx] = -1.0
                continue

            out[det_idx] = int(track_ids[trk_idx])
            used_det.add(int(det_idx))
            used_track.add(int(trk_idx))
            iou_matrix[det_idx, :] = -1.0
            iou_matrix[:, trk_idx] = -1.0

            if len(used_det) == len(det_boxes) or len(used_track) == len(track_boxes):
                break

        return out

    @staticmethod
    def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorised IoU between every pair of boxes in a (N,4) and b (M,4)."""
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)

        ax1 = a[:, 0][:, None];  ay1 = a[:, 1][:, None]
        ax2 = a[:, 2][:, None];  ay2 = a[:, 3][:, None]
        bx1 = b[:, 0][None, :];  by1 = b[:, 1][None, :]
        bx2 = b[:, 2][None, :];  by2 = b[:, 3][None, :]

        inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
        inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
        inter_area = inter_w * inter_h

        area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
        area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
        union  = area_a + area_b - inter_area

        return np.where(union > 0.0, inter_area / union, 0.0).astype(np.float32)