"""
identity_guardian.py
====================
Appearance-based ID correction layer that sits on top of BotSort.

Problem it solves
-----------------
BotSort assigns a NEW id every time a person reappears after occlusion.
This class detects that switch and reverts the id to the original one
in the same frame, before anything else in the pipeline sees it.

How it works
------------
1. First time a track id is seen → crop the person, build an embedding,
   save snapshot to  reid_snapshots/<id>/  folder.

2. Every frame → check all ACTIVE ids. If a brand-new id appears that
   was NOT seen in the previous frame (= sudden appearance), compare its
   embedding against ALL known-dead ids. If similarity > threshold,
   the new id is a re-appeared old person → revert to old id instantly.

3. Embedding is a concatenation of:
   - Upper-body HSV histogram  (shirt colour)
   - Lower-body HSV histogram  (trouser / skirt colour)
   - Hair-region HSV histogram (top 12% of crop)
   - Grayscale edge histogram  (body silhouette shape)
   All normalised to unit length so cosine similarity works cleanly.

Usage (inside detector.py)
--------------------------
    guardian = IdentityGuardian(snapshot_root="reid_snapshots")
    corrected_ids = guardian.update(frame, xyxy_array, raw_botsort_ids)
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Minimum crop height to bother building an embedding (pixels).
MIN_CROP_H = 60
MIN_CROP_W = 25

# How many frames a track must be gone before it becomes a "re-id candidate".
# Set to 1 so even a single-frame disappearance triggers re-id.
FRAMES_MISSING_BEFORE_REID = 1

# Cosine similarity threshold to accept a re-id match.
# 0.82 means embeddings must be very close — lowers false merges.
REID_SIMILARITY_THRESHOLD = 0.82

# After this many frames missing, the identity is truly gone (not just occluded).
# At 15fps and FRAME_SKIP=2 that's ~30 real seconds. Raise if people leave the
# scene and come back after a very long time.
MAX_FRAMES_MISSING = 450

# Number of snapshots to keep per identity (rolling average embedding).
MAX_SNAPSHOTS = 8

# How many frames a new id must be present before we lock in its embedding.
# This prevents us from building an embedding on a half-visible detection.
MIN_FRAMES_TO_ENROLL = 2


# ---------------------------------------------------------------------------
# Embedding builder
# ---------------------------------------------------------------------------

def _build_embedding(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Build a rich appearance descriptor from a person crop.

    Returns a normalised float32 vector, or None if the crop is too small.
    """
    h, w = crop_bgr.shape[:2]
    if h < MIN_CROP_H or w < MIN_CROP_W:
        return None

    crop = cv2.resize(crop_bgr, (64, 128), interpolation=cv2.INTER_LINEAR)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    parts: List[np.ndarray] = []

    # --- 1. Hair region (top 12%) ---
    hair_h = max(1, int(128 * 0.12))
    hair   = hsv[:hair_h, :, :]
    parts.append(_hsv_hist(hair))

    # --- 2. Upper body / shirt (12%–50%) ---
    upper = hsv[hair_h: int(128 * 0.50), :, :]
    parts.append(_hsv_hist(upper))

    # --- 3. Lower body / trousers (50%–100%) ---
    lower = hsv[int(128 * 0.50):, :, :]
    parts.append(_hsv_hist(lower))

    # --- 4. Edge histogram — silhouette shape ---
    edges = cv2.Canny(gray, 40, 120)
    # Divide into 8 horizontal bands and count edge pixels in each
    band_h = 128 // 8
    edge_hist = np.array(
        [edges[i * band_h:(i + 1) * band_h, :].sum() for i in range(8)],
        dtype=np.float32,
    )
    parts.append(edge_hist)

    vec = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    return vec / norm


def _hsv_hist(region: np.ndarray) -> np.ndarray:
    """16-bin H + 8-bin S histogram for a region (already in HSV)."""
    h_hist = cv2.calcHist([region], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([region], [1], None, [8],  [0, 256]).flatten()
    vec    = np.concatenate([h_hist, s_hist]).astype(np.float32)
    norm   = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else vec


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalised vectors."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# IdentityGuardian
# ---------------------------------------------------------------------------

class IdentityGuardian:
    """
    Corrects BotSort ID switches using appearance snapshots.

    Drop-in: feed it the raw BotSort ids every frame, get corrected ids back.
    """

    def __init__(self, snapshot_root: str = "reid_snapshots"):
        self.snapshot_root = Path(snapshot_root)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

        # track_id -> rolling list of embeddings (averaged for matching)
        self._embeddings:   Dict[int, List[np.ndarray]] = defaultdict(list)

        # track_id -> how many consecutive frames seen (used for enrolment gate)
        self._frame_count:  Dict[int, int] = defaultdict(int)

        # track_id -> how many consecutive frames MISSING
        self._missing:      Dict[int, int] = defaultdict(int)

        # ids seen in the previous frame
        self._prev_ids:     set = set()

        # id remapping: new_id -> original_id (active redirects)
        self._id_map:       Dict[int, int] = {}

        # ids that have been fully retired (gone too long)
        self._retired:      set = set()

        logger.info("IdentityGuardian initialised. Snapshots → %s", self.snapshot_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_bgr: np.ndarray,
        xyxy: np.ndarray,
        raw_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            frame_bgr : full BGR frame (original resolution).
            xyxy      : (N, 4) bounding boxes in original-frame pixel coords.
            raw_ids   : (N,)   track ids from BotSort (may contain switches).

        Returns:
            corrected_ids: (N,) ids after appearance-based correction.
        """
        if len(raw_ids) == 0:
            self._tick_missing(set())
            self._prev_ids = set()
            return raw_ids.copy()

        current_ids = set(int(i) for i in raw_ids if i > 0)

        # Apply any active remaps first (handles multi-frame corrections)
        working_ids = np.array(
            [self._id_map.get(int(i), int(i)) for i in raw_ids],
            dtype=np.int32,
        )

        # Detect brand-new ids that weren't in the previous frame
        newly_appeared = {
            int(i) for i in raw_ids
            if int(i) > 0 and int(i) not in self._prev_ids
        }

        fh, fw = frame_bgr.shape[:2]

        for idx, raw_id in enumerate(raw_ids):
            raw_id = int(raw_id)
            if raw_id <= 0:
                continue

            corrected_id = self._id_map.get(raw_id, raw_id)
            box = xyxy[idx]
            crop = self._safe_crop(frame_bgr, box, fh, fw)

            if crop is None:
                working_ids[idx] = corrected_id
                continue

            # --- ENROLMENT: build / update embedding for known ids ---
            if raw_id not in self._id_map:
                # It's a "real" id (not a remapped one) — enrol it
                self._frame_count[corrected_id] += 1
                if self._frame_count[corrected_id] >= MIN_FRAMES_TO_ENROLL:
                    emb = _build_embedding(crop)
                    if emb is not None:
                        self._update_embedding(corrected_id, emb)
                        self._save_snapshot(corrected_id, crop)

            # --- RE-ID: new id that just appeared → try to match ---
            if raw_id in newly_appeared and raw_id not in self._id_map:
                matched_id = self._match_against_known(raw_id, crop, current_ids)
                if matched_id is not None:
                    logger.info(
                        "IdentityGuardian: ID %d → reverted to %d (similarity match)",
                        raw_id, matched_id,
                    )
                    self._id_map[raw_id] = matched_id
                    corrected_id = matched_id
                    # Merge frame count into the original identity
                    self._frame_count[matched_id] = max(
                        self._frame_count[matched_id],
                        self._frame_count.get(raw_id, 0),
                    )

            working_ids[idx] = corrected_id

        # Update missing counters for ids not seen this frame
        self._tick_missing(current_ids)
        self._prev_ids = current_ids

        return working_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_embedding(self, track_id: int, emb: np.ndarray) -> None:
        """Rolling update: keep up to MAX_SNAPSHOTS embeddings per identity."""
        bank = self._embeddings[track_id]
        bank.append(emb)
        if len(bank) > MAX_SNAPSHOTS:
            bank.pop(0)

    def _mean_embedding(self, track_id: int) -> Optional[np.ndarray]:
        bank = self._embeddings.get(track_id)
        if not bank:
            return None
        mean = np.mean(bank, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-6 else None

    def _match_against_known(
        self,
        new_id: int,
        crop: np.ndarray,
        active_ids: set,
    ) -> Optional[int]:
        """
        Compare a new track's appearance against all dead/missing identities.
        Returns the best matching old id, or None if no match above threshold.
        """
        new_emb = _build_embedding(crop)
        if new_emb is None:
            return None

        best_id:  Optional[int] = None
        best_sim: float         = REID_SIMILARITY_THRESHOLD  # must beat this

        for cand_id, bank in self._embeddings.items():
            # Skip ids that are currently active (they're already tracked)
            if cand_id in active_ids:
                continue
            # Skip ids that have been retired (gone too long)
            if cand_id in self._retired:
                continue
            # Skip if the candidate hasn't been missing long enough
            if self._missing.get(cand_id, 0) < FRAMES_MISSING_BEFORE_REID:
                continue

            cand_emb = self._mean_embedding(cand_id)
            if cand_emb is None:
                continue

            sim = _cosine_sim(new_emb, cand_emb)
            if sim > best_sim:
                best_sim = sim
                best_id  = cand_id

        if best_id is not None:
            logger.debug(
                "ReID match: new_id=%d → old_id=%d  sim=%.3f",
                new_id, best_id, best_sim,
            )
            # Reset missing counter — they're back
            self._missing[best_id] = 0

        return best_id

    def _tick_missing(self, active_ids: set) -> None:
        """
        Increment missing counters for absent ids.
        Retire ids that have been gone too long.
        """
        all_known = set(self._embeddings.keys())
        for tid in all_known:
            if tid not in active_ids:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > MAX_FRAMES_MISSING:
                    self._retired.add(tid)
            else:
                self._missing[tid] = 0

    def _save_snapshot(self, track_id: int, crop: np.ndarray) -> None:
        """Save one snapshot image per identity (overwrites on update)."""
        folder = self.snapshot_root / str(track_id)
        folder.mkdir(parents=True, exist_ok=True)
        count = len(self._embeddings.get(track_id, []))
        path  = folder / f"snap_{count:03d}.jpg"
        try:
            cv2.imwrite(str(path), crop)
        except Exception as exc:
            logger.debug("Snapshot save failed for id %d: %s", track_id, exc)

    @staticmethod
    def _safe_crop(
        frame: np.ndarray,
        box: np.ndarray,
        fh: int,
        fw: int,
    ) -> Optional[np.ndarray]:
        """Clip box to frame bounds and return the crop, or None if degenerate."""
        x1, y1, x2, y2 = box[:4]
        x1 = max(0, int(x1));  y1 = max(0, int(y1))
        x2 = min(fw, int(x2)); y2 = min(fh, int(y2))
        if x2 - x1 < MIN_CROP_W or y2 - y1 < MIN_CROP_H:
            return None
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "known_identities":  len(self._embeddings),
            "active_remaps":     len(self._id_map),
            "retired_ids":       len(self._retired),
        }