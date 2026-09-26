"""
heatmap_utils.py
================
Standalone utilities for generating person movement heatmaps and trajectory
overlays for the Shelby Surveillance System.

Two visualizations are provided:
  1. Gaussian density heatmap  — shows WHERE a person spent the most time.
  2. Colour-gradient trajectory — shows the exact PATH and movement ORDER
     (blue = early frames, red = late frames).

Both are returned as BGR numpy arrays suitable for st.image() display or
cv2.imwrite() saving.  Neither function modifies session state or imports
any Streamlit symbols, so they can be unit-tested independently.

Usage (from streamlit_app.py):
    from utils.heatmap_utils import (
        record_position,
        generate_heatmap,
        generate_trajectory,
        generate_combined,
        get_display_name,
    )
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
import config
from typing import Dict, List, Tuple, Optional, Deque


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# positions store: {track_id: [(cx, cy, frame_num), ...]}
PositionStore = Dict[int, Deque[Tuple[int, int, int]]]


# ---------------------------------------------------------------------------
# Data collection helper (called inside the surveillance loop)
# ---------------------------------------------------------------------------

def record_position(
    store: PositionStore,
    track_id: int,
    bbox: np.ndarray,
    frame_id: int,
) -> None:
    """
    Record the foot-centre position of a tracked person.

    The foot-centre (bottom-centre of the bounding box) is used instead of
    the box centre because it stays on the floor plane and gives a more
    accurate "where did this person stand" reading.

    Args:
        store:    The shared position dict kept in session state.
        track_id: Integer track ID assigned by BoTrack / StrongSORT.
        bbox:     [x1, y1, x2, y2] numpy array (original resolution).
        frame_id: Current frame counter from the surveillance loop.
    """
    x1, y1, x2, y2 = bbox[:4]
    cx = int((x1 + x2) / 2)
    # Use the centre of the bounding box rather than the foot (y2).
    # Foot position sounds logical but pushes heat to the wrong area when
    # the person is sitting, leaning over a desk, or detected in the upper
    # portion of the frame. Box-centre tracks the actual person body location.
    cy = int((y1 + y2) / 2)
    if track_id not in store:
        limit = max(1, int(getattr(config, "HEATMAP_MAX_TRACKS", 256)))
        if len(store) >= limit:
            oldest = min(store, key=lambda tid: store[tid][-1][2] if store[tid] else -1)
            del store[oldest]
        store[track_id] = deque(maxlen=max(1, int(getattr(config, "HEATMAP_MAX_POINTS_PER_TRACK", 18000))))
    store[track_id].append((cx, cy, frame_id))


# ---------------------------------------------------------------------------
# Core visualization functions
# ---------------------------------------------------------------------------

def generate_heatmap(
    positions: List[Tuple[int, int, int]],
    reference_frame: np.ndarray,
    sigma: int = 35,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Generate a Gaussian density heatmap overlaid on the reference frame.

    Each recorded position contributes a Gaussian "blob" to a 2-D
    accumulator.  The accumulator is normalised and colourised, then
    alpha-blended on top of the reference frame.

    Args:
        positions:       List of (cx, cy, frame_id) tuples for ONE person.
        reference_frame: BGR frame used as background (first frame of video).
        sigma:           Gaussian blur radius in pixels.  Larger = smoother
                         but less spatially precise.  25 px works well for
                         640-wide footage; increase for higher resolutions.
        alpha:           Heatmap opacity (0.0 = invisible, 1.0 = opaque).
        colormap:        OpenCV colourmap.  COLORMAP_JET gives the classic
                         blue-green-yellow-red scale.  COLORMAP_INFERNO is
                         a good dark-background alternative.

    Returns:
        BGR numpy array — reference frame with heatmap overlay.
    """
    h, w = reference_frame.shape[:2]
    accumulator = np.zeros((h, w), dtype=np.float32)

    for cx, cy, _ in positions:
        # clamp to frame bounds
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        accumulator[cy, cx] += 1.0

    if accumulator.max() == 0:
        # No data — return plain reference frame
        return reference_frame.copy()

    # Gaussian blur spreads each point into a smooth heat radius
    blurred = cv2.GaussianBlur(accumulator, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Normalise to 0-255
    normalised = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    heat_uint8 = normalised.astype(np.uint8)

    # Colourmap: low density = dark blue, high density = bright red
    coloured = cv2.applyColorMap(heat_uint8, colormap)

    # Mask out zero regions so the background shows through cleanly
    mask = heat_uint8 > 2          # low threshold so sparse paths still show
    mask_3ch = np.stack([mask] * 3, axis=-1)

    output = reference_frame.copy()
    blended = cv2.addWeighted(output, 1.0 - alpha, coloured, alpha, 0)
    # Only blend where there is actual heat
    output[mask_3ch] = blended[mask_3ch]

    return output


def generate_trajectory(
    positions: List[Tuple[int, int, int]],
    reference_frame: np.ndarray,
    line_thickness: int = 2,
    dot_radius: int = 5,
    arrow_freq: int = 10,
) -> np.ndarray:
    """
    Draw a colour-gradient trajectory showing movement direction over time.

    The path is drawn as a polyline where colour transitions from
    BLUE (earliest position) → GREEN → YELLOW → RED (latest position),
    so the viewer can immediately read the movement chronology.
    Arrows are drawn at regular intervals to reinforce direction.

    Args:
        positions:       List of (cx, cy, frame_id) tuples for ONE person.
        reference_frame: BGR frame used as background.
        line_thickness:  Width of the trajectory line in pixels.
        dot_radius:      Radius of start/end marker dots.
        arrow_freq:      Draw a direction arrow every N points.

    Returns:
        BGR numpy array — reference frame with trajectory overlay.
    """
    output = reference_frame.copy()

    if len(positions) < 2:
        return output

    # Sort by frame number to guarantee chronological order
    pts = sorted(positions, key=lambda p: p[2])
    n = len(pts)

    # Draw line segments with colour gradient
    for idx in range(1, n):
        t = idx / (n - 1)           # 0.0 (start) → 1.0 (end)

        # Colour interpolation: blue → cyan → green → yellow → red
        # Using HSV: hue 240° (blue) → 0° (red)
        hue = int((1.0 - t) * 120)  # 120 = blue-ish, 0 = red
        colour_hsv = np.uint8([[[hue, 255, 255]]])
        colour_bgr = cv2.cvtColor(colour_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

        p1 = (int(pts[idx - 1][0]), int(pts[idx - 1][1]))
        p2 = (int(pts[idx][0]),     int(pts[idx][1]))

        cv2.line(output, p1, p2, colour_bgr, line_thickness, cv2.LINE_AA)

        # Draw directional arrows at regular intervals
        if idx % arrow_freq == 0:
            cv2.arrowedLine(
                output, p1, p2,
                colour_bgr,
                line_thickness + 1,
                cv2.LINE_AA,
                tipLength=0.4,
            )

    # Start marker — blue filled circle with white border
    sx, sy = int(pts[0][0]), int(pts[0][1])
    cv2.circle(output, (sx, sy), dot_radius + 2, (255, 255, 255), -1)
    cv2.circle(output, (sx, sy), dot_radius,     (255, 100, 0),   -1)
    cv2.putText(output, "START", (sx + 8, sy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "START", (sx + 8, sy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0),   1, cv2.LINE_AA)

    # End marker — red filled circle with white border
    ex, ey = int(pts[-1][0]), int(pts[-1][1])
    cv2.circle(output, (ex, ey), dot_radius + 2, (255, 255, 255), -1)
    cv2.circle(output, (ex, ey), dot_radius,     (0, 50, 255),    -1)
    cv2.putText(output, "END", (ex + 8, ey - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "END", (ex + 8, ey - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 50, 255),    1, cv2.LINE_AA)

    return output


def generate_combined(
    positions: List[Tuple[int, int, int]],
    reference_frame: np.ndarray,
    sigma: int = 35,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Generate heatmap + trajectory overlaid on the same reference frame.

    The heatmap is rendered first (lower layer) and the trajectory path
    is drawn on top so both are readable simultaneously.

    Args:
        positions:       List of (cx, cy, frame_id) tuples for ONE person.
        reference_frame: BGR frame used as background.
        sigma:           Gaussian sigma for the heatmap layer.
        alpha:           Heatmap opacity (slightly lower than standalone so
                         the trajectory line remains visible on top).

    Returns:
        BGR numpy array — reference frame with both overlays.
    """
    # Layer 1: heatmap
    with_heat = generate_heatmap(positions, reference_frame, sigma=sigma, alpha=alpha)
    # Layer 2: trajectory on top of heatmap
    combined  = generate_trajectory(positions, with_heat)
    return combined


# ---------------------------------------------------------------------------
# Legend / colour bar helper
# ---------------------------------------------------------------------------

def draw_heatmap_legend(
    image: np.ndarray,
    label: str = "Low → High Density",
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Draw a small horizontal colour bar legend in the bottom-right corner.

    Args:
        image:    BGR image to annotate in-place (a copy is made).
        label:    Text label placed below the colour bar.
        colormap: Must match the colourmap used in generate_heatmap().

    Returns:
        Copy of the image with legend drawn.
    """
    out = image.copy()
    h, w = out.shape[:2]

    bar_w, bar_h = 160, 14
    margin = 10
    x0 = w - bar_w - margin
    y0 = h - bar_h - 30

    # Build gradient strip
    gradient = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, -1)
    gradient_3ch = np.repeat(gradient, bar_h, axis=0)
    coloured_bar = cv2.applyColorMap(gradient_3ch, colormap)

    out[y0:y0 + bar_h, x0:x0 + bar_w] = coloured_bar
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    cv2.putText(
        out, label,
        (x0, y0 + bar_h + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# Utility: resolve display name from identity_map
# ---------------------------------------------------------------------------

def get_display_name(track_id: int, identity_map: dict) -> str:
    """
    Return the human-readable label for a track ID.

    If the track has been face-identified, return the recognised name.
    Otherwise return 'Person_<id>'.
    """
    name = identity_map.get(track_id)
    return name if name else f"Person_{track_id}"
