import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(override=True)
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "data", "uploaded_videos")
LOG_PATH = os.path.join(BASE_DIR, "storage", "event_logs.csv")
# Model Configuration
MODEL_PATH = "yolov8n-pose.pt"
YOLO_CPU_BACKEND = "onnx"  # Falls back to PyTorch if a validated export is absent.
YOLO_CPU_MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
CPU_ONNX_THREADS = 4
CONFIDENCE_THRESHOLD = 0.45
# CPU execution only; these settings do not opt any model into CUDA.
VISION_DEVICE = "cpu"
TRACKER_DEVICE = "cpu"
TRACKER_FP16 = False
CPU_TORCH_THREADS = 4
CPU_OPENCV_THREADS = 1
TRACKER_FUSE_OSNET = True  # Fold eval BatchNorm into convolutions, retaining FP32.
FRAME_SKIP = 2
INFERENCE_WIDTH = 640
PREVIEW_FPS = 10.0
# Source-frame interval: preserves the existing refresh cadence at FRAME_SKIP=2.
DEPTH_REFRESH_INTERVAL = 5
INACTIVE_TRACK_TTL = 600  # Source frames; greater than the tracker's lost-track buffer.
HEATMAP_MAX_POINTS_PER_TRACK = 18000
HEATMAP_MAX_TRACKS = 256
# Visual Colors (BGR)
COLOR_SECURE = (0, 255, 0)         # Green
COLOR_UNAUTHORIZED = (0, 165, 255) # Orange
COLOR_THEFT = (0, 0, 255)          # Red
COLOR_TRESPASSER = (0, 0, 0)       # Black
COLOR_DRAWING = (255, 0, 0)        # Blue
COLOR_AUTHORIZED = (255, 0, 0)     # Blue (for authorized personnel)
COLOR_PASSIVE = (0, 255, 255)      # Yellow
# Skeleton Links (COCO Format)
SKELETON_EDGES = [
    (5, 6), (5, 11), (6, 12), (11, 12), # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),    # Arms
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]
# Logic Settings
THEFT_THRESHOLD = 35.0    # If 35% of an object disappears, it triggers a theft alert.
GRID_SIZE = 8
THEFT_FRAME_PERSISTENCE = 37    # AI must see the object is missing for 37 frames in a row before shouting "Theft!"
# --- ZONE ALIGNMENT (ORB + RANSAC homography) ---
# align_zones() re-registers the drawn zones against frame 0 so they stay on the
# object if the camera is bumped.
#
# !! DISABLED BY DEFAULT. DO NOT RE-ENABLE WITHOUT READING THIS. !!
#
# 1. It does nothing useful on the deployment target. Fixed wall-mounted cameras
#    were measured at 0.10 px of frame-to-frame zone movement — alignment is
#    correcting a movement that is not there.
# 2. It is ACTIVELY HARMFUL on moving footage. On hand-held/phone clips it
#    produced 199 px mean frame-to-frame drift and threw the zone entirely
#    off-screen on 9 of 170 frames — while reporting ZERO failures, because the
#    homography succeeds and is nonsense (CLAUDE.md VF-37, OQ-17).
# 3. It is expensive: 47.6 ms/call on 1080p and 47.8 ms on 640x480 — the single
#    largest stage in the pipeline, 33-45% of total runtime, costing MORE per
#    frame than YOLO inference (VF-34). Disabling it took office cctv from
#    6.99 to 11.86 fps.
#
# So it spends a third of the frame budget to either do nothing or do damage.
# The code and both keys are kept deliberately: this is a default change, not a
# removal. Before re-enabling for a genuinely moving camera, implement OQ-17's
# sanity check on the computed homography — otherwise zones silently sit in the
# wrong place and intrusion decisions are made against them.
ZONE_ALIGN_ENABLED = False
# Recompute the homography every Nth PROCESSED frame; reuse the last one between
# recomputes. 1 = the old every-frame behaviour.
ZONE_ALIGN_INTERVAL = 15
# ORB feature budget. The cost is dominated by BFMatcher(crossCheck=True), which
# is O(n^2) in this number, NOT by feature detection — which is why the per-call
# cost is the same at 1080p and 640x480. Lowering this is a second, independent 
# lever on alignment cost, but there is a fixed ~21 ms floor from ORB detection
# that no reduction touches (VF-35).
# Left at 1000: lowering to 300 is 56% cheaper and looked accurate (mean 0.30 px
# corner error), but the test was INCONCLUSIVE, not positive — the test clips
# contain only 2-6 px of jitter and <0.1 deg of rotation, so there is no genuine
# camera motion to check degradation against (VF-40). Largely moot while
# ZONE_ALIGN_ENABLED is False.
ZONE_ALIGN_NFEATURES = 1000
# --- MiDaS DEPTH CONFIGURATION ---
DEPTH_MODEL_TYPE = "MiDaS_small" 
DEPTH_TOLERANCE = 0.10          # This decides how "close" a hand must be to an object in 3D space to count as a touch.
# --- DEPTH REASONING CONSTANTS ---
DEPTH_SCORE_THRESHOLD = 0.70       # 0.0 to 1.0 (Higher is stricter)
MIN_KEYPOINT_CONFIDENCE = 0.40     # Minimum confidence to trust a joint
MAX_INTERACTION_SCALE = 1.8        # If person is >1.8x taller than zone, they are too close
MIN_INTERACTION_SCALE = 0.6        # If person is <0.6x taller than zone, they are too far
# --- FACE RECOGNITION CONFIGURATION ---
FACE_DIR = os.path.join(BASE_DIR, "data", "authorized_faces")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "face_embeddings.pkl")
# ArcFace is state-of-the-art for identification
FACE_MODEL = "ArcFace" 
# 0.40 is a strict threshold for ArcFace
FACE_MATCH_THRESHOLD = 0.50  # How similar a face must be to the saved photo.
FACE_CHECK_INTERVAL = 30 #  check faces every 30 frames to save computer power.
# Re-check an already-identified person every N frames to make sure the name still matches.
FACE_REVALIDATE_INTERVAL = 20
# Drop a previously recognized identity after this many consecutive failed rechecks.
FACE_REVALIDATION_FAILS = 3
# How many repeated face hits in a row are needed before an identity is locked to a track.
FACE_CONFIRMATION_REQUIRED = 2
# --- TRACKER CONFIGURATION ---
# How many tracker updates a lost track is kept alive before boxmot drops it.
# NOTE: the unit is PROCESSED frames (one tracker update per processed frame),
# so the real-time meaning depends on FRAME_SKIP in streamlit_app.py.
# At FRAME_SKIP=2 on 30fps footage, 75 updates is about 5 seconds.
TRACKER_MAX_AGE = 75
# THIS -- not TRACKER_MAX_AGE -- is the lost-track lifetime knob.
# BotSort removes a lost track when `frame_count - end_frame > max_time_lost`,
# and max_time_lost is derived from track_buffer, never from max_age. Measured:
# a gap of 31 tracker updates preserves the track id, 32 drops it, and that
# boundary is UNCHANGED at max_age 5, 75 and 1200 -- but moves to 10 when
# track_buffer is set to 10. (CLAUDE.md VF-1; don't re-derive it.)
# 30 is boxmot's default and preserves current behaviour. Units are PROCESSED
# frames, so at FRAME_SKIP=2 on 30fps footage this is only ~2 seconds.
TRACKER_TRACK_BUFFER = 30
# Minimum IoU for the greedy matcher in StrongSortTracker to bind one of
# BotSort's emitted track boxes back to a YOLO detection.
# EXPOSED FOR VISIBILITY, NOT AS A WORKING LEVER: measurement showed it
# contributes ZERO unmatched detections. Every unmatched detection was starved --
# BotSort emitted fewer track boxes than there were detections -- and not one
# failed because its best IoU fell below this value (244/244 and 156/156 starved
# on the two crowd clips). Lowering it cannot conjure track boxes that were never
# emitted. See CLAUDE.md VF-8.
TRACKER_MATCH_IOU_THRESHOLD = 0.2
# Minimum detection confidence at which BotSort will CREATE a new track.
#
# !! MUST STAY ALIGNED WITH CONFIDENCE_THRESHOLD ABOVE !!
# Any value above CONFIDENCE_THRESHOLD reopens a dead band: detections whose
# confidence falls between the two are handed to the tracker but can never spawn
# a track, so they stay unmatched every frame and are emitted as ephemeral
# negative ids that accumulate NO state (no intrusion persistence, no face
# identity, no trajectory). If you change CONFIDENCE_THRESHOLD, change this too.
#
# boxmot's default is 0.6, which with CONFIDENCE_THRESHOLD=0.45 left 91% of
# unmatched detections stranded in the band. Lowering it to 0.45 cut ephemeral
# detections from 31% to 15% on the worst benchmark clip with no change to
# detection counts or track lifetimes, and the tracks it adds were verified to
# be genuine people, not duplicates. See CLAUDE.md VF-19/VF-20.
# Do NOT instead raise CONFIDENCE_THRESHOLD to 0.6: measured pre-1b, that
# fragmented the office cctv regression clip (identity count up by one, median
# track life halved 214 -> 115) and discarded 56% of detections on crowd sample.
# It drops a person mid-track and re-acquires them as a new id.
TRACKER_NEW_TRACK_THRESH = 0.45
# --- RE-ID SNAPSHOT / EMBEDDING BANKS ---
# Master switch for the in-house IdentityGuardian ReID layer that sits on top of
# boxmot's own OSNet ReID. When False, raw BotSort track ids pass through
# untouched. The class stays in the tree either way so the two can be compared.
IDENTITY_GUARDIAN_ENABLED = False
REID_SNAPSHOT_ROOT = os.path.join(BASE_DIR, "data", "reid_snapshots")
REID_SNAPSHOT_INTERVAL = 25
REID_SNAPSHOT_MIN_COUNT = 3
REID_EMBEDDING_MATCH_THRESHOLD = 0.72
REID_EMBEDDING_FALLBACK_THRESHOLD = 0.58
REID_EMBEDDING_DRIFT_THRESHOLD = 0.48
REID_IDENTITY_MATCH_MARGIN = 0.04
REID_MAX_EMBEDDINGS_PER_IDENTITY = 30
REID_REASSIGN_GRACE_FRAMES = 8
# --- LLM / RAG CONFIGURATION (NEW) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = GOOGLE_API_KEY 
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "chroma_db")
# Models
LLM_MODEL_NAME = "gpt-4o-mini"  
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # Standard Google Embeddings

# --- VLM ACTIVITY RECOGNITION (Vision-Language Model) ---
# Master switch for VLM-based human activity recognition.
# When enabled, the system analyzes tracked person activities using OpenAI's
# vision model, providing higher-level semantic understanding beyond pose.
# Does NOT run on every frame; triggered at configurable intervals.
USE_VLM = True
# Compatibility for older entry points; USE_VLM is the canonical setting.
VLM_ACTIVITY_ENABLED = USE_VLM
VLM_SHUTDOWN_TIMEOUT = 3.0
VLM_INACTIVE_TRACK_TTL = 300  # Source frames.

# Activity analysis trigger interval (source frames).
# 30 source frames = 1 second at 30 FPS, or 0.5 seconds at 60 FPS.
# Lower values = more frequent analysis (more API calls, more latency).
# Higher values = less frequent (cheaper but slower to detect activity changes).
VLM_ACTIVITY_ANALYSIS_INTERVAL = 30

# Temporal window size for frame buffering.
# Tracks are kept in a sliding window of this many frames for context.
# When analysis is triggered, frames are sampled from this window.
VLM_ACTIVITY_WINDOW_SIZE = 30

# Sample rate within the temporal window.
# Sends every Nth frame to the VLM to reduce data volume and API cost.
# Sample rate of 3 means frames 0, 3, 6, ... are included.
# Value of 1 sends every frame (high data volume, better temporal resolution).
VLM_ACTIVITY_SAMPLE_RATE = 3

# VLM model to use for activity analysis.
# Must be a vision-capable OpenAI model.
VLM_MODEL_NAME = "gpt-4o-mini"

# API timeout for VLM calls (seconds).
# If a VLM request takes longer than this, it is aborted.
# Set higher if you have poor connectivity; lower for responsive UI.
VLM_ACTIVITY_API_TIMEOUT = 10.0

# Minimum track age (frames) before activity analysis is attempted.
# Very new tracks may have insufficient context for reliable analysis.
# At FRAME_SKIP=2, 15 frames ≈ 1 second of real time.
VLM_ACTIVITY_MIN_TRACK_AGE = 15

# Only analyze tracks with at least this many unique positions in the buffer.
# Prevents analyzing stationary people (e.g., always standing in same spot).
# NOTE: read by nothing today. Kept as a documented intent, not a live knob.
VLM_ACTIVITY_MIN_MOVEMENT = 5

# --- VLM COST / RATE CONTROL ---
# Measured prompt tokens for one call on real crops (gpt-4o-mini):
#   11 images, detail=auto : 85,319   <-- blew the 200k tokens/min account limit
#   11 images, detail=low  : 28,649
#    5 images, detail=low  : 14,484
#    4 images, detail=low  : 11,651
# Every one of those settings returned the correct label ("sitting"), so the
# cheap settings cost accuracy nothing on this footage.
#
# detail="low" bills a flat ~2833 tokens per image instead of tiling it. This is
# the single biggest lever -- a 3x cut on its own. Set to None for full detail.
VLM_ACTIVITY_IMAGE_DETAIL = "low"

# Maximum images per call. Token cost is linear in this.
VLM_ACTIVITY_MAX_IMAGES = 5

# Wall-clock floor between VLM calls, across ALL tracks.
# VLM_ACTIVITY_ANALYSIS_INTERVAL counts SOURCE frames, so its real-time meaning
# changes with clip frame rate and machine speed -- it cannot bound API spend.
# This can. Measured at 14,484 tokens/call:
#   5.0s -> 13.3 calls/min -> 192,788 tok/min  (only 7k headroom -- too tight)
#   6.0s -> ~10  calls/min -> ~145,000 tok/min (~55k left for the analyst)
# The 200k/min limit is SHARED with the RAG analyst's embedding and chat calls,
# so the VLM must not spend all of it. Raise this if you still see 429s.
VLM_ACTIVITY_MIN_SECONDS_BETWEEN_CALLS = 6.0

# On HTTP 429 the analyzer backs off globally, doubling per occurrence up to
# this cap, and decays back down as calls start landing again.
VLM_ACTIVITY_MAX_BACKOFF_SECONDS = 60.0

# SDK-level retries. Kept low: the SDK sleeps inside our worker thread, holding
# the per-track in-flight slot. Our own backoff handles rate limiting.
VLM_ACTIVITY_API_MAX_RETRIES = 1

# Longest side (pixels) of each image sent to the VLM, and JPEG quality.
# The API tiles images internally, so sending source-resolution lossless PNG
# only bought upload latency -- measured at ~4 MB per call before this was
# added. Latency is what bounds label freshness, because only one analysis per
# track is in flight at a time.
VLM_ACTIVITY_MAX_IMAGE_SIDE = 512
VLM_ACTIVITY_JPEG_QUALITY = 85

# How long (in SOURCE frames) an activity label stays valid for display.
# A VLM verdict describes the ~1 s window it was computed from, not the present.
# Past this age the label is dropped rather than shown as if it were current --
# a seated person was displaying "walking" from several seconds earlier because
# results never expired.
# MUST be >= the source frames that elapse between calls, or labels blank out
# between refreshes: at the 6.0s floor and ~10 processed fps with FRAME_SKIP=2,
# that is 6 * 10 * 2 = ~120 source frames. 200 leaves margin on slower machines
# while still bounding how old a displayed label can be.
VLM_ACTIVITY_LABEL_TTL = 200


