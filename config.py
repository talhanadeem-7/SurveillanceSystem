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
CONFIDENCE_THRESHOLD = 0.45
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
# --- LLM / RAG CONFIGURATION (NEW) ---is the mebedding causing ther 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = GOOGLE_API_KEY 
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "chroma_db")
# Models
LLM_MODEL_NAME = "gpt-4o-mini"  
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # Standard Google Embeddings
# --- ACTION RECOGNITION (ST-GCN) ---
ACTION_MODEL_PATH = os.path.join(BASE_DIR, "data", "weights", "st_gcn.kinetics.pt")
ACTION_WINDOW = 30
# --- ENHANCED ACTION THRESHOLDS ---
RUNNING_VELOCITY_THRESHOLD = 0.025  # How fast a person must move to be "Running."
BENDING_RATIO_THRESHOLD = 0.7       # How much the body must fold to count as "Bending."
PICKUP_HAND_KNEE_RELATION = 0.05    # Hands must be below knees to trigger "Picking Up"