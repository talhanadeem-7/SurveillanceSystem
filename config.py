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
# --- RE-ID SNAPSHOT / EMBEDDING BANKS ---
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