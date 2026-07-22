# Shelby: Intelligent Surveillance Dashboard

Shelby is a multi-stage video surveillance system that combines person detection, pose tracking, zone-based intrusion detection, theft detection, face recognition, appearance-based ID repair, movement heatmaps, and an LLM-powered analyst over logged events.

The project has two main entry points:
- `streamlit_app.py` for the full web dashboard
- `app.py` for the terminal/OpenCV version of the same surveillance pipeline

## What the system does

At a high level, Shelby can:
- Upload a video and read it frame by frame.
- Let the user define restricted and passive zones on the first frame.
- Track people with pose estimation and BotSort.
- Correct tracker ID switches with appearance-based re-identification.
- Detect trespassing into restricted zones.
- Detect passive-zone entry using feet and 3D depth checks.
- Detect theft by checking whether a restricted object visually disappears.
- Recognize faces of authorized people.
- Log every security event to CSV.
- Build heatmaps and trajectory plots from movement history.
- Summarize the logs with an LLM-powered analyst chatbot.

## Project structure

### Top-level files

| File / Folder | Purpose |
| --- | --- |
| `streamlit_app.py` | Main web dashboard with zone drawing, surveillance, analysis, and heatmap tabs. |
| `app.py` | Terminal/OpenCV version of the same surveillance pipeline. |
| `config.py` | Central configuration for thresholds, paths, model names, and feature settings. |
| `requirements.txt` | Python dependencies used by the project. |
| `.env` | Stores the OpenAI API key used by the chatbot and embeddings. |
| `README.md` | This document. |
| `FYP_Proposal_extracted.txt` | Project proposal/reference material. |
| `yolov8n.pt`, `yolov8n-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt` | Local YOLO weights bundled with the project. |

### Main folders

| Folder | Purpose |
| --- | --- |
| `vision/` | Core computer-vision logic: detection, tracking, re-ID, face recognition, and depth reasoning. |
| `storage/` | CSV event logging and audit trail helpers. |
| `utils/` | UI helpers, zone drawing, CSV log formatting, heatmap/trajectory generation. |
| `agents/` | RAG analyst and log retrieval for the chatbot. |
| `chatbot/` | Chat wrapper / placeholder chatbot support files. |
| `data/` | Input and generated runtime data such as faces, embeddings, snapshots, and uploaded videos. |

### Runtime data folders

| Folder | Purpose |
| --- | --- |
| `data/authorized_faces/` | Reference face images for enrolled authorized people. |
| `data/embeddings/face_embeddings.pkl` | Cached face embeddings generated from the authorized faces folder. |
| `data/embeddings/chroma_db/` | Persistent Chroma vector database for log retrieval. |
| `data/reid_snapshots/` | Appearance snapshots used by the identity guardian for ID correction. |
| `data/uploaded_videos/` | Uploaded surveillance videos. |
| `data/processed_frames/` | Generated/processed frames, if the app writes them. |
| `data/snapshots/` | Runtime snapshot output folder. |
| `storage/event_logs.csv` | Main event log file used by the analyst. |
| `storage/video_metadata.csv` | Video metadata output file. |

### Placeholder modules

These files currently exist but are empty placeholders in the repository:

- `agents/report_generator.py`
- `chatbot/prompt_templates.py`
- `chatbot/memory.py`
- `vision/event_generator.py`
- `vision/frame_processor.py`

They are useful extension points if you later want to add report exports, long-term chatbot memory, or a separate frame-processing layer.

## Core runtime flow

### 1) Streamlit startup

When you run `streamlit run streamlit_app.py`, the app does the following:

1. Loads configuration from `config.py` and `.env`.
2. Creates the Streamlit page shell and initializes session state.
3. Instantiates the heavy models once:
   - YOLO pose detector.
   - BotSort tracker wrapper.
   - IdentityGuardian re-identification layer.
   - Event logger.
   - Face recognizer.
4. Lets the user upload a video.
5. Saves the uploaded video under `data/uploaded_videos/`.
6. Reads the first frame for zone drawing.
7. Allows the user to draw and name zones.
8. Starts the surveillance loop when the engine is launched.
9. After processing, unlocks the Shelby Analyst tab for log-based Q&A.

### 2) Per-frame backend flow

For each processed frame, Shelby performs the following steps:

1. Read the frame from the video.
2. Skip frames according to the frame-skip policy to save compute.
3. Optionally downscale the frame for faster inference.
4. Align user-defined zones to the current frame using ORB homography.
5. Run YOLO pose detection.
6. Run BotSort tracking to keep track IDs stable.
7. Correct tracker switches using IdentityGuardian.
8. Rescale boxes and keypoints back to the original frame size.
9. Record motion points for heatmaps and trajectory visualisation.
10. Decide whether a person is in a passive zone or restricted zone.
11. Run MiDaS depth checks only when a person is close enough to a zone.
12. Run face recognition at intervals for unknown people.
13. Revalidate already-identified people at intervals.
14. Log intrusion, access, theft, removal, and identity events.
15. Draw overlays on the frame and push the result to the UI.

### 3) What happens when the analyst tab opens

When the surveillance run finishes, or when cached logs already exist, the analyst tab can be initialized:

1. `SecurityAnalyst` loads the log retriever.
2. The retriever reads `storage/event_logs.csv`.
3. The log rows are converted into natural-language sentences.
4. The sentences are embedded and stored in Chroma.
5. User questions are embedded and matched against the log vectors.
6. The retrieved context is passed into the OpenAI chat model.
7. The model answers in a forensic, story-like style.

## Workflow diagram

```mermaid
flowchart LR
    A[Start streamlit_app.py] --> B[Load config and session state]
    B --> C[Upload video]
    C --> D[Read first frame]
    D --> E[Draw and name zones]
    E --> F[Start surveillance loop]
    F --> G[Frame skip and optional downscale]
    G --> H[ORB zone alignment]
    H --> I[YOLO pose detection]
    I --> J[BotSort tracking]
    J --> K[IdentityGuardian ReID correction]
    K --> L[Rescale boxes and keypoints]
    L --> M[Depth check and trespass logic]
    L --> N[Face recognition]
    M --> O[Event logger CSV]
    N --> O
    O --> P[Heatmap and trajectory history]
    O --> Q[Chroma vector DB]
    Q --> R[Shelby Analyst RAG chatbot]
    P --> S[Heatmap tab]
    R --> T[Chat response]
```

## How the 100-frame example works

The live Streamlit loop uses these runtime choices:

- `FRAME_SKIP = 2`
- `INFERENCE_WIDTH = 640`
- `DEPTH_REFRESH_INTERVAL = 5`
- `PERSISTENCE_THRESHOLD = 5`

For a 100-frame video:

- The app still reads all 100 frames from the file.
- Only every second frame is analyzed for inference, so roughly 50 frames go through YOLO/tracking.
- If the original video is 30 fps, the surveillance engine effectively analyzes about 15 frames per second.
- Depth estimation is not run on every frame; it is only refreshed when needed and then reused for a few frames.
- Heatmap positions are recorded only for the frames that are actually processed, so the heatmap reflects the sampled motion path, not every raw video frame.

This design keeps the app usable on weaker hardware while still preserving the main security signals.

## Module-by-module explanation

### `vision/detector.py`

This is the pose-detection wrapper. It:

- Loads the Ultralytics YOLO model from `config.MODEL_PATH`.
- Detects only class `0` (person).
- Sends boxes to BotSort for stable tracking.
- Sends the tracked result into IdentityGuardian to fix ID switches.
- Rebuilds the Ultralytics `Results` object with corrected track IDs.

### `vision/tracker.py`

This wraps BotSort from `boxmot`.

BotSort is used because it combines motion matching and appearance-based ReID. That makes it much better than a simple IoU tracker when people overlap, turn around, or briefly disappear behind objects.

The tracker uses:

- `TRACKER_REID_WEIGHTS` or `STRONGSORT_REID_WEIGHTS` if configured.
- GPU/CPU selection from `config.TRACKER_DEVICE` or the available CUDA device.
- FP16 support when CUDA is available.

### `vision/identity_guardian.py`

This sits on top of BotSort and corrects ID switches using appearance descriptors.

It builds an embedding from:

- Hair region colour.
- Upper-body colour.
- Lower-body colour.
- Silhouette edge histogram.

This is useful when BotSort assigns a new ID after occlusion. IdentityGuardian attempts to map the new ID back to the original one.

### `vision/pose_analyzer.py`

This is the spatial reasoning engine. It handles:

- ORB homography to keep zones aligned if the camera moves.
- MiDaS monocular depth to add 3D awareness.
- Restricted-zone checks using hand keypoints.
- Passive-zone checks using foot keypoints.
- Theft detection by comparing object edges over time.
- Kalman smoothing to reduce flicker in depth measurements.

### `vision/face_recognition.py`

This handles face enrollment and recognition using DeepFace with ArcFace.

It:

- Loads saved face embeddings from `data/embeddings/face_embeddings.pkl` if present.
- Otherwise generates them from the images in `data/authorized_faces/`.
- Extracts face crops from YOLO keypoints.
- Filters low-quality crops.
- Compares face embeddings with cosine distance.
- Returns a known name or `Person_X` when the person is unknown.

### `storage/event_logger.py`

This writes the CSV audit trail.

It logs:

- Timestamp
- Entity
- Action
- Status
- Location

It also debounces repeated events so the CSV is not spammed with identical entries every frame.

### `utils/video_utils.py`

This provides the terminal/OpenCV zone-selection UI.

It supports:

- Restricted zones drawn as rectangles.
- Passive zones drawn as polygons.
- Zone naming.
- Reference patch generation for theft detection.
- Drawing boxes, skeletons, and labels on frames.

### `utils/polygon_utils.py`

This contains the polygon drawing helper used by the CLI zone selector.

It supports:

- Left-click to add points.
- Right-click to close a polygon.
- Escape to cancel.

### `utils/heatmap.py`

This generates movement visualizations.

It provides:

- `record_position()` to store person positions per frame.
- `generate_heatmap()` to show where a person spent the most time.
- `generate_trajectory()` to show movement order and path.
- `generate_combined()` to overlay both.
- `draw_heatmap_legend()` for the colour bar.
- `get_display_name()` to turn `Person_12` into a known name if face recognition has identified the track.

### `utils/csv_utils.py`

This reads the event logs and converts them into LLM-friendly text.

It provides:

- `load_data()` to read the CSV safely.
- `get_recent_logs_as_text()` for short summaries.
- `get_statistics()` for high-level counts.
- `get_all_logs_formatted()` for RAG ingestion.

### `agents/retriever.py`

This builds the Chroma vector database from the event logs.

It:

- Reads the CSV logs.
- Converts each log entry into a LangChain `Document`.
- Uses `OpenAIEmbeddings` with `text-embedding-3-small`.
- Stores the embeddings in `data/embeddings/chroma_db/`.
- Performs similarity search for the chatbot.

### `agents/reasoning_agent.py`

This is the main analyst brain.

It:

- Loads the retriever.
- Loads `ChatOpenAI` with `gpt-4o-mini`.
- Passes retrieved log context and the user question into a large prompt.
- Produces a human-style forensic answer.

### `chatbot/chat_engine.py`

This is the terminal chatbot wrapper.

It:

- Starts `SecurityAnalyst`.
- Reads user input in a loop.
- Prints the response.
- Supports `exit`, `quit`, and `q`.

### Empty / placeholder modules

These files currently do not contain runtime logic, but they are part of the project structure:

- `agents/report_generator.py`
- `chatbot/prompt_templates.py`
- `chatbot/memory.py`
- `vision/event_generator.py`
- `vision/frame_processor.py`

## Libraries and what they are used for

### Computer vision and tracking

| Library | Purpose |
| --- | --- |
| `ultralytics` | YOLOv8 pose detection and the results object used throughout the pipeline. |
| `boxmot` | BotSort multi-object tracking with motion matching and ReID appearance matching. |
| `opencv-python` | Video I/O, drawing, ORB matching, Canny edges, homography, and UI windows. |
| `torch` | MiDaS inference, YOLO backend support, GPU acceleration, tensor operations. |
| `torchvision` | Vision utilities and supporting tensor ops. |

### Depth and 3D reasoning

| Library | Purpose |
| --- | --- |
| `timm` | Supports MiDaS model loading. |
| MiDaS via `torch.hub` | Monocular depth estimation for proximity reasoning. |

MiDaS is invoked at runtime to produce monocular depth maps that the pipeline consults when determining proximity (for example, when a hand or foot approaches a restricted or passive zone). Depth estimation is run sparingly (see `DEPTH_REFRESH_INTERVAL`) to limit GPU work.

To reduce frame-to-frame jitter in per-person and per-zone depth readings, the system applies a Kalman-style smoother (see `vision/pose_analyzer.py`'s `KalmanSmoother`). This stabilises proximity checks and theft heuristics so decisions do not flicker when depth estimates are noisy.

### Face recognition

| Library | Purpose |
| --- | --- |
| `deepface` | Face embedding generation. |
| ArcFace model | Face identity representation used by DeepFace. |
| `tf-keras` | Backend support required by DeepFace on this setup. |

### LLM and retrieval

| Library | Purpose |
| --- | --- |
| `langchain` | Orchestration layer for prompt templates and retrieval-based chat. |
| `langchain-openai` | OpenAI embeddings and chat model wrappers. |
| `langchain-community` | Chroma vector store integration. |
| `chromadb` | Local vector database for log similarity search. |
| `python-dotenv` | Loads API keys from `.env`. |

### Data and utilities

| Library | Purpose |
| --- | --- |
| `pandas` | CSV ingestion and log analysis. |
| `pydantic` | Schema and validation support from LangChain dependencies. |
| `tqdm` | Progress display for longer operations. |
| `Pillow` | Image handling for Streamlit canvas and uploads. |
| `numpy` | Array and geometry operations. |
| `einops` | Tensor reshaping support used by some model stacks. |

### UI

| Library | Purpose |
| --- | --- |
| `streamlit` | Web dashboard. |
| `streamlit-drawable-canvas` | Interactive rectangle and polygon drawing. |

## Thresholds and what they do

These are the most important tuning values in `config.py` and the surveillance loop.

### Detection and scene layout

| Setting | Default | What it controls | If you increase it | If you decrease it |
| --- | --- | --- | --- | --- |
| `MODEL_PATH` | `yolov8m-pose.pt` | YOLO pose model used for person detection | Larger models can be more accurate but slower | Smaller models are faster but less accurate |
| `CONFIDENCE_THRESHOLD` | `0.55` | Minimum YOLO confidence to keep a person detection | Fewer false positives, but more missed people | More detections, but more false positives |
| `FRAME_SKIP` | `2` | How many frames to skip during live processing | Faster runtime, less temporal detail | Slower runtime, more accurate motion capture |
| `INFERENCE_WIDTH` | `640` | Downscaled width used for inference | More detail, but heavier computation | Faster inference, but more detail loss |
| `DEPTH_REFRESH_INTERVAL` | `5` | How often the depth map is recomputed | Less GPU work, but depth becomes stale sooner | More accurate depth updates, but slower runtime |

### Restricted zone / theft detection

| Setting | Default | What it controls | If you increase it | If you decrease it |
| --- | --- | --- | --- | --- |
| `THEFT_THRESHOLD` | `35.0` | Percent edge loss in a grid cell needed to treat that cell as damaged or missing | More tolerant to partial occlusion | More sensitive to disappearance, but more false theft alerts |
| `GRID_SIZE` | `8` | Number of grid cells per side for theft comparison | Finer detection of missing objects, more computation | Coarser detection, less sensitive |
| `THEFT_FRAME_PERSISTENCE` | `37` | Number of consecutive frames the theft condition must remain true | Slower alerts, fewer false alarms | Faster alerts, more false alarms |

How edge-based disappearance detection works:

- The system computes an edge map (Canny) of the reference patch for each restricted zone and splits that patch into a `GRID_SIZE x GRID_SIZE` grid (default `8x8`). Each cell stores a local edge fingerprint used for comparison.
- During runtime the same region is re-sampled and an edge map is computed; the algorithm compares local edge energy per cell against the reference. If a cell loses more than `THEFT_THRESHOLD` percent of its edge energy and that condition persists for `THEFT_FRAME_PERSISTENCE` frames, the cell is marked as damaged/missing and a theft/removal event is raised for the object covering that cell.
- Using a grid (default `GRID_SIZE = 8`) localises disappearance to specific parts of the zone: finer grids (larger `GRID_SIZE`) allow detecting small, local losses (useful for small objects), while coarser grids make the check more tolerant to small movements or lighting changes.

This per-cell comparison improves resilience to partial occlusions and small viewpoint shifts while allowing the system to report which part of a zone changed visually.

### Depth-based trespassing

| Setting | Default | What it controls | If you increase it | If you decrease it |
| --- | --- | --- | --- | --- |
| `DEPTH_MODEL_TYPE` | `MiDaS_small` | Which MiDaS depth backbone is used | Better depth quality with larger models, but slower | Faster depth estimation with lower accuracy |
| `DEPTH_TOLERANCE` | `0.10` | How close the hand depth must be to the zone/object depth | More forgiving, more likely to count a touch | Stricter, fewer touch detections |
| `DEPTH_SCORE_THRESHOLD` | `0.70` | General confidence threshold for 3D interaction logic | More permissive, more positives | Stricter, fewer positives |
| `MIN_KEYPOINT_CONFIDENCE` | `0.40` | Minimum keypoint confidence for hand/foot checks | Uses only very reliable joints | Accepts weaker joints, but more noisy decisions |
| `MAX_INTERACTION_SCALE` | `1.8` | If a person is much larger than the zone, they are considered too close | More tolerant to close-up interactions | Stricter about proximity |
| `MIN_INTERACTION_SCALE` | `0.6` | If a person is too small relative to the zone, they are considered too far away | Requires closer presence to count as interaction | More tolerant of distant detections |

### Face recognition

| Setting | Default | What it controls | If you increase it | If you decrease it |
| --- | --- | --- | --- | --- |
| `FACE_MODEL` | `ArcFace` | Face embedding model used by DeepFace | Not usually changed in this project | Not usually changed |
| `FACE_MATCH_THRESHOLD` | `0.57` | Cosine distance threshold for a match | Looser matching, more false accepts | Stricter matching, more false rejects |
| `FACE_MATCH_MARGIN` | `0.02` | How much better the best match must be than the second-best | More conservative identity assignment | More likely to assign an identity when close matches exist |
| `FACE_CHECK_INTERVAL` | `20` | How often unknown people are checked for a face match | Less compute, slower recognition | More compute, quicker recognition |
| `FACE_CONFIRMATION_REQUIRED` | `4` | How many repeated face hits are needed before identity is locked | More stable, slower confirmation | Faster confirmation, more risk of wrong identity |
| `FACE_REVALIDATE_INTERVAL` | `20` | How often a known identity is rechecked | Fewer rechecks, faster runtime | More rechecks, more safety against drift |
| `FACE_REVALIDATION_FAILS` | `3` | How many consecutive recheck failures revoke a known identity | More tolerant to temporary misses | More aggressive revocation |
| `FACE_KEYPOINT_MIN_CONF` | `0.65` | Minimum YOLO keypoint confidence for face crop keypoints | Only very clear face keypoints are accepted | More keypoints are allowed, but crops become less reliable |
| `FACE_MIN_KEYPOINTS` | `4` | Minimum number of valid face keypoints required for a crop | Requires a more complete face view | Accepts weaker face views, but more false crops |
| `FACE_MIN_AREA_RATIO` | `0.015` | Minimum crop area ratio relative to the whole frame | Rejects tiny face crops | Accepts smaller face crops |
| `FACE_MIN_BLUR_VAR` | `80.0` | Minimum sharpness for a usable face crop | Requires sharper faces | Allows blurrier faces, but matching quality drops |
| `FACE_MIN_CONFIDENCE` | `0.85` | Minimum confidence derived from the best match score | Stricter acceptance of face IDs | More matches accepted, but with more risk |
| `FACE_SINGLE_ID_THRESHOLD` | `0.22` | Special threshold when only one enrolled identity exists | More conservative single-person matching | Easier single-person matching, but more false positives |
| `FACE_MATCH_RATIO_THRESHOLD` | `0.85` | Ratio between best and second-best match distance | Requires a clearer winner | Allows more ambiguous matches |

### Re-ID / appearance correction

| Setting | Default | What it controls | If you increase it | If you decrease it |
| --- | --- | --- | --- | --- |
| `REID_SNAPSHOT_INTERVAL` | `25` | How often IdentityGuardian stores appearance snapshots | More history, more storage and compute | Less history, faster but less robust re-ID |
| `REID_SNAPSHOT_MIN_COUNT` | `3` | Minimum snapshots needed before an identity bank is useful | More stable embeddings | Faster enrollment, weaker history |
| `REID_EMBEDDING_MATCH_THRESHOLD` | `0.72` | Main similarity threshold for Re-ID matching | More strict, fewer mistaken merges | More permissive, more chance of wrong ID merges |
| `REID_EMBEDDING_FALLBACK_THRESHOLD` | `0.58` | Backup similarity threshold when confidence is weaker | More strict fallback | More tolerant fallback |
| `REID_EMBEDDING_DRIFT_THRESHOLD` | `0.48` | When appearance is considered drifted over time | More sensitive to change | More tolerant of clothing/pose variation |
| `REID_IDENTITY_MATCH_MARGIN` | `0.04` | Best match must beat second-best by this margin | Safer identity assignment | More permissive assignment |
| `REID_MAX_EMBEDDINGS_PER_IDENTITY` | `30` | How many appearance samples are kept per identity | Better long-term modelling, more memory | Less memory, less temporal coverage |
| `REID_REASSIGN_GRACE_FRAMES` | `8` | Frames allowed before reassigning an identity | More tolerant to brief occlusion | Faster reassignments |

### BotSort / tracking

`vision/tracker.py` uses BotSort from `boxmot`.

- `max_age = 1200` means tracks can survive for a long time through temporary disappearance.
- `n_init = 1` means tracks become active quickly.
- `max_iou_distance = 0.9` allows loose IoU matching.
- `max_cosine_distance = 0.55` sets the ReID distance tolerance.
- `nn_budget = 150` limits how much appearance history is used.

In practice:
- Higher `max_age` keeps IDs alive longer after occlusion.
- Lower `max_age` makes stale tracks disappear sooner.
- Higher IoU tolerance can reduce track breaks, but may merge nearby people incorrectly.
- Lower cosine distance is stricter and safer, but may break IDs more often.

### Heatmap and trajectory visualization

The heatmap tab uses runtime samples recorded from each person track.

- `sigma` in the heatmap controls how spread out the blobs are.
  - Higher `sigma` creates smoother, broader blobs.
  - Lower `sigma` creates sharper, more precise blobs.
- `alpha` controls overlay opacity.
  - Higher `alpha` makes the heatmap dominate the frame.
  - Lower `alpha` keeps the camera image more visible.

## Log ingestion and RAG analyst

The analyst uses `agents/retriever.py` and `agents/reasoning_agent.py`.

1. `storage/event_logger.py` writes structured events into `storage/event_logs.csv`.
2. `utils/csv_utils.py` formats those rows into natural language sentences.
3. `agents/retriever.py` embeds those sentences with `OpenAIEmbeddings` and stores them in Chroma.
4. `agents/reasoning_agent.py` retrieves the most relevant logs and asks `ChatOpenAI` to explain them.
5. The response is shown in the Shelby Analyst tab or in the CLI chat engine.

Important note:
- The analyst requires a valid `OPENAI_API_KEY`.
- If the OpenAI quota is exhausted, the chatbot will show a rate-limit or insufficient-quota error.
- The surveillance pipeline itself can still run without the analyst tab.

## Example presentation narrative

If you need to explain this system to colleagues, use this story:

"A video is uploaded into Shelby. The app takes the first frame and lets the operator draw restricted and passive zones. Then the surveillance loop starts. YOLO finds people, BotSort assigns track IDs, IdentityGuardian repairs ID switches, and PoseAnalyzer checks whether hands or feet enter the protected areas. If the person is not recognized, face matching tries to identify them. All events are written to a CSV log, and the system remembers each track so it can later generate heatmaps and trajectories. Once the video run is complete, the analyst tab loads the logs, stores them in a Chroma vector database, and lets the user ask what happened in natural language."

## Setup and run

Use the Python 3.11 environment in this workspace.

```powershell
cd C:\Users\TALHA\Downloads\SurveillanceSystem
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```powershell
streamlit run streamlit_app.py
```

Run the terminal version:

```powershell
python app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\venv311\Scripts\Activate.ps1
```

## Data folders and what they store

- `data/authorized_faces/` - face images used to enroll known people.
- `data/embeddings/face_embeddings.pkl` - cached face embeddings.
- `data/embeddings/chroma_db/` - persistent vector database for RAG.
- `data/reid_snapshots/` - appearance snapshots used by IdentityGuardian.
- `data/uploaded_videos/` - uploaded footage.
- `data/processed_frames/` - processed output frames if saved.
- `data/snapshots/` - extra screenshots or evidence assets.
- `storage/event_logs.csv` - main audit log used by the analyst.
- `storage/video_metadata.csv` - video metadata storage.

## Current empty or placeholder files

These files exist in the repository but are currently empty or effectively placeholders:

- `agents/report_generator.py`
- `chatbot/prompt_templates.py`
- `chatbot/memory.py`
- `vision/event_generator.py`
- `vision/frame_processor.py`

They are useful extension points if you later want to add report exports, long-term chatbot memory, or a separate frame-processing layer.

## Troubleshooting

- If BotSort fails to import, install `boxmot` in `venv311`.
- If the analyst tab shows OpenAI quota errors, check your API key and billing.
- If the face system cannot recognize anyone, make sure images exist in `data/authorized_faces/` and that they contain clear faces.
- If zone placement seems wrong, check whether the video was resized before inference; the app rescales boxes and keypoints back to original coordinates.

## Short summary

Shelby is a layered surveillance pipeline:

YOLO detects people, BotSort tracks them, IdentityGuardian stabilizes IDs, PoseAnalyzer handles zone and depth reasoning, FaceIdentifier names known people, EventLogger stores the story, and the RAG analyst turns that story into a chat experience.

## Proposal items mentioned in the FYP document but not yet implemented

The original FYP proposal (`FYP_Proposal_extracted.txt`) lists several features and deployment options. Below are the key items that are mentioned in the proposal but are not implemented (or only partially implemented) in this codebase yet, together with planned next steps.

- **SQL database (SQLite/PostgreSQL) for events and metadata**: currently events are written to `storage/event_logs.csv` and the analyst uses Chroma for vector search. Plan: add a relational backend (SQLite for single-host, PostgreSQL for production) and a migration path from CSV into a structured events table.
- **Action recognition (trained action models)**: the proposal suggests running action recognition models to classify activities (walking, interacting, suspicious behavior). Presently the system relies on pose/keypoint heuristics and zone rules; plan: integrate or train an action-recognition model and add a pipeline stage for temporal action classification.
- **Cross-camera / multi-camera tracking**: proposal mentions cross-camera person tracking. Current implementation supports single-stream processing; plan: extend `video_loader.py` and tracker logic to handle multiple synchronized streams and a global identity map.
- **Scalable real-time monitoring (multi-camera, deployment)**: the repo supports single-machine runs. Plan: add multi-stream ingestion, queueing, and a lightweight server mode for real-time deployments (optionally containerised).
- **Cloud deployment / Docker / platform integration**: cloud options (GCP/Azure/AWS) and Docker were listed as potential targets but are not configured. Plan: provide Dockerfiles, helm charts or deployment guides for a chosen cloud target.
- **User management and access control (UI + DB integration)**: the proposal envisions a user interface for queries and access control; current UI is Streamlit without user accounts. Plan: add auth layers and a web frontend or integration with a simple user table in the SQL backend.
- **Comprehensive evaluation experiments**: the proposal described experiments and metrics (precision/recall/F1) for action detection and tracking. Remaining work: create evaluation scripts, test datasets, and automated reporting to validate models.

If you'd like, I can: (a) add a short migration script to copy `storage/event_logs.csv` into a local SQLite database, (b) scaffold a `Dockerfile` and `docker-compose.yml` for the Streamlit app + optional Postgres, or (c) begin integrating an off-the-shelf action-recognition model as a first pass. Tell me which to prioritise and I'll add it to the todo list and implement the scaffold.
