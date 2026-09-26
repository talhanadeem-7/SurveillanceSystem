# CPU surveillance performance — 23 September 2026

## Implemented

- `config.py`: `USE_VLM` is the canonical activity-recognition switch; the old name remains an alias. Explicit CPU/FP32 devices, thread counts, preview rate, frame/depth intervals, shutdown grace, and retention limits are configurable.
- `vision/cpu_runtime.py`, `vision/detector.py`, `scripts/export_cpu_pose.py`: FP32 dynamic-shape ONNX execution of the existing YOLOv8n-pose weights. Ultralytics still handles preprocessing and pose postprocessing. Source/export SHA-256 checks reject mismatched exports; missing dependencies or exports fall back to PyTorch. Four ONNX threads, one inter-op thread, and no idle spinning prevent contention with subsequent vision stages.
- `vision/tracker.py`: fold 79 OSNet Conv/BatchNorm pairs at load time, retaining FP32 embeddings and ReID on every analyzed frame. No tracker thresholds or association algorithm changes.
- `vision/pose_analyzer.py`: cache raw zone depth statistics while the depth map/geometry is unchanged, preserve Kalman update cadence, cache reference edge counts, vectorize the original theft grid calculation, use inference mode, and guard constant-depth normalization.
- `streamlit_app.py`: grab every source frame but retrieve only the existing every-second-frame analysis selection; reuse box arrays and zone lists; cap preview drawing/conversion/publishing separately at 10 FPS; show processed/source FPS and elapsed time. Reset tracker/identity state between videos and keep saved zone definitions separate from per-run counters. RAG initialization and heatmap rendering require explicit UI actions.
- `vision/vlm_activity_analyzer.py`: disabled mode creates no client or executor and collects no images; enabled mode resizes crops before buffering, skips sample preparation when throttled, expires inactive buffers, rejects callbacks from previous runs, and drains requests for up to three seconds at EOF. Late requests are invalidated and reported. Final fresh results are displayed. Prompt instructions discourage unsupported conversation/intent claims.
- `utils/heatmap.py`: retain at most 256 tracks and 18,000 points per track. Other inactive per-track state and depth filters are pruned after the configured grace period. These limits change historical retention, not detection decisions.

## Measurements

Hardware detected: AMD Ryzen 5 5600, 6 cores/12 logical processors; PyTorch 2.11.0+cpu. The user's RTX 3060 and 32 GB RAM remain the target machine context; these optimizations do not use CUDA. Thread sweep selected four PyTorch threads and one OpenCV thread: 60 detection/tracking frames took 3.88–3.89 seconds with four threads versus 3.93 with six, 4.40–4.43 with two, and 5.98 with one. Thread tuning alone was a small improvement.

Full `streamlit_app.run_surveillance()` was exercised through Streamlit AppTest, with real detection, tracking, depth, face matching, and (where enabled) VLM calls. No `app.py`, chatbot, or RAG calls were run. A restricted test rectangle used the same zone/reference-patch format as the UI. Camera-shake zone alarms were excluded from correctness scoring.

The table uses steady **analyzed frames/second**, excluding initial setup and final drain. Both baseline and optimized off-mode runs used the same clips, thresholds, resolution, frame skip, and test harness, without cProfile. Browser transport/rendering and live RTSP were not measured. Figures are individual runs, not confidence intervals; short-clip and cold-model effects cause variability.

| Clip | Before, VLM off | Optimized, VLM off | Improvement | Optimized, VLM on |
|---|---:|---:|---:|---:|
| sample12 | 11.72 | 14.60 | 24.6% | 14.15 |
| sample11 | 10.08 | 11.84 | 17.5% | 13.20 |
| sample20 | 13.34 | 16.87 | 26.5% | 16.64 |

VLM on/off differences are noisy, particularly the short sample11 run; enabling VLM does not accelerate inference. The baseline off-mode processing times including setup were 22.68 / 17.06 / 38.99 seconds; optimized off-mode times were 20.36 / 14.52 / 30.83 seconds. sample12 includes first-use MiDaS loading. The three roughly 60-FPS clips contain 418 / 344 / 1040 source frames and analyze 209 / 172 / 520 frames.

The isolated same-weight FP32 detector comparison was **41.46 ms PyTorch versus 25.00 ms ONNX** over 30 sampled frames. The model remains YOLOv8n-pose: 3.29M fused parameters and 9.2 GFLOPs at 640 input; the ONNX artifact is about 12.6 MB versus the 6.5 MB checkpoint. This improves execution speed without switching learned weights or quantizing.

## Accuracy and lifecycle checks

- On the three complete sample videos, all 901 analyzed frames retained identical detection counts and track IDs. Maximum bounding-box coordinate difference was 0.00120 pixels; maximum keypoint data difference was 0.00070. The PyTorch-only optimization pass was bit-identical for boxes/keypoints and IDs on the checked sample videos.
- Additional detector/tracker regression compared PyTorch/unfused OSNet against ONNX/fused OSNet over all decodable frames of three separate clips. Every per-frame ID, detection count, track lifetime distribution, and ephemeral-detection count matched:

| Clip | Analyzed frames | Tracks | Ephemeral detections | PyTorch FPS | ONNX FPS |
|---|---:|---:|---:|---:|---:|
| office cctv | 739 | 6 | 29 | 13.01 | 17.49 |
| crowd sample | 170 | 27 | 117 | 11.52 | 15.07 |
| crowded sample2 | 336 | 43 | 124 | 9.94 | 12.52 |

These last FPS values cover detection/tracking only, not the full surveillance loop. Maximum box-coordinate difference across these tests was 0.00037 pixels. Local decoder output for crowd sample was shorter than its container's advertised frame count in both passes.

- All three VLM-enabled runs reached EOF normally, with zero frame errors and no pending requests when the function returned. Nine successful responses took 1.87–5.01 seconds. Walking, standing, person interaction, and object interaction appeared; the six-second global call floor still leaves coverage gaps and can miss brief actions. Faster offline processing changes which video windows are sampled. This is not an accuracy percentage, and running/sitting remain outside the selected sample validation.
- 22 unit tests passed, including disabled mode with no client/worker/API calls, old-run callback rejection, bounded shutdown, inactive cleanup, OSNet embedding equivalence, cached depth/theft equivalence, exact grab/retrieve frames, heatmap retention/rendering, and model-export fallback. Expected injected API failures in unit tests are not production test failures.
- A final same-session smoke check ran sample12 twice with VLM disabled: each completed all 209 analyzed frames with zero errors, no executor, fresh tracker state, and unchanged saved zone counters. This also exercised the final buffer-release and per-run zone isolation changes.

## Bottlenecks and real-time implications

Before optimization, YOLO plus BotSort/ReID accounted for approximately 66–79% of steady pipeline time. Depth and occasional face identification were secondary costs; decoding was only around 1–2%, preview updates around 3–4%, and asynchronous VLM bookkeeping under 0.08 seconds per clip. Network latency affects label freshness rather than blocking the frame loop. This justified optimizing the existing detector backend and tracker rather than removing ReID, increasing frame skips, or reducing image resolution.

With `FRAME_SKIP=2`, the optimized full pipeline handles roughly **24–34 source FPS** in steady state. It can approach a single 30-FPS feed in the lighter case, but does not consistently sustain it in all tested scenes. A 60-FPS feed needs 30 analyzed FPS, still above this CPU result. The RTX 3060 contributes no benefit while intentionally unused. RAM capacity does not remove CPU inference or network bottlenecks.

Before live deployment, add a bounded latest-frame RTSP capture queue, capture timestamps/age metrics, reconnect/read timeouts, and an independently responsive stop path. Recalibrate source-frame thresholds against timestamps when frames are dropped; otherwise identity and intrusion timing will change. The current UI remains an uploaded-file workflow, and browser Stop responsiveness has not been validated. Do not treat file benchmarks as a live end-to-end latency guarantee. Further CPU work should profile OSNet and face inference under multi-person load; preserve per-person association tests when changing backends.

## YOLO27 decision

YOLO27 is an **unreleased preview** with no public weights/code to benchmark. Its advertised GPU detection results use an RTX PRO 6000, so they do not establish CPU pose speed, memory usage, or RTX 3060 performance. Pose support is planned, but this pipeline requires actual keypoints. Retain the validated YOLOv8n-pose export now; compare released nano pose weights on these identity/zone regressions before migrating. [Ultralytics YOLO27 preview](https://docs.ultralytics.com/models/yolo27)

## Reproduction and retained files

Run from the project root using Python 3.11:

```powershell
.\venv311\Scripts\python.exe -m pip install -r requirements.txt
.\venv311\Scripts\python.exe scripts/export_cpu_pose.py
.\venv311\Scripts\python.exe -m unittest discover -s tests -v
.\venv311\Scripts\python.exe -m streamlit run streamlit_app.py
```

Set `USE_VLM` in `config.py`; restart Streamlit after configuration edits. `YOLO_CPU_BACKEND="pytorch"` restores the original detector backend. The deployment model and checksum manifest in `data/models/` are necessary generated assets, ignored by Git and reproducible with the export script. Permanent regression tests and this report are retained. Temporary benchmark scripts, JSON, logs, images, and their task-created folders were removed after validation; the user's existing event log edits were preserved.
