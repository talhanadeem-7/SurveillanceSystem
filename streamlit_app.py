import streamlit as st
import cv2
import os
import logging
import numpy as np
import time
import datetime
import copy
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import config

from vision.detector import PoseDetector
from vision.pose_analyzer import PoseAnalyzer
from vision.face_recognition import FaceIdentifier
from vision.vlm_activity_analyzer import VLMActivityAnalyzer

from storage.event_logger import EventLogger
from utils.video_utils import draw_surveillance_ui  # use the drawing logic
from utils.csv_utils import LogAnalyzer

from utils.heatmap import (
    record_position,
    generate_heatmap,
    generate_trajectory,
    generate_combined,
    draw_heatmap_legend,
    get_display_name,
)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        force=True,
    )

def handle_video_upload():
    # creates a button on the sidebar: "Upload Footage".
    st.sidebar.header("1. Upload Footage")
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Create directory if it doesn't exist
        if not os.path.exists(config.VIDEO_DIR):
            os.makedirs(config.VIDEO_DIR)

        # Save the file to the path config.py expects
        video_path = os.path.join(config.VIDEO_DIR, uploaded_file.name)
        
        # Only write if it hasn't been saved yet
        if st.session_state.video_path != video_path:
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.video_path = video_path
            st.session_state.zones = []
            st.session_state.processing_complete = False
            st.session_state.track_positions = {}
            st.session_state.identity_map = {}
            
            # Extract the first frame for Zone Setup
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                # Convert to RGB for Streamlit/PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.session_state.first_frame = frame_rgb
                st.session_state.frame_shape = frame.shape # (H, W, C)
            cap.release()
            
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    
    return st.session_state.video_path


# --- WEB-BASED ZONE DRAWING WITH MANUAL NAMING ---

def define_zones_ui():
    st.header("2. Define Surveillance Zones")
    
    if 'first_frame' not in st.session_state:
        st.warning("Please upload a video first.")
        return

    # 1. UI Selection for Drawing
    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        zone_type = st.radio("Drawing Mode", ["Restricted (Rectangle)", "Passive (Polygon)"], horizontal=False)
    with col_info:
        st.info("1. Draw your zones on the image. \n2. Scroll down to name them. \n3. Click Save.")

    drawing_mode = "rect" if zone_type == "Restricted (Rectangle)" else "polygon"
    stroke_color = "#FF0000" if drawing_mode == "rect" else "#FFFF00"

    # Scaling Logic
    orig_h, orig_w = st.session_state.frame_shape[:2]
    display_width = 800
    scale_factor = orig_w / display_width
    display_height = int(orig_h / scale_factor)

    # 2. The Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color=stroke_color,
        background_image=Image.fromarray(st.session_state.first_frame),
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # 3. Dynamic Naming Logic
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        
        if len(objects) > 0:
            st.subheader("🏷️ Name your detected zones")
            
            # We use a form to collect names and shapes all at once
            with st.form("zone_naming_form"):
                zone_names = []
                for i, obj in enumerate(objects):
                    z_type = "Restricted" if obj["type"] == "rect" else "Passive"
                    # Generate a text input for every shape on the canvas
                    name = st.text_input(f"Name for {z_type} Shape #{i+1}", value=f"{z_type}_{i+1}")
                    zone_names.append(name)
                
                submit = st.form_submit_button("Confirm and Save Zones")
                
                if submit:
                    processed_zones = []
                    for i, obj in enumerate(objects):
                        custom_name = zone_names[i]
                        
                        # --- CASE A: RESTRICTED (RECTANGLES) ---
                        if obj["type"] == "rect":
                            x1 = int(obj["left"] * scale_factor)
                            y1 = int(obj["top"] * scale_factor)
                            x2 = int((obj["left"] + obj["width"]) * scale_factor)
                            y2 = int((obj["top"] + obj["height"]) * scale_factor)
                            
                            # Match app.py logic for patch extraction
                            # (analyzer expects BGR/Grayscale for theft detection)
                            patch = cv2.cvtColor(st.session_state.first_frame[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
                            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                            
                            processed_zones.append({
                                "name": custom_name,
                                "type": "restricted",
                                "status": "SECURE",
                                "color": config.COLOR_SECURE,
                                "coords": (x1, y1, x2, y2),
                                "orig_coords": (x1, y1, x2, y2),
                                "reference_patch": cv2.GaussianBlur(patch_gray, (5,5), 0),
                                "missing_counter": 0,
                                "last_interactor": None
                            })
                        
                        # --- CASE B: PASSIVE (POLYGONS) ---
                        elif obj["type"] == "path":
                            path = obj["path"]
                            poly_pts = []
                            for segment in path:
                                if segment[0] in ['M', 'L']: 
                                    poly_pts.append([segment[1] * scale_factor, segment[2] * scale_factor])
                            
                            processed_zones.append({
                                "name": custom_name,
                                "type": "passive",
                                "status": "MONITORING",
                                "color": config.COLOR_PASSIVE,
                                "polygon": np.array(poly_pts, np.int32),
                                "coords": (0,0,0,0), # Consistency placeholder
                                "last_interactor": None
                            })

                    st.session_state.zones = processed_zones
                    st.success(f"✅ Saved {len(processed_zones)} zones with custom names!")

def run_surveillance():
    #  It processes the video frame by frame.
    st.header("3. Live Surveillance")
    run_started = time.perf_counter()

    if not st.session_state.zones:
        st.warning("Please define zones in 'Zone Setup' first.")
        return

    # It creates the layout for the video player and the "Stop" button. It also calls setup_surveillance_memory() to create the empty dictionaries needed to track people (like remembering who is "Person 1" and where they were 5 seconds ago).
    col1, col2 = st.columns([3, 1])
    with col1:
        frame_placeholder = st.empty()  #  "Web Screen"
        stop_btn = st.button("Stop Surveillance")
    with col2:
        st.subheader("Live Status")
        status_text = st.empty()

    activity_analyzer = ensure_activity_analyzer()
    st.caption(f"VLM: {'enabled' if activity_analyzer.enabled else 'disabled'} | CPU vision")
    st.session_state.processing_complete = False
    st.session_state.track_positions = {}
    st.session_state.final_activities = {}
    if st.session_state.get("surveillance_runs", 0):
        from vision.tracker import StrongSortTracker
        st.session_state.detector.tracker = StrongSortTracker()
    st.session_state.surveillance_runs = st.session_state.get("surveillance_runs", 0) + 1

    # --- INJECTING MEMORY SETUP ---
    (
        analyzer,
        identity_map,
        unknown_check_counters,
        face_candidate_state,
        intrusion_persistence,
        person_location_state,
        activity_cache,
        PERSISTENCE_THRESHOLD,
    ) = setup_surveillance_memory()
    identity_revalidation_state = {}
    last_seen = {}
    st.session_state.identity_map = identity_map

    # Reset IdentityGuardian so stale embeddings from a previous run
    # don't cause wrong matches in the new session.
    from vision.identity_guardian import IdentityGuardian
    snapshot_root = st.session_state.detector.guardian.snapshot_root
    st.session_state.detector.guardian = IdentityGuardian(
        snapshot_root=str(snapshot_root)
    )

    # Setup Video capture
    cap = cv2.VideoCapture(st.session_state.video_path)

    # --- PERFORMANCE: read native FPS and set buffer size ---
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # small buffer prevents stale frames

    # --- PERFORMANCE: process every Nth frame (skip frames) ---
    # At 30 fps native, FRAME_SKIP=2 means we run inference at ~15 fps,
    # which is plenty for surveillance while halving GPU load.
    #
    # FRAME_SKIP=3 was MEASURED AND REJECTED — do not re-propose it without new
    # evidence. It reaches live-camera real time (>=30 source fps) on only one of
    # three clips (office cctv 31.93, crowd sample 27.86, crowded sample2 25.23),
    # and it buys that by dropping the tracker's update rate from 15 Hz to 10 Hz:
    # the VF-13 regression split moves from frames 429/450 to 286/342 and median
    # track life falls on all three clips (147->100, 23->15, 40->31). For a system
    # whose purpose is attributing events to specific people, that is the wrong
    # trade. See CLAUDE.md VF-38/VF-39.
    #
    # grab() still advances the codec on every source frame. retrieve() runs only
    # for analyzed frames, avoiding unnecessary color conversion/copies.
    FRAME_SKIP = max(1, int(getattr(config, "FRAME_SKIP", 2)))

    # --- PERFORMANCE: downscale inference resolution ---
    # We detect at a smaller size and draw on the original frame.
    # Preserve the existing resize and model input size for accuracy parity.
    INFERENCE_WIDTH = int(getattr(config, "INFERENCE_WIDTH", 640))

    # Per-run counters and alignment must not mutate the saved zone definitions.
    zones = copy.deepcopy(st.session_state.zones)
    frame_id = 0
    last_depth_map = None
    DEPTH_REFRESH_INTERVAL = max(1, int(getattr(config, "DEPTH_REFRESH_INTERVAL", 5)))

    # Timing: keep display smooth at a capped rate
    target_display_interval = 1.0 / max(1.0, min(native_fps / FRAME_SKIP, config.PREVIEW_FPS))
    last_display_time = 0.0

    # --- ROBUSTNESS: per-frame error handling -------------------------
    # One bad frame must not kill an entire run. On file playback that costs
    # a run; against a live camera it means surveillance stops overnight on a
    # single corrupt frame and nobody notices until morning.
    MAX_CONSECUTIVE_FAILURES = 30   # 30 in a row is systemic, not a bad frame
    consecutive_failures = 0
    total_failures = 0
    aborted_reason = None
    processed_frames = 0
    reached_eof = False
    unfinished_vlm = 0
    loop_started = time.perf_counter()

    try:
        while cap.isOpened() and not stop_btn:
            try:
                # Advance the decoder on every source frame; convert/retrieve only
                # the frames selected for analysis. No extra analysis frames lost.
                if not cap.grab():
                    reached_eof = True
                    break
                frame_id += 1

                # --- FRAME SKIP: forward-read to maintain real-time pace ---
                if frame_id % FRAME_SKIP != 0:
                    continue
                ret, frame = cap.retrieve()
                if not ret:
                    raise RuntimeError("Video frame retrieval failed")

                # --- DOWNSCALE for inference, keep original for drawing ---
                orig_h, orig_w = frame.shape[:2]
                scale = INFERENCE_WIDTH / orig_w
                if scale < 1.0:                         # only shrink, never upscale
                    infer_frame = cv2.resize(
                        frame, (INFERENCE_WIDTH, int(orig_h * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    infer_frame = frame                 # already small enough

                # A. Align Zones (Homography)
                zones = analyzer.align_zones(frame, zones)
                restricted_zones = [z for z in zones if z["type"] == "restricted"]
                passive_zones = [z for z in zones if z["type"] == "passive"]

                # B. Run YOLO on the (possibly downscaled) inference frame.
                results = st.session_state.detector.track_and_detect(infer_frame)

                # Scale detections back to original resolution so downstream zone
                # logic, face crops, and skeleton drawing stay in one coordinate space.
                # We clone() first because YOLO returns inference-mode tensors which
                # do not allow in-place operations (PyTorch InferenceMode restriction).
                if results.boxes is not None and scale < 1.0:
                    # xyxy is a read-only property — scale the underlying .data tensor.
                    # Boxes.data format: [x1, y1, x2, y2, (track_id,) conf, cls]
                    boxes_data = results.boxes.data.clone()
                    boxes_data[:, [0, 2]] /= scale   # x-coords
                    boxes_data[:, [1, 3]] /= scale   # y-coords
                    results.boxes.data = boxes_data

                    if results.keypoints is not None and results.keypoints.data is not None:
                        kpts_data = results.keypoints.data.clone()
                        kpts_data[:, :, 0] /= scale  # x-coords
                        kpts_data[:, :, 1] /= scale  # y-coords
                        results.keypoints.data = kpts_data

                # --- CONDITIONAL DEPTH ---
                #  It checks if any person is visually overlapping a zone box in 2D.
                # This block ensures we only turn on the "Depth Brain" (MiDaS) if someone is actually close to a zone. If everyone is far away, we skip this to save speed.
                boxes_xyxy = (results.boxes.xyxy.cpu().numpy()
                              if results.boxes is not None else np.empty((0, 4)))
                depth_map = None
                needs_depth = False
                if results.boxes is not None:
                    for i in range(len(results.boxes)):
                        bbox = boxes_xyxy[i]
                        if analyzer.check_2d_overlap(bbox, zones):
                            needs_depth = True
                            break

                if needs_depth:
                    should_refresh_depth = (
                        last_depth_map is None
                        or frame_id % DEPTH_REFRESH_INTERVAL == 0
                    )
                    if should_refresh_depth:
                        last_depth_map = analyzer.get_depth_map(frame)
                    depth_map = last_depth_map

                # Reset labels
                for z in zones:
                    if z["type"] == "restricted":
                        z["status"], z["color"] = "SECURE", config.COLOR_SECURE

                person_statuses = []
                anybody_overlapping_2d = False

                if results.boxes is not None and results.boxes.id is not None:
                    ids = results.boxes.id.cpu().numpy().astype(int)


                    for i in range(len(results.boxes)):
                        current_id = ids[i]
                        bbox = boxes_xyxy[i]
                        kpts = results[i].keypoints

                        # Detections the tracker could not match carry a unique
                        # negative id from PoseDetector. They are EPHEMERAL: the same
                        # person may get a different id next frame, so they must never
                        # key any cross-frame state. They are still drawn and still
                        # evaluated against zones for THIS frame; they simply cannot
                        # accumulate persistence, identity or trajectory history.
                        is_ephemeral = current_id < 0

                        # --- HEATMAP: record foot position for this track ---
                        if not is_ephemeral:
                            last_seen[current_id] = frame_id
                            record_position(
                                st.session_state.track_positions,
                                current_id,
                                bbox,
                                frame_id,
                            )

                        # --- VLM ACTIVITY COLLECTION ---
                        # Collect frames for VLM activity analysis (only for persistent tracks).
                        # Activity analysis is triggered at an interval and does not block the main loop.
                        if not is_ephemeral and st.session_state.activity_analyzer.enabled:
                            # Pass bbox so the analyzer buffers a CROP of this
                            # person, not the whole scene. The prompt names a
                            # track id, and a full frame gives the model no way
                            # to tell which person that id is.
                            st.session_state.activity_analyzer.add_frame(
                                current_id, frame, frame_id, bbox
                            )
                            # Trigger analysis if interval is reached
                            st.session_state.activity_analyzer.analyze_if_ready(
                                current_id, frame_id
                            )
                            # Always read through get_activity so the TTL applies:
                            # analyze_if_ready returns the cached result while a
                            # call is in flight, which would bypass expiry.
                            completed_activity = st.session_state.activity_analyzer.get_activity(
                                current_id, frame_id
                            )
                            if completed_activity:
                                activity_cache[current_id] = completed_activity
                            else:
                                activity_cache.pop(current_id, None)

                        # --- A. DETERMINE CURRENT LOCATION (Passive Context) ---
                        #  It checks where the person is standing.
                        current_loc = "General Area" 

                        if analyzer.check_trespassing(
                            kpts, bbox, passive_zones, depth_map, person_id=current_id
                        ):
                            for z in passive_zones:
                                fx = (kpts.xy[0][15][0] + kpts.xy[0][16][0]) / 2
                                fy = (kpts.xy[0][15][1] + kpts.xy[0][16][1]) / 2
                                foot_x, foot_y = (int(fx), int(fy))
                                if cv2.pointPolygonTest(
                                    z["polygon"], (foot_x, foot_y), False
                                ) >= 0:
                                    current_loc = z["name"]
                                    break

                        # --- B. 1. FACE RECOGNITION ---
                        #  Every 30 frames, it checks the person's face against the database. If recognized, it updates their name from "Person_1" to "Talha."
                        display_name = f"Person_{current_id}"
                        is_authorized = False

                        # Face recognition keys ALL of its state by track id, so it is
                        # skipped for ephemeral detections. They keep the default
                        # Person_<id> label and are never treated as authorized.
                        if not is_ephemeral:
                            if current_id not in unknown_check_counters:
                                unknown_check_counters[current_id] = 0
                            if current_id not in face_candidate_state:
                                face_candidate_state[current_id] = {"name": None, "count": 0}
                            if current_id not in identity_revalidation_state:
                                identity_revalidation_state[current_id] = {"fails": 0}

                        if not is_ephemeral and current_id in identity_map:
                            mapped_name = identity_map[current_id]
                            display_name = mapped_name
                            is_authorized = True

                            if (
                                unknown_check_counters[current_id]
                                % config.FACE_REVALIDATE_INTERVAL
                                == 0
                            ):
                                face_crop = (
                                    st.session_state.face_recognizer.extract_face_crop(
                                        frame, kpts
                                    )
                                )
                                recheck_name = None
                                if face_crop is not None:
                                    recheck_name = (
                                        st.session_state.face_recognizer.identify_person(
                                            face_crop
                                        )
                                    )

                                if recheck_name == mapped_name:
                                    # Confirmed: still the same person.
                                    identity_revalidation_state[current_id]["fails"] = 0
                                elif recheck_name is None:
                                    # No face visible this check (turned away, out of
                                    # frame, poor angle) or no confident match - this is
                                    # inconclusive, NOT a contradiction. A locked identity
                                    # should not erode just because the face was briefly
                                    # unobservable, so leave the fail counter untouched.
                                    pass
                                else:
                                    # A DIFFERENT enrolled identity was confidently
                                    # matched on this track - a genuine mismatch signal
                                    # (e.g. an ID/track swap), so this still counts.
                                    identity_revalidation_state[current_id]["fails"] += 1
                                    if (
                                        identity_revalidation_state[current_id]["fails"]
                                        >= config.FACE_REVALIDATION_FAILS
                                    ):
                                        identity_map.pop(current_id, None)
                                        face_candidate_state[current_id] = {
                                            "name": None,
                                            "count": 0,
                                        }
                                        identity_revalidation_state[current_id]["fails"] = 0
                                        display_name = f"Person_{current_id}"
                                        is_authorized = False
                                        st.session_state.logger._write_to_csv_and_terminal(
                                            f"Person_{current_id}",
                                            "Identity",
                                            f"Revoked {mapped_name} (mismatch)",
                                            current_loc,
                                        )
                        elif not is_ephemeral:
                            if (
                                unknown_check_counters[current_id]
                                % config.FACE_CHECK_INTERVAL
                                == 0
                            ):
                                face_crop = (
                                    st.session_state.face_recognizer.extract_face_crop(
                                        frame, kpts
                                    )
                                )
                                if face_crop is not None:
                                    found_name = (
                                        st.session_state.face_recognizer.identify_person(
                                            face_crop
                                        )
                                    )
                                    candidate = face_candidate_state[current_id]

                                    if found_name:
                                        if candidate["name"] == found_name:
                                            candidate["count"] += 1
                                        else:
                                            candidate["name"] = found_name
                                            candidate["count"] = 1

                                        if candidate["count"] >= config.FACE_CONFIRMATION_REQUIRED:
                                            identity_map[current_id] = found_name
                                            display_name = found_name
                                            is_authorized = True

                                            st.session_state.logger._write_to_csv_and_terminal(
                                                f"Person_{current_id}",
                                                "Identity",
                                                f"Recognized as {found_name}",
                                                current_loc,
                                            )
                                    else:
                                        candidate["name"] = None
                                        candidate["count"] = 0

                        if not is_ephemeral:
                            unknown_check_counters[current_id] += 1
                            person_location_state[current_id] = current_loc

                        # --- E. GLOBAL PERSISTENCE ---
                        raw_is_tres = analyzer.check_trespassing(
                            kpts,
                            bbox,
                            restricted_zones,
                            depth_map,
                            person_id=current_id,
                        )

                        if is_ephemeral:
                            # Persistence needs a stable id across frames. An ephemeral
                            # detection is evaluated for this frame only and can never
                            # reach PERSISTENCE_THRESHOLD, so it is never logged.
                            confirmed_tres = False
                        else:
                            if current_id not in intrusion_persistence:
                                intrusion_persistence[current_id] = 0

                            if raw_is_tres:
                                intrusion_persistence[current_id] += 1
                            else:
                                intrusion_persistence[current_id] = 0

                            confirmed_tres = (
                                intrusion_persistence[current_id] >= PERSISTENCE_THRESHOLD
                            )
                        person_statuses.append(confirmed_tres)

                        is_overlap = analyzer.check_2d_overlap(
                            bbox, restricted_zones
                        )
                        if is_overlap:
                            anybody_overlapping_2d = True

                        # --- F. ZONE-SPECIFIC LOGIC ---
                        # If an unauthorized person touches a Restricted Zone for a few frames, it triggers an "Intrusion" alert.
                        for z in zones:
                            if z["type"] == "restricted":
                                this_zone_tres = analyzer.check_trespassing(
                                    kpts, bbox, [z], depth_map, person_id=current_id
                                )

                                if this_zone_tres and confirmed_tres:
                                    z["last_interactor"] = display_name
                                    person_location_state[current_id] = z["name"]

                                    if is_authorized:
                                        z["status"] = "AUTHORIZED ACCESS"
                                        z["color"] = config.COLOR_AUTHORIZED
                                        st.session_state.logger.log_event(
                                            display_name,
                                            "Access",
                                            "AUTHORIZED",
                                            True,
                                            location=z["name"],
                                        )
                                    else:
                                        z["status"] = "UNAUTHORIZED ACCESS"
                                        z["color"] = config.COLOR_UNAUTHORIZED
                                        st.session_state.logger.log_event(
                                            display_name,
                                            "Intrusion",
                                            "UNAUTHORIZED",
                                            True,
                                            location=z["name"],
                                        )

                    results.is_trespassing = person_statuses

                # --- G. Theft Detection ---
                # It checks if an object has disappeared.
                # runs only when people move away (no overlap). It compares the "Edge Density" of the zone now vs. the beginning. If the edges are gone (laptop missing), it triggers a Theft alert. It checks who the last person was to decide if it was "Stolen" (Stranger) or "Removed" (Authorized Person).
                if not anybody_overlapping_2d:
                    for z in zones:
                        if z["type"] == "restricted":
                            is_theft = analyzer.detect_theft(frame, z)
                            zone_name = z["name"]

                            if is_theft:
                                last_user = z["last_interactor"]
                                last_user_is_authorized = (
                                    last_user in identity_map.values()
                                    if last_user
                                    else False
                                )

                                if last_user_is_authorized:
                                    z["status"] = f"REMOVED BY {last_user}"
                                    z["color"] = config.COLOR_AUTHORIZED
                                    st.session_state.logger.log_event(
                                        "ASSET",
                                        "Removal",
                                        f"BY_{last_user}",
                                        True,
                                        location=zone_name,
                                    )
                                else:
                                    z["status"] = "THEFT DETECTED"
                                    z["color"] = config.COLOR_THEFT
                                    st.session_state.logger.log_event(
                                        "ASSET",
                                        "Theft",
                                        "STOLEN",
                                        True,
                                        location=zone_name,
                                    )
                            else:
                                st.session_state.logger.log_event(
                                    "ASSET",
                                    "Theft",
                                    "STOLEN",
                                    False,
                                    location=zone_name,
                                )
                                st.session_state.logger.log_event(
                                    "ASSET",
                                    "Removal",
                                    "BY_AUTH",
                                    False,
                                    location=zone_name,
                                )

                # --- H. RENDERING AND UI UPDATES ---
                for tid in list(activity_cache):
                    if activity_analyzer.get_activity(tid, frame_id) is None:
                        activity_cache.pop(tid, None)
                # --- PERFORMANCE: throttle display — only push frame to browser
                # at the target rate.  Heavy inference can run every loop tick,
                # but st.image() is expensive; skipping a render never drops a frame
                # from the analysis, just from the preview.
                now = time.perf_counter()
                if now - last_display_time >= target_display_interval:
                    draw_surveillance_ui(frame, results, zones, identity_map, activity_cache)
                    frame_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_column_width=True,
                    )
                    # Show IdentityGuardian live stats in the status panel
                    guardian_stats = st.session_state.detector.guardian.get_stats()
                    
                    # Build activity status display
                    activity_status = "No activities tracked"
                    if activity_cache:
                        activities = []
                        for track_id, result in list(activity_cache.items())[-3:]:  # Show last 3
                            activities.append(
                                f"Track {track_id}: **{result.activity}** "
                                f"(conf: {result.confidence:.2f})"
                            )
                        activity_status = "\n\n".join(activities)
                    
                    status_text.markdown(
                        f"**Frame:** {frame_id}\n\n"
                        f"**Analyzed FPS:** {(processed_frames + 1) / max(now - loop_started, 1e-6):.1f}\n\n"
                        f"**Source FPS:** {frame_id / max(now - loop_started, 1e-6):.1f} / {native_fps:.1f}\n\n"
                        f"**Known Identities:** {guardian_stats['known_identities']}\n\n"
                        f"**ID Corrections:** {guardian_stats['active_remaps']}\n\n"
                        f"**Retired IDs:** {guardian_stats['retired_ids']}\n\n"
                        f"**Recent Activities:**\n\n{activity_status}"
                    )
                    last_display_time = now

                st.session_state.logger.update_logs()

                # Frame completed without raising -> reset the failure streak.
                consecutive_failures = 0
                processed_frames += 1
                if processed_frames % 30 == 0:
                    activity_analyzer.cleanup_inactive(frame_id)
                    ttl = max(config.INACTIVE_TRACK_TTL, config.TRACKER_TRACK_BUFFER * FRAME_SKIP)
                    expired = [tid for tid, seen in last_seen.items() if frame_id - seen > ttl]
                    for tid in expired:
                        for state in (last_seen, identity_map, unknown_check_counters,
                                      face_candidate_state, identity_revalidation_state,
                                      intrusion_persistence, person_location_state, activity_cache):
                            state.pop(tid, None)
                    analyzer.cleanup_tracks(expired)
                    # Ephemeral detections never accumulate meaningful filter state.
                    for key in list(analyzer.person_depth_filters):
                        if key.startswith("-"):
                            del analyzer.person_depth_filters[key]

                if stop_btn:
                    break
            except (KeyboardInterrupt, SystemExit):
                # Never swallow these — let the interpreter unwind. The finally
                # block below still releases the capture handle.
                raise
            except Exception as exc:
                total_failures += 1
                consecutive_failures += 1
                logging.exception(
                    "run_surveillance: frame %d failed with %s "
                    "(consecutive=%d, total=%d)",
                    frame_id, type(exc).__name__,
                    consecutive_failures, total_failures,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    aborted_reason = (
                        f"Aborted after {consecutive_failures} consecutive "
                        f"frame failures at frame {frame_id}. Last error: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    logging.error("run_surveillance: %s", aborted_reason)
                    break
                continue
    finally:
        # Always runs: normal end, abort path, and user interruption.
        cap.release()
        loop_seconds = time.perf_counter() - loop_started
        unfinished_vlm = activity_analyzer.shutdown(
            timeout=config.VLM_SHUTDOWN_TIMEOUT if reached_eof else 0.0
        )
        st.session_state.processing_complete = reached_eof and not aborted_reason
        st.session_state.surveillance_metrics = {
            "source_frames": frame_id, "processed_frames": processed_frames,
            "processing_seconds": loop_seconds,
            "elapsed_seconds": time.perf_counter() - run_started,
            "processed_fps": processed_frames / max(loop_seconds, 1e-6),
            "source_fps": frame_id / max(loop_seconds, 1e-6),
            "frame_errors": total_failures, "unfinished_vlm": unfinished_vlm,
        }
        st.session_state.final_activities = {
            tid: activity_analyzer.get_activity(tid, frame_id)
            for tid in list(activity_analyzer.activity_cache)
            if activity_analyzer.get_activity(tid, frame_id) is not None
        }

    metrics = st.session_state.surveillance_metrics
    st.info(f"Processed {processed_frames} frames in {metrics['processing_seconds']:.2f}s "
            f"({metrics['processed_fps']:.1f} analyzed FPS; {metrics['source_fps']:.1f} source FPS).")
    if unfinished_vlm:
        st.warning(f"{unfinished_vlm} VLM request(s) exceeded the shutdown grace period; late results were discarded.")
    if st.session_state.final_activities:
        st.write("Latest VLM observations (may precede the final frame):")
        for tid, activity in st.session_state.final_activities.items():
            st.text(f"Track {tid}: {activity.activity} ({activity.confidence:.2f})")

    # Zone alignment health: a camera that has drifted out of ORB match range
    # fails silently otherwise — the zones simply stop tracking the scene.
    if getattr(analyzer, "align_failures_total", 0):
        st.warning(
            f"⚠️ Zone alignment failed on {analyzer.align_failures_total} of "
            f"{analyzer.align_recomputes} recompute attempts "
            f"(longest run of consecutive failures ended at "
            f"{analyzer.align_consecutive_failures}). Zones reused the last good "
            f"homography. Check the log for reasons."
        )

    if aborted_reason:
        st.error(f"🛑 {aborted_reason}")
    elif not reached_eof:
        st.info("Surveillance stopped.")
    elif total_failures:
        st.warning(
            f"⚠️ Surveillance finished, but {total_failures} frame(s) were skipped "
            f"due to errors. See the log for tracebacks."
        )
        st.success(
            "✅ Surveillance Processing Finished. You can now use the 'Shelby Analyst' tab."
        )
    else:
        st.success(
            "✅ Surveillance Processing Finished. You can now use the 'Shelby Analyst' tab."
        )


def run_shelby_analyst():
    # The Chatbot Interface.
    st.subheader("🕵️ Chat with Shelby")
    
    # 1. Initialize your existing SecurityAnalyst class
    if st.session_state.analyst is None:
        with st.spinner("Shelby is analyzing the logs..."):
            from agents.reasoning_agent import SecurityAnalyst
            st.session_state.analyst = SecurityAnalyst()
        st.success("Shelby is ready!")

    # 2. Handle User Input (Memory-less Version)
    if prompt := st.chat_input("Ask Shelby about the footage..."):
        # Just show the current question
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Call your agent's logic
        with st.chat_message("assistant", avatar="🛡️"):
            try:
                response = st.session_state.analyst.consult(prompt)
                st.markdown(response)
            except Exception as e:
                # OpenAI specific error handling
                if "429" in str(e):
                    st.error("🛑 OpenAI Rate Limit: You are sending requests too fast or your trial credits ran out.")
                elif "insufficient_quota" in str(e):
                    st.error("🛑 OpenAI Error: Your API key has no remaining credits.")
                else:
                    st.error(f"Error: {e}")

def run_heatmap_tab():
    """
    Heatmap & Trajectory tab — visualizes movement data collected during
    the surveillance run.  No surveillance logic is touched here.
    """
    st.header("🔥 Person Movement Heatmap & Trajectory")

    track_positions = st.session_state.get("track_positions", {})
    identity_map    = st.session_state.get("identity_map", {})
    first_frame_rgb = st.session_state.get("first_frame", None)

    if not track_positions:
        st.info("No movement data yet. Run the Surveillance Engine first.")
        return

    if first_frame_rgb is None:
        st.warning("Reference frame not available. Please upload a video first.")
        return

    # Reference frame in BGR for OpenCV drawing functions
    ref_bgr = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)

    # --- Person selector ---
    st.subheader("Select Person")
    track_ids = sorted(track_positions.keys())
    id_labels  = {tid: get_display_name(tid, identity_map) for tid in track_ids}

    # "All" option + individual persons
    all_label   = "👥 All Persons (combined)"
    select_opts = [all_label] + [f"{id_labels[tid]} (ID {tid})" for tid in track_ids]
    selected    = st.selectbox("Choose who to visualize:", select_opts)

    # --- View mode ---
    st.subheader("Visualization Mode")
    view_mode = st.radio(
        "Select view:",
        ["🌡️ Heatmap only", "🛤️ Trajectory only", "🔀 Combined (Heatmap + Trajectory)"],
        horizontal=True,
    )

    # --- Advanced settings (collapsed by default) ---
    with st.expander("⚙️ Advanced Settings"):
        sigma = st.slider(
            "Heatmap blur radius (sigma)",
            min_value=5, max_value=80, value=35, step=5,
            help="Higher = smoother/broader heat blobs. Lower = sharper/tighter.",
        )
        alpha = st.slider(
            "Heatmap opacity",
            min_value=0.1, max_value=0.9, value=0.55, step=0.05,
            help="How strongly the heatmap overlays the background frame.",
        )
        show_legend = st.checkbox("Show colour bar legend", value=True)

    # --- Generate button ---
    if st.button("🎨 Generate Visualization", type="primary"):

        # Gather positions for the selected person(s)
        if selected == all_label:
            # Merge all tracks into one flat list
            all_positions = []
            for tid in track_ids:
                all_positions.extend(track_positions[tid])
            positions_to_use = all_positions
            vis_label = "All Persons"
        else:
            # Parse track ID from the label string
            chosen_tid = int(selected.split("ID ")[-1].rstrip(")"))
            positions_to_use = track_positions[chosen_tid]
            vis_label = id_labels[chosen_tid]

        if not positions_to_use:
            st.warning("No position data available for the selected person.")
            return

        with st.spinner("Generating visualization..."):

            if "Heatmap" in view_mode:
                result_bgr = generate_heatmap(
                    positions_to_use, ref_bgr,
                    sigma=sigma, alpha=alpha,
                )
            elif "Trajectory" in view_mode:
                result_bgr = generate_trajectory(positions_to_use, ref_bgr)
            else:  # Combined
                result_bgr = generate_combined(
                    positions_to_use, ref_bgr,
                    sigma=sigma, alpha=alpha,
                )

            if show_legend and "Trajectory" not in view_mode:
                result_bgr = draw_heatmap_legend(result_bgr)

        # Display
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption=f"{vis_label} — {view_mode}", use_column_width=True)

        # Stats
        n_frames = len(set(p[2] for p in positions_to_use))
        n_points  = len(positions_to_use)
        c1, c2 = st.columns(2)
        c1.metric("Total positions recorded", n_points)
        c2.metric("Frames active", n_frames)

        # Download button
        _, img_encoded = cv2.imencode(".png", result_bgr)
        st.download_button(
            label="⬇️ Download as PNG",
            data=img_encoded.tobytes(),
            file_name=f"shelby_heatmap_{vis_label.replace(' ', '_')}.png",
            mime="image/png",
        )


def main():
    st.title("🛡️ Shelby: Intelligent Surveillance Dashboard")
    
    # Sidebar
    v_path = handle_video_upload()

    if v_path:
        # In Streamlit, we use tabs to navigate the app's workflow
        tab1, tab2, tab3, tab4 = st.tabs(["Zone Setup", "Surveillance Feed", "Shelby Analyst", "🔥 Heatmap & Trajectory"])
        
        with tab1:
            define_zones_ui()
            if st.session_state.zones:
                st.write("Current Zones Configured:")
                for z in st.session_state.zones:
                    st.text(f"📍 {z['name']} ({z['type']})")

        with tab2:
            st.header("Surveillance Execution")
            # This is the "Start Surveillance" button
            if st.button("🚀 Start Surveillance Engine"):
                run_surveillance()
        
        with tab3:
            st.header("Shelby Intelligence Analyst")
            if not st.session_state.get('processing_complete', False):
                st.info("📊 Shelby needs you to run the Surveillance Engine first so she has logs to analyze.")
                
                # Option to initialize anyway if logs already exist from a previous run
                if os.path.exists(config.LOG_PATH):
                    if st.button("Initialize Analyst with existing logs"):
                        st.session_state.processing_complete = True
                        st.rerun()
            else:
                # Tabs execute eagerly: require an explicit action before loading RAG.
                if st.session_state.get("analyst") is not None or st.button("Open Shelby Analyst"):
                    run_shelby_analyst()

        with tab4:
            if st.checkbox("Show heatmap and trajectory", value=False):
                run_heatmap_tab()




def initialize_surveillance_components():
    # It loads the heavy AI models (YOLO and Face Recognition) once when the app starts, so it doesn't have to reload them every time you click a button. This makes the app faster.
    # 1. Setup basic components (Mapped to Session State for performance)
    if 'detector' not in st.session_state:
        # PoseDetector already includes the StrongSORT-backed tracker used in live surveillance.
        st.session_state.detector = PoseDetector()
    
    if 'logger' not in st.session_state:
        st.session_state.logger = EventLogger()

    # Note: ZoneSelector is replaced by the web canvas, but the logic 
    # for processing zones will be handled in define_zones_ui()

    if 'face_recognizer' not in st.session_state:
        with st.spinner("Initializing Face Recognition System..."):
            st.session_state.face_recognizer = FaceIdentifier()

    ensure_activity_analyzer()

def ensure_activity_analyzer():
    enabled = bool(getattr(config, "USE_VLM", getattr(config, "VLM_ACTIVITY_ENABLED", True)))
    current = st.session_state.get("activity_analyzer")
    if current is None or current.enabled != enabled:
        if current is not None:
            current.shutdown(timeout=0.0)
        current = VLMActivityAnalyzer(enabled=enabled)
        st.session_state.activity_analyzer = current
    return current


def setup_surveillance_memory():

    if 'analyst' not in st.session_state:
        # We don't initialize yet to save memory, just set a placeholder
        st.session_state.analyst = None
    # 2. Initialize Analyzer (Done at the start of the processing loop)
    # Reference frame is grabbed from st.session_state.first_frame (Step 3)
    # We convert back to BGR to keep PoseAnalyzer logic identical
    ref_frame_bgr = cv2.cvtColor(st.session_state.first_frame, cv2.COLOR_RGB2BGR)
    analyzer = PoseAnalyzer(ref_frame_bgr)

    # Reset VLM activity analyzer for this run
    st.session_state.activity_analyzer.reset()

    # --- MEMORY STORES (Exactly as per your app.py) ---
    identity_map = {}
    unknown_check_counters = {}
    face_candidate_state = {}
    intrusion_persistence = {}
    person_location_state = {}  # Tracks where each person was last seen
    activity_cache = {}  # {track_id: ActivityResult} -- populated by VLMActivityAnalyzer
    PERSISTENCE_THRESHOLD = 5

    return (analyzer, identity_map, unknown_check_counters, face_candidate_state, intrusion_persistence,
            person_location_state, activity_cache, PERSISTENCE_THRESHOLD)




# Run the initialization
if __name__ == "__main__":
    # The Navigator.
    st.set_page_config(page_title="Intelligent Surveillance - Shelby", layout="wide")
    configure_logging()
    
    if 'initialized' not in st.session_state:
        initialize_surveillance_components()
        st.session_state.initialized = True
        st.session_state.video_path = None
        st.session_state.zones = []
        st.session_state.identity_map = {}
        st.session_state.processing_complete = False
        st.session_state.track_positions = {}   # {track_id: [(cx, cy, frame_id), ...]}
        # REMOVED: st.session_state.chat_history = [] 
    
    main()
