import streamlit as st
import cv2
import os
import numpy as np
import time
import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# 1. Import your existing project configuration
import config

# 2. Import your Vision Logic (The "Watchman")
from vision.detector import PoseDetector
from vision.pose_analyzer import PoseAnalyzer
from vision.face_recognition import FaceIdentifier
from vision.action_recognizer import ActionRecognizer

# 3. Import your Storage and Utility Logic
from storage.event_logger import EventLogger
from utils.video_utils import draw_surveillance_ui  # We use the drawing logic
from utils.csv_utils import LogAnalyzer

# 4. Import your Agentic AI (The "Analyst" - Shelby)
from agents.reasoning_agent import SecurityAnalyst

# --- STEP 3: FILE UPLOADER & PRE-PROCESSING ---

def handle_video_upload():
    st.sidebar.header("1. Upload Footage")
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # 1. Create directory if it doesn't exist
        if not os.path.exists(config.VIDEO_DIR):
            os.makedirs(config.VIDEO_DIR)

        # 2. Save the file to the path config.py expects
        video_path = os.path.join(config.VIDEO_DIR, uploaded_file.name)
        
        # Only write if it hasn't been saved yet
        if st.session_state.video_path != video_path:
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.video_path = video_path
            
            # 3. Extract the first frame for Zone Setup
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

# --- STEP 4: WEB-BASED ZONE DRAWING ---

# --- UPDATED STEP 4: WEB-BASED ZONE DRAWING WITH MANUAL NAMING ---

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
# --- ADD THIS FUNCTION BELOW define_zones_ui() ---

def run_surveillance():
    st.header("3. Live Surveillance")

    if not st.session_state.zones:
        st.warning("Please define zones in 'Zone Setup' first.")
        return

    # UI Layout: Create a spot for the video and a spot for live logs
    col1, col2 = st.columns([3, 1])
    with col1:
        frame_placeholder = st.empty()  # This is your "Web Screen"
        stop_btn = st.button("Stop Surveillance")
    with col2:
        st.subheader("Live Status")
        status_text = st.empty()

    # --- INJECTING YOUR MEMORY SETUP ---
    (
        analyzer,
        identity_map,
        unknown_check_counters,
        intrusion_persistence,
        person_location_state,
        behavior_logged_ids,
        logged_general_ids,
        track_ids_logged_general,
        person_action_state,
        person_action_persistence,
        ACTION_STABILITY_THRESHOLD,
        PERSISTENCE_THRESHOLD,
    ) = setup_surveillance_memory()

    # Setup Video capture
    cap = cv2.VideoCapture(st.session_state.video_path)
    zones = st.session_state.zones  # Get zones from the canvas setup

    # --- THE LOGIC YOU PROVIDED GOES HERE ---
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        # A. Align Zones (Homography)
        zones = analyzer.align_zones(frame, zones)

        # B. Detect People
        results = st.session_state.detector.track_and_detect(frame)

        # --- CONDITIONAL DEPTH ---
        depth_map = None
        needs_depth = False
        if results.boxes is not None:
            for i in range(len(results.boxes)):
                bbox = results.boxes.xyxy[i].cpu().numpy()
                if analyzer.check_2d_overlap(bbox, zones):
                    needs_depth = True
                    break

        if needs_depth:
            depth_map = analyzer.get_depth_map(frame)

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
                bbox = results.boxes.xyxy[i].cpu().numpy()
                kpts = results[i].keypoints

                # --- A. DETERMINE CURRENT LOCATION (Passive Context) ---
                current_loc = "General Area"
                passive_zones = [z for z in zones if z["type"] == "passive"]

                if analyzer.check_trespassing(
                    kpts, bbox, passive_zones, depth_map, person_id=current_id
                ):
                    for z in passive_zones:
                        fx = (kpts.xyn[0][15][0] + kpts.xyn[0][16][0]) / 2
                        fy = (kpts.xyn[0][15][1] + kpts.xyn[0][16][1]) / 2
                        foot_x, foot_y = (
                            int(fx * frame.shape[1]),
                            int(fy * frame.shape[0]),
                        )
                        if cv2.pointPolygonTest(
                            z["polygon"], (foot_x, foot_y), False
                        ) >= 0:
                            current_loc = z["name"]
                            break

                # --- B. 1. FACE RECOGNITION ---
                display_name = f"Person_{current_id}"
                is_authorized = False

                if current_id in identity_map:
                    display_name = identity_map[current_id]
                    is_authorized = True
                else:
                    if current_id not in unknown_check_counters:
                        unknown_check_counters[current_id] = 0

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
                            if found_name:
                                identity_map[current_id] = found_name
                                display_name = found_name
                                is_authorized = True

                                st.session_state.logger._write_to_csv_and_terminal(
                                    f"Person_{current_id}",
                                    "Identity",
                                    f"Recognized as {found_name}",
                                    current_loc,
                                )

                                behavior_logged_ids.add(found_name)
                                logged_general_ids.add(found_name)

                    unknown_check_counters[current_id] += 1

                # --- C. HANDLE LOCATION STATE CHANGE ---
                detected_action = (
                    st.session_state.action_recognizer.update(current_id, kpts)
                    or "Standing"
                )

                if current_id not in person_location_state:
                    person_location_state[current_id] = None
                if current_id not in person_action_state:
                    person_action_state[current_id] = "Standing"
                if current_id not in person_action_persistence:
                    person_action_persistence[current_id] = 0

                last_loc = person_location_state[current_id]
                last_act = person_action_state[current_id]
                should_log = False

                if current_loc != last_loc:
                    should_log = True
                    person_action_persistence[current_id] = 0
                elif detected_action == "Picking Up":
                    should_log = True
                    person_action_persistence[current_id] = 0
                elif detected_action in ["Punch", "Kick", "Fall"]:
                    should_log = True
                    person_action_persistence[current_id] = 0
                elif detected_action != last_act:
                    person_action_persistence[current_id] += 1
                    if (
                        person_action_persistence[current_id]
                        >= ACTION_STABILITY_THRESHOLD
                    ):
                        should_log = True
                        person_action_persistence[current_id] = 0
                else:
                    person_action_persistence[current_id] = 0

                if should_log and current_loc == "General Area":
                    generic_label = f"Person_{current_id}"
                    if (
                        display_name in logged_general_ids
                        or generic_label in logged_general_ids
                    ):
                        if detected_action in ["Walking", "Standing"]:
                            should_log = False

                # --- D. ACTION RECOGNITION LOGGING ---
                if detected_action:
                    is_priority = detected_action in [
                        "Punch",
                        "Kick",
                        "Fall",
                        "Picking Up",
                    ]

                    should_log_act = False
                    if is_priority:
                        should_log_act = True
                    elif should_log:
                        should_log_act = True

                    if should_log_act:
                        if (
                            current_id not in behavior_logged_ids
                            and display_name not in behavior_logged_ids
                        ):
                            st.session_state.logger.log_event(
                                display_name,
                                "Behavior",
                                detected_action,
                                True,
                                location=current_loc,
                            )

                            behavior_logged_ids.add(current_id)
                            behavior_logged_ids.add(display_name)
                            if current_loc == "General Area":
                                track_ids_logged_general.add(current_id)

                person_location_state[current_id] = current_loc
                person_action_state[current_id] = detected_action

                # --- E. GLOBAL PERSISTENCE ---
                raw_is_tres = analyzer.check_trespassing(
                    kpts,
                    bbox,
                    [z for z in zones if z["type"] == "restricted"],
                    depth_map,
                    person_id=current_id,
                )

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
                    bbox, [z for z in zones if z["type"] == "restricted"]
                )
                if is_overlap:
                    anybody_overlapping_2d = True

                # --- F. ZONE-SPECIFIC LOGIC ---
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
        draw_surveillance_ui(frame, results, zones, identity_map)

        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

        st.session_state.logger.update_logs()

        if stop_btn:
            break

    cap.release()
    st.session_state.processing_complete = True
    st.success(
        "✅ Surveillance Processing Finished. You can now use the 'Shelby Analyst' tab."
    )


def run_shelby_analyst():
    st.subheader("🕵️ Chat with Shelby")
    
    # 1. Initialize your existing SecurityAnalyst class
    if st.session_state.analyst is None:
        with st.spinner("Shelby is analyzing the logs..."):
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

# Update your main() to include this
def main():
    st.title("🛡️ Shelby: Intelligent Surveillance Dashboard")
    
    # Sidebar
    v_path = handle_video_upload()

    if v_path:
        # In Streamlit, we use tabs to navigate the app's workflow
        tab1, tab2, tab3 = st.tabs(["Zone Setup", "Surveillance Feed", "Shelby Analyst"])
        
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
                # Run the chat interface
                run_shelby_analyst()


# --- SESSION STATE INITIALIZATION ---


def initialize_surveillance_components():
    # 1. Setup basic components (Mapped to Session State for performance)
    if 'detector' not in st.session_state:
        st.session_state.detector = PoseDetector()
    
    if 'logger' not in st.session_state:
        st.session_state.logger = EventLogger()

    # Note: ZoneSelector is replaced by the web canvas, but the logic 
    # for processing zones will be handled in define_zones_ui()

    if 'face_recognizer' not in st.session_state:
        with st.spinner("Initializing Face Recognition System..."):
            st.session_state.face_recognizer = FaceIdentifier()

    if 'action_recognizer' not in st.session_state:
        with st.spinner("Initializing Action Brain..."):
            st.session_state.action_recognizer = ActionRecognizer(model_path=config.ACTION_MODEL_PATH)

def setup_surveillance_memory():

    if 'analyst' not in st.session_state:
        # We don't initialize yet to save memory, just set a placeholder
        st.session_state.analyst = None
    # 2. Initialize Analyzer (Done at the start of the processing loop)
    # Reference frame is grabbed from st.session_state.first_frame (Step 3)
    # We convert back to BGR to keep PoseAnalyzer logic identical
    ref_frame_bgr = cv2.cvtColor(st.session_state.first_frame, cv2.COLOR_RGB2BGR)
    analyzer = PoseAnalyzer(ref_frame_bgr)

    # --- MEMORY STORES (Exactly as per your app.py) ---
    identity_map = {}
    unknown_check_counters = {}
    intrusion_persistence = {}
    person_location_state = {}  # Tracks where each person was last seen
    behavior_logged_ids = set()  
    logged_general_ids = set()  
    track_ids_logged_general = set() # Tracks physical body IDs
    person_action_state = {}
    person_action_persistence = {}
    
    # Logic Settings
    ACTION_STABILITY_THRESHOLD = 10 # frames (~0.5 seconds)
    PERSISTENCE_THRESHOLD = 5
    
    return (analyzer, identity_map, unknown_check_counters, intrusion_persistence, 
            person_location_state, behavior_logged_ids, logged_general_ids, 
            track_ids_logged_general, person_action_state, person_action_persistence, 
            ACTION_STABILITY_THRESHOLD, PERSISTENCE_THRESHOLD)




# Run the initialization
if __name__ == "__main__":
    st.set_page_config(page_title="Intelligent Surveillance - Shelby", layout="wide")
    
    if 'initialized' not in st.session_state:
        initialize_surveillance_components()
        st.session_state.initialized = True
        st.session_state.video_path = None
        st.session_state.zones = []
        st.session_state.identity_map = {}
        st.session_state.processing_complete = False
        # REMOVED: st.session_state.chat_history = [] 
    
    main()