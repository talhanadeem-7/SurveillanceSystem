import cv2
import os
import config
from vision.detector import PoseDetector
from vision.pose_analyzer import PoseAnalyzer
from vision.face_recognition import FaceIdentifier
from vision.action_recognizer import ActionRecognizer
from utils.video_utils import ZoneSelector, draw_surveillance_ui
from storage.event_logger import EventLogger

def main():
    # 1. Setup basic components
    detector = PoseDetector()
    logger = EventLogger()
    selector = ZoneSelector()

    print("Initializing Face Recognition System...")
    face_recognizer = FaceIdentifier()

    print("Initializing Action Brain...")
    action_recognizer = ActionRecognizer(model_path=config.ACTION_MODEL_PATH)

    # 2. Load Video
    video_files = os.listdir(config.VIDEO_DIR)
    if not video_files:
        print("No videos found.")
        return
    video_path = os.path.join(config.VIDEO_DIR, video_files[0])
    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    # 3. User Selects Zones (New Menu System)
    zones = selector.get_zones(first_frame)

    for z in zones:
        z['last_interactor'] = None

    # 4. Initialize Analyzer
    analyzer = PoseAnalyzer(first_frame)

    # --- MEMORY STORES ---
    identity_map = {}
    unknown_check_counters = {}
    intrusion_persistence = {}
    person_location_state = {}  # Tracks where each person was last seen
    behavior_logged_ids = set()  # <--- ADD THIS LINE
    logged_general_ids = set()  # <--- ADD THIS LINE
    track_ids_logged_general = set() # <--- ADD THIS: Tracks physical body IDs
    person_action_state = {}
# Tracks how many frames a new action has been seen (to prevent flicker)
    person_action_persistence = {}
    ACTION_STABILITY_THRESHOLD = 15 # frames (~0.5 seconds)




    PERSISTENCE_THRESHOLD = 5

    print("Starting Surveillance... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # A. Align Zones (Homography)
        zones = analyzer.align_zones(frame, zones)

        # B. Detect People
        results = detector.track_and_detect(frame)

        # --- CONDITIONAL DEPTH ---
        depth_map = None
        needs_depth = False
        if results.boxes is not None:
            for i in range(len(results.boxes)):
                bbox = results.boxes.xyxy[i].cpu().numpy()
                # Run depth if near ANY zone (Restricted or Passive)
                if analyzer.check_2d_overlap(bbox, zones): # <--- Changed this to 'zones'
                    needs_depth = True
                    break

        if needs_depth:
            depth_map = analyzer.get_depth_map(frame)

        # Reset labels for restricted zones
        for z in zones:
            if z['type'] == 'restricted':
                z['status'], z['color'] = "SECURE", config.COLOR_SECURE

        person_statuses = []
        anybody_overlapping_2d = False

        if results.boxes is not None and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)

            for i in range(len(results.boxes)):
                current_id = ids[i]
                bbox = results.boxes.xyxy[i].cpu().numpy()
                kpts = results[i].keypoints

                current_loc = "General Area"

                # Check Passive Zones with MiDaS Depth
                passive_zones = [z for z in zones if z['type'] == 'passive']
                if analyzer.check_trespassing(kpts, bbox, passive_zones, depth_map, person_id=current_id):
                    # If trespassing returns true for a passive zone, find which one
                    for z in passive_zones:
                         # Quick 2D check to find which specific one it is
                         fx = (kpts.xyn[0][15][0] + kpts.xyn[0][16][0]) / 2
                         fy = (kpts.xyn[0][15][1] + kpts.xyn[0][16][1]) / 2
                         foot_x, foot_y = int(fx * frame.shape[1]), int(fy * frame.shape[0])
                         if cv2.pointPolygonTest(z['polygon'], (foot_x, foot_y), False) >= 0:
                             current_loc = z['name']
                             break

                # 1. FACE RECOGNITION
                display_name = f"Person_{current_id}"
                is_authorized = False

                if current_id in identity_map:
                    display_name = identity_map[current_id]
                    is_authorized = True
                else:
                    if current_id not in unknown_check_counters:
                        unknown_check_counters[current_id] = 0
                    if unknown_check_counters[current_id] % config.FACE_CHECK_INTERVAL == 0:
                        face_crop = face_recognizer.extract_face_crop(frame, kpts)
                        if face_crop is not None:
                            found_name = face_recognizer.identify_person(face_crop)
                            if found_name:
                                # Link name to ID
                                identity_map[current_id] = found_name
                                display_name = found_name
                                is_authorized = True

                                # --- IDENTITY DISCOVERY LOG ---
                                # Log: "Person_1 Recognized as Talha"
                                logger._write_to_csv_and_terminal(f"Person_{current_id}", "Identity", f"Recognized as {found_name}", current_loc)

                                # IMMEDIATE MIGRATION: Prevent new name from re-triggering general logs
                                behavior_logged_ids.add(found_name)
                                logged_general_ids.add(found_name)
                    unknown_check_counters[current_id] += 1

                # --- B. HANDLE LOCATION STATE CHANGE ---
                detected_action = action_recognizer.update(current_id, kpts) or "Standing"

                # Initialize state if new person
                if current_id not in person_location_state: person_location_state[current_id] = None
                if current_id not in person_action_state: person_action_state[current_id] = "Standing"
                if current_id not in person_action_persistence: person_action_persistence[current_id] = 0

                last_loc = person_location_state[current_id]
                last_act = person_action_state[current_id]

                # Determine if we should log
                should_log = False

                # 1. Location change (Always log)
                if current_loc != last_loc:
                    should_log = True
                    person_action_persistence[current_id] = 0

                # 2. TRANSIENT ACTION: Picking Up (Log immediately!)
                elif detected_action == "Picking Up":
                    should_log = True
                    person_action_persistence[current_id] = 0

                # 3. HIGH PRIORITY ACTIONS: Punch, Kick, Fall (Log immediately!)
                elif detected_action in ["Punch", "Kick", "Fall"]:
                    should_log = True
                    person_action_persistence[current_id] = 0

                # 4. PERSISTENT ACTIONS: Sitting, Lying, Running (Wait for stability)
                elif detected_action != last_act:
                    person_action_persistence[current_id] += 1
                    if person_action_persistence[current_id] >= ACTION_STABILITY_THRESHOLD:
                        should_log = True
                        person_action_persistence[current_id] = 0
                else:
                    person_action_persistence[current_id] = 0

                # Suppression for General Area noise
                if should_log and current_loc == "General Area":
                    generic_label = f"Person_{current_id}"
                    if display_name in logged_general_ids or generic_label in logged_general_ids:
                        if detected_action in ["Walking", "Standing"]:
                            should_log = False


                # C. ACTION RECOGNITION (Now using Location Context)
                detected_action = action_recognizer.update(current_id, kpts)
                if detected_action:
                    # Determine if this is a high-priority action
                    is_priority = detected_action in ["Punch", "Kick", "Fall", "Picking Up"]

                    # Determine if we should log based on state change OR priority
                    should_log_act = False
                    if is_priority:
                        should_log_act = True
                    elif should_log: # 'should_log' comes from the stability check in Block B
                        should_log_act = True

                    if should_log_act:
                        # Logic: Only log if this identity hasn't been logged for this action yet
                        if current_id not in behavior_logged_ids and display_name not in behavior_logged_ids:
                            logger.log_event(display_name, "Behavior", detected_action, True, location=current_loc)

                            # Add both to suppression
                            behavior_logged_ids.add(current_id)
                            behavior_logged_ids.add(display_name)
                            if current_loc == "General Area":
                                track_ids_logged_general.add(current_id)

                # 3. GLOBAL PERSISTENCE CALCULATION
                raw_is_tres = analyzer.check_trespassing(kpts, bbox, [z for z in zones if z['type']=='restricted'], depth_map, person_id=current_id)

                if current_id not in intrusion_persistence:
                    intrusion_persistence[current_id] = 0

                if raw_is_tres:
                    intrusion_persistence[current_id] += 1
                else:
                    intrusion_persistence[current_id] = 0

                confirmed_tres = intrusion_persistence[current_id] >= PERSISTENCE_THRESHOLD
                person_statuses.append(confirmed_tres)

                is_overlap = analyzer.check_2d_overlap(bbox, [z for z in zones if z['type']=='restricted'])
                if is_overlap:
                    anybody_overlapping_2d = True

                # 4. ZONE-SPECIFIC LOGIC (RESTRICTED ONLY)
                for z in zones:
                    if z['type'] == 'restricted':
                        this_zone_tres = analyzer.check_trespassing(kpts, bbox, [z], depth_map, person_id=current_id)

                        if this_zone_tres and confirmed_tres:
                            z['last_interactor'] = display_name
                            person_location_state[current_id] = z['name']

                            if is_authorized:
                                z['status'] = "AUTHORIZED ACCESS"
                                z['color'] = config.COLOR_AUTHORIZED
                                logger.log_event(display_name, "Access", "AUTHORIZED", True, location=z['name'])
                            else:
                                z['status'] = "UNAUTHORIZED ACCESS"
                                z['color'] = config.COLOR_UNAUTHORIZED
                                logger.log_event(display_name, "Intrusion", "UNAUTHORIZED", True, location=z['name'])

        results.is_trespassing = person_statuses

        # C. Theft Detection (Restricted Zones Only)
        if not anybody_overlapping_2d:
            for z in zones:
                if z['type'] == 'restricted':
                    is_theft = analyzer.detect_theft(frame, z)
                    zone_name = z['name']

                    if is_theft:
                        last_user = z['last_interactor']
                        last_user_is_authorized = last_user in identity_map.values() if last_user else False

                        if last_user_is_authorized:
                            z['status'] = f"REMOVED BY {last_user}"
                            z['color'] = config.COLOR_AUTHORIZED
                            logger.log_event("ASSET", "Removal", f"BY_{last_user}", True, location=zone_name)
                        else:
                            z['status'] = "THEFT DETECTED"
                            z['color'] = config.COLOR_THEFT
                            logger.log_event("ASSET", "Theft", "STOLEN", True, location=zone_name)
                    else:
                        logger.log_event("ASSET", "Theft", "STOLEN", False, location=zone_name)
                        logger.log_event("ASSET", "Removal", "BY_AUTH", False, location=zone_name)

        # D. Rendering
        draw_surveillance_ui(frame, results, zones, identity_map)
        cv2.imshow("Intelligent Surveillance System", frame)

        logger.update_logs()

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()