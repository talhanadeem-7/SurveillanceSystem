import cv2
import numpy as np
import config
from utils.polygon_utils import PolygonDrawer # <--- Import new tool

class ZoneSelector:
    def __init__(self):
        self.zones = [] 
        
        # Rectangle variables
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.curr_x, self.curr_y = -1, -1
        self.finished_box = None

    # --- RECTANGLE LOGIC (PHASE 1) ---
    def rect_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.curr_x, self.curr_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1, x2, y2 = min(self.ix, x), min(self.iy, y), max(self.ix, x), max(self.iy, y)
            if (x2 - x1) > 20 and (y2 - y1) > 20: 
                self.finished_box = (x1, y1, x2, y2)

    def get_zones(self, frame):
        # ---------------------------------------------------------
        # PHASE 1: RESTRICTED ZONES (RECTANGLES)
        # ---------------------------------------------------------
        cv2.namedWindow("Setup")
        cv2.setMouseCallback("Setup", self.rect_mouse_callback)
        
        print("\n=== PHASE 1: RESTRICTED ZONES (Red) ===")
        print("Draw boxes around items. Press ENTER to move to Phase 2.")

        while True:
            display = frame.copy()
            
            # Draw saved zones
            for z in self.zones:
                if z['type'] == 'restricted':
                    c = z['coords']
                    cv2.rectangle(display, (c[0], c[1]), (c[2], c[3]), config.COLOR_THEFT, 2)
                    cv2.putText(display, z['name'], (c[0], c[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_THEFT, 2)

            if self.drawing:
                cv2.rectangle(display, (self.ix, self.iy), (self.curr_x, self.curr_y), (255, 255, 255), 1)

            cv2.imshow("Setup", display)
            
            # Input Logic
            key = cv2.waitKey(1) & 0xFF
            if key == 13: # ENTER -> Go to Phase 2
                break
                
            if self.finished_box:
                # Ask for name
                print(f"Defining Restricted Zone...")
                name = input(">> Name this Restricted Zone (e.g. Laptop): ").strip()
                if not name: name = f"Restricted_{len(self.zones)}"
                
                self.zones.append({
                    "type": "restricted",
                    "name": name,
                    "coords": self.finished_box
                })
                self.finished_box = None

        # ---------------------------------------------------------
        # PHASE 2: PASSIVE ZONES (POLYGONS)
        # ---------------------------------------------------------
        print("\n=== PHASE 2: PASSIVE ZONES (Yellow) ===")
        print("Left-Click to add points. Right-Click to close shape.")
        print("Press ESC to finish setup.")
        
        while True:
            # We instantiate a new drawer for every shape
            drawer = PolygonDrawer()
            print("\n>> Draw a Passive Zone (or press ESC in window to finish)...")
            
            # This blocks until user finishes drawing ONE polygon
            points = drawer.run(frame, "Setup")
            
            if points is None: # User pressed ESC
                break
            
            # Ask for name
            name = input(">> Name this Passive Zone (e.g. Hallway): ").strip()
            if not name: name = f"Passive_{len(self.zones)}"
            
            self.zones.append({
                "type": "passive",
                "name": name,
                "polygon": np.array(points, np.int32) # Store as numpy array for OpenCV
            })
            
            # Draw it on the "background" for the next loop
            cv2.polylines(frame, [np.array(points, np.int32)], True, config.COLOR_PASSIVE, 2)

        cv2.destroyWindow("Setup")
        
        # Prepare final structure
        final_zones = []
        for z in self.zones:
            zone_data = {
                "name": z['name'],
                "type": z['type'],
                "status": "SECURE",
                "color": config.COLOR_SECURE if z['type'] == 'restricted' else config.COLOR_PASSIVE,
                "last_interactor": None
            }
            
            if z['type'] == 'restricted':
                c = z['coords']
                zone_data["coords"] = c
                zone_data["orig_coords"] = c
                # Prepare patch for theft
                patch = frame[c[1]:c[3], c[0]:c[2]]
                zone_data["reference_patch"] = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (5,5), 0)
                zone_data["missing_counter"] = 0
            else:
                # Passive zones don't have coords/theft logic, just a polygon
                zone_data["polygon"] = z['polygon']
                zone_data["coords"] = (0,0,0,0) # Placeholder
                
            final_zones.append(zone_data)
            
        return final_zones

def draw_surveillance_ui(frame, results, zones, identity_map=None, activity_cache=None):
    # 1. Draw Zones (Logic unchanged)
    for z in zones:
        if z['type'] == 'restricted':
            cv2.rectangle(frame, (z['coords'][0], z['coords'][1]), (z['coords'][2], z['coords'][3]), z['color'], 2)
            cv2.putText(frame, f"{z['name']}: {z['status']}", (z['coords'][0], z['coords'][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, z['color'], 2)
        elif z['type'] == 'passive':
            cv2.polylines(frame, [z['polygon']], True, config.COLOR_PASSIVE, 2)
            M = cv2.moments(z['polygon'])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, z['name'], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_PASSIVE, 2)

    # 2. Draw Skeleton, Box, and Identity
    if results.boxes is not None and results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
        
        for i, box_data in enumerate(results.boxes):
            p_id = ids[i]
            label = f"ID:{p_id}"
            
            # Use your existing color logic
            box_color = config.COLOR_SECURE 
            if identity_map and p_id in identity_map:
                label = identity_map[p_id]
                box_color = config.COLOR_AUTHORIZED 
            
            # A. Draw Bounding Box
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw Name/ID Label
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Draw the latest VLM activity beside the person's box.
            if activity_cache and p_id in activity_cache:
                activity = activity_cache[p_id]
                activity_label = f"{activity.activity} ({activity.confidence:.2f})"
                cv2.putText(
                    frame,
                    activity_label,
                    (x2 + 5, max(20, y1 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                )

            # --- DRAW SKELETON ---
            kpts = results.keypoints[i]
            pts = kpts.xy[0].cpu().numpy()
            conf = kpts.conf[0].cpu().numpy()

            # B. Draw Lines (Skeletal Links)
            for edge in config.SKELETON_EDGES:
                p1_idx, p2_idx = edge
                if conf[p1_idx] > 0.5 and conf[p2_idx] > 0.5:
                    p1 = (int(pts[p1_idx][0]), int(pts[p1_idx][1]))
                    p2 = (int(pts[p2_idx][0]), int(pts[p2_idx][1]))
                    cv2.line(frame, p1, p2, box_color, 2)

            # C. Draw Joints (White dots)
            for j in range(len(pts)):
                if conf[j] > 0.5:
                    kx, ky = int(pts[j][0]), int(pts[j][1])
                    cv2.circle(frame, (kx, ky), 4, (255, 255, 255), -1)