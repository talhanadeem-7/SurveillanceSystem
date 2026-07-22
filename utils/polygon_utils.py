import cv2
import numpy as np
import config

class PolygonDrawer:
    def __init__(self):
        self.points = [] # List of (x, y) tuples
        self.done = False
        self.current_mouse = (0, 0)

    def mouse_callback(self, event, x, y, flags, param):
        # listens for what your mouse is doing.
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse = (x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Add a point
            self.points.append((x, y))
            print(f"Point added: {x}, {y}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to close the shape
            if len(self.points) > 2:
                self.done = True

    def run(self, frame, window_name):
        # The Drawing Loop. It pauses the video and lets you draw until you are finished.
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while not self.done:
            display = frame.copy()
            
            # Draw existing lines
            if len(self.points) > 0:
                # Draw points
                for pt in self.points:
                    cv2.circle(display, pt, 3, config.COLOR_PASSIVE, -1)
                
                # Draw lines between points
                if len(self.points) > 1:
                    cv2.polylines(display, [np.array(self.points)], False, config.COLOR_PASSIVE, 2)
                
                # Draw "rubber band" line to mouse cursor
                cv2.line(display, self.points[-1], self.current_mouse, (200, 200, 200), 1)

            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC to cancel this shape
                return None
            if key == 13: # Enter to force finish
                if len(self.points) > 2:
                    self.done = True

        return self.points