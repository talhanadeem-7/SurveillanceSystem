from ultralytics import YOLO
import config

class PoseDetector:
    def __init__(self):
        # Load YOLOv8n-pose model
        self.model = YOLO(config.MODEL_PATH)

    def track_and_detect(self, frame):
        """
        Runs tracking on the frame.
        Returns:
            results: The raw YOLO results object
        """
        # We use .track to maintain ID persistence across frames
        results = self.model.track(
            frame, 
            persist=True, 
            conf=config.CONFIDENCE_THRESHOLD, 
            verbose=False,
            classes=[0] # Only detect 'Person' class
        )
        return results[0]