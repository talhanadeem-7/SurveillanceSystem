"""
Activity observation storage layer.

Stores VLM-recognized activities in a separate CSV file to avoid cluttering
the main event_logs.csv with every single activity (which could be hundreds/thousands).

Activity observations are periodic (e.g., every 30 frames) and represent ongoing
behavior, not discrete events like "Intrusion" or "Theft".
"""

import csv
import os
import datetime
import logging

logger = logging.getLogger(__name__)


class ActivityObserver:
    """
    Stores human activity observations from VLM analysis.
    
    Maintains a separate CSV file with columns:
    - Timestamp
    - Track ID
    - Activity Label
    - Description
    - Confidence
    - Duration (seconds, if track is still alive)
    """

    def __init__(self, storage_path: str = None):
        """
        Args:
            storage_path: Path to activity_observations.csv (if None, uses default)
        """
        if storage_path is None:
            import config
            base_dir = os.path.dirname(os.path.abspath(config.__file__))
            storage_path = os.path.join(base_dir, "storage", "activity_observations.csv")
        
        self.storage_path = storage_path
        self._initialize_storage()

    def _initialize_storage(self):
        """Create CSV file with headers if it doesn't exist."""
        file_exists = os.path.exists(self.storage_path)
        if not file_exists or os.stat(self.storage_path).st_size == 0:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            try:
                with open(self.storage_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp",
                        "Track_ID",
                        "Activity_Label",
                        "Description",
                        "Confidence",
                        "Frame_ID"
                    ])
                logger.info(f"Activity observation storage initialized: {self.storage_path}")
            except IOError as e:
                logger.error(f"Failed to initialize activity storage: {e}")

    def log_activity(self, activity_result):
        """
        Log an activity observation.
        
        Args:
            activity_result: ActivityResult object from VLMActivityAnalyzer
        """
        if not activity_result:
            return
        
        try:
            with open(self.storage_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    activity_result.timestamp,
                    activity_result.track_id,
                    activity_result.activity,
                    activity_result.description,
                    f"{activity_result.confidence:.3f}",
                    activity_result.frame_ids[-1] if activity_result.frame_ids else ""
                ])
        except IOError as e:
            logger.error(f"Failed to write activity observation: {e}")

    def log_activities(self, activity_results: list):
        """Log multiple activity observations."""
        for result in activity_results:
            self.log_activity(result)
