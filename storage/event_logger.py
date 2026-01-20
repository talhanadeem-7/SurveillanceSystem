# event_logger.py

import csv
import os
import datetime
import time
import config

class EventLogger:
    def __init__(self):
        self.log_file = config.LOG_PATH
        self.active_events = {}
        self.cooldown_seconds = 3.0
        self._initialize_log_file()

    def _initialize_log_file(self):
        """
        Creates the CSV file with headers if it is missing OR empty.
        """
        file_exists = os.path.exists(self.log_file)
        if not file_exists or os.stat(self.log_file).st_size == 0:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # --- NEW HEADER: Location ---
                writer.writerow(["Timestamp", "Entity", "Action", "Status", "Location"])

    def log_event(self, entity_id, event_type, status, is_active, location="General Area"):
        """
        Handles event logging with Debouncing.
        Added 'location' parameter.
        """
        # We include location in the key so "Theft at Safe" and "Theft at Desk" are different events
        event_key = f"{entity_id}_{event_type}_{location}"
        current_time = time.time()

        if is_active:
            if event_key not in self.active_events:
                self._write_to_csv_and_terminal(entity_id, event_type, status, location)
            self.active_events[event_key] = current_time
        pass

    def update_logs(self):
        current_time = time.time()
        keys_to_remove = []
        for event_key, last_seen_time in self.active_events.items():
            if (current_time - last_seen_time) > self.cooldown_seconds:
                keys_to_remove.append(event_key)
        for key in keys_to_remove:
            del self.active_events[key]

    def _write_to_csv_and_terminal(self, entity_id, event_type, status, location):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = ""
        if event_type == "Intrusion":
            message = f"Alert! {entity_id} intruded into {location}"
        elif event_type == "Theft":
            message = f"Alert! Theft detected at {location}"
        elif event_type == "Removal":
            message = f"Info: Asset removed from {location} by {entity_id}"
        elif event_type == "Access":
            message = f"Info: {entity_id} accessed {location}"
        else:
            message = f"{event_type}: {entity_id} at {location}"

        print(f"[{timestamp}] {message}")

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, entity_id, event_type, status, location])