# event_logger.py

import datetime
import time
from storage import StorageFatalError

class EventLogger:
    def __init__(self, repository=None, run_id=None):
        self.repository = repository
        self.run_id = run_id
        self.frame_no = None
        self.active_events = {}
        self.cooldown_seconds = 3.0
        self.person_ids = {}

    def bind_run(self, repository, run_id):
        self.repository = repository
        self.run_id = run_id
        self.active_events.clear()
        self.person_ids.clear()

    def observe_person(self, track_id, display_name, is_enrolled=False):
        person_id = self.repository.upsert_person(
            self.run_id, track_id, display_name, is_enrolled)
        self.person_ids[f"Person_{track_id}"] = person_id
        self.person_ids[display_name] = person_id
        return person_id

    def log_event(self, entity_id, event_type, status, is_active, location="General Area"):
        # The Gatekeeper.. LOgs event
        """
        Handles event logging with Debouncing.
        Added 'location' parameter.
        """
        # We include location in the key so "Theft at Safe" and "Theft at Desk" are different events
        event_key = f"{entity_id}_{event_type}_{location}"
        current_time = time.time()

        if is_active:
            if event_key not in self.active_events:
                self.write_event(entity_id, event_type, status, location)
            self.active_events[event_key] = current_time

    def update_logs(self):
        current_time = time.time()
        keys_to_remove = []
        for event_key, last_seen_time in self.active_events.items():
            if (current_time - last_seen_time) > self.cooldown_seconds:
                keys_to_remove.append(event_key)
        for key in keys_to_remove:
            del self.active_events[key]

    def write_event(self, entity_id, event_type, status, location):
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

        if self.repository is None or self.run_id is None:
            raise StorageFatalError("No active database run is bound to the event logger")
        self.repository.log_event(
            self.run_id, event_type, status, entity_id, location,
            person_id=self.person_ids.get(entity_id),
            timestamp=timestamp, frame_no=self.frame_no,
        )

    # Compatibility for the out-of-scope desktop entry point; never writes CSV.
    _write_to_csv_and_terminal = write_event
