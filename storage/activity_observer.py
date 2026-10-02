"""Persist completed VLM observations on the pipeline thread, once per result."""


class ActivityObserver:
    def __init__(self, repository, run_id):
        self.repository = repository
        self.run_id = run_id
        self.last_result = {}

    def log_activity(self, result):
        if result is None or result.track_id < 0:
            return
        signature = (result.timestamp, tuple(result.frame_ids or []))
        if self.last_result.get(result.track_id) == signature:
            return
        self.repository.log_activity(
            self.run_id, result.track_id, result.activity, result.description,
            result.confidence, result.timestamp,
            frame_no=result.frame_ids[-1] if result.frame_ids else None,
            details={'frame_ids': result.frame_ids or []},
        )
        self.last_result[result.track_id] = signature

    def log_activities(self, results):
        for result in results:
            self.log_activity(result)
