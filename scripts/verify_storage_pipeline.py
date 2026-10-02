"""Exercise the real pipeline with local models and a non-rendering Streamlit UI.

Run before and after storage changes with --label; VLM is disabled to exclude
network variability. Detection outputs are captured without changing the pipeline.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class UI:
    def __init__(self):
        self.session_state = State()
        self.errors = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def columns(self, widths):
        return [self] * len(widths)

    def empty(self):
        return self

    def button(self, *args, **kwargs):
        return False

    def error(self, message):
        self.errors.append(message)

    def __getattr__(self, name):
        return lambda *a, **k: None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--clip', default='office cctv.mp4')
    parser.add_argument('--rules', action='store_true', help='Enable occupancy and unknown-dwell verification rules')
    parser.add_argument('--compare-label', help='Compare security event payloads with a previous isolated run')
    args = parser.parse_args()
    (ROOT / 'scratchpad').mkdir(exist_ok=True)
    import config
    config.USE_VLM = False
    config.DATABASE_URL = 'sqlite:///' + str(ROOT / 'scratchpad' / f'{args.label}.db')
    import cv2
    import numpy as np
    import streamlit_app as app
    ui = UI()
    app.st = ui
    # spinner needs a context manager during face-model initialization.
    ui.spinner = lambda *a, **k: ui
    app.initialize_surveillance_components()
    state = ui.session_state
    from datetime import datetime
    state.recording_start_time = datetime(2026, 9, 27, 10, 0, 0)
    if args.rules:
        from storage.repository import Repository
        with Repository() as repository:
            assert not repository.get_rules(), 'Use a fresh benchmark label for rule verification'
            repository.save_rule('Desk occupancy', 'occupancy',
                dict(zone_name='Test desk', max_people=0, duration_seconds=.5), cooldown_seconds=5)
            repository.save_rule('Unknown dwell', 'unknown_dwell',
                dict(duration_seconds=2.0), cooldown_seconds=5)
    state.video_path = str(ROOT / 'data/uploaded_videos' / args.clip)
    cap = cv2.VideoCapture(state.video_path)
    ok, frame = cap.read()
    cap.release()
    assert ok
    state.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(w*.35), int(h*.35), int(w*.65), int(h*.65)
    state.zones = [dict(name='Test desk', type='restricted', status='SECURE',
                        color=config.COLOR_SECURE, coords=(x1,y1,x2,y2),
                        orig_coords=(x1,y1,x2,y2), missing_counter=0, last_interactor=None,
                        reference_patch=cv2.GaussianBlur(cv2.cvtColor(
                            frame[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY), (5,5), 0))]
    counts = Counter()
    spans = {}
    detector = state.detector
    # Discover the exact public entry point from the pipeline, not a copy of it.
    original = detector.track_and_detect
    def detect(*a, **k):
        result = original(*a, **k)
        counts['frames'] += 1
        if result is not None and result.boxes is not None and result.boxes.id is not None:
            for tid in result.boxes.id.cpu().numpy().astype(int):
                counts['detections'] += 1
                counts['ephemeral'] += int(tid < 0)
                if tid >= 0:
                    spans.setdefault(str(tid), []).append(counts['frames'])
        return result
    detector.track_and_detect = detect
    emitted_events = []
    original_write = state.logger.write_event
    def write_event(entity, action, status, location):
        original_write(entity, action, status, location)
        emitted_events.append((entity, action, status, location))
    state.logger.write_event = write_event
    app.run_surveillance()
    from storage.repository import Repository
    with Repository() as repository:
        run = next(r for r in repository.get_runs() if r['id'] == state.run_id)
        people = repository.get_people(state.run_id)
        zones = repository.get_zones(state.run_id)
        events = repository.get_events(state.run_id)
        camera_ids = {c['id'] for c in repository.get_cameras()}
        person_ids = {p['id'] for p in people}
        zone_ids = {z['id'] for z in zones}
        assert run['camera_id'] in camera_ids
        assert run['status'] == 'completed' and run['ended_at'] is not None
        assert run['frames_processed'] == state.surveillance_metrics['processed_frames']
        assert run['frames_skipped'] == state.surveillance_metrics['frame_errors']
        assert {str(p['track_id']) for p in people} == set(spans)
        assert len(zones) == len(state.zones)
        assert zones[0]['geometry_json']['coords'] == [x1,y1,x2,y2]
        assert [(e['details_json']['entity'],e['action'],e['status'],e['details_json']['location'])
                for e in events] == emitted_events
        for event in events:
            assert event['frame_no'] is not None and event['frame_no'] % config.FRAME_SKIP == 0
            assert event['zone_id'] in zone_ids or event['details_json']['location'] == 'General Area'
            assert event['person_id'] in person_ids or event['details_json']['entity'] == 'ASSET'
        database_counts = dict(runs=1, persons=len(people), zones=len(zones),
                               events=len(events), activities=len(repository.get_activities(state.run_id)),
                               status=run['status'], emitted_events_match=True)
        saved_alerts = repository.get_alerts(state.run_id)
        if args.rules:
            assert {a['details_json']['rule_name'] for a in saved_alerts} == {'Desk occupancy','Unknown dwell'}
        else:
            assert saved_alerts == []
        if args.compare_label:
            before_url = 'sqlite:///' + str(ROOT/'scratchpad'/f'{args.compare_label}.db')
            with Repository(before_url) as before:
                previous = before.get_events()
            def comparable(rows):
                return [(e['frame_no'],e['action'],e['status'],e['details_json']) for e in rows]
            assert comparable(events) == comparable(previous), 'Security events changed'
    result = dict(metrics=state.surveillance_metrics, counts=dict(counts),
                  spans={k: [min(v), max(v), len(v)] for k,v in spans.items()},
                  database=database_counts, errors=ui.errors,
                  alerts=[dict(rule=a['details_json']['rule_name'], timestamp=a['triggered_at'].isoformat(),
                               offset=a['video_offset_s'], frame=a['frame_no'], message=a['message'])
                          for a in saved_alerts])
    (ROOT/'scratchpad'/f'{args.label}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    expected = {'office cctv.mp4': (1050, 29, 6, 147),
                'crowd sample.mp4': (787, 117, 27, 23),
                'crowded sample2.mp4': (2502, 124, 43, 40)}
    if args.clip in expected:
        actual = (counts['detections'], counts['ephemeral'], len(spans),
                  int(np.median([len(v) for v in spans.values()])))
        assert actual == expected[args.clip], (args.clip, actual, expected[args.clip])
        if args.clip == 'office cctv.mp4':
            assert max(spans['4']) == 429 and min(spans['6']) == 450
    assert not ui.errors, ui.errors


if __name__ == '__main__':
    main()

