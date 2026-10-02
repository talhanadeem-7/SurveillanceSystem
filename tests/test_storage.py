"""Repository persistence and real pipeline control-flow regression tests."""
import ast
import copy
import datetime as datetime_module
from alerts.rules_engine import RulesEngine, FrameState, TrackState, resolve_rules
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import config
from scripts.verify_storage_pipeline import UI
from storage import StorageFatalError
from storage.activity_observer import ActivityObserver
from storage.event_logger import EventLogger
from storage.repository import Repository
from utils.log_analyzer import LogAnalyzer


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.url = 'sqlite:///' + str(self.root / 'test.db')
        self.base_patch = patch.object(config, 'BASE_DIR', str(self.root))
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)
        self.repo = Repository(self.url)
        self.addCleanup(self.repo.close)

    def run_record(self):
        return self.repo.start_run('office.mp4', source='uploads/office.mp4')

    def test_batch_visibility_filters_and_run_isolation(self):
        run = self.run_record()
        person = self.repo.upsert_person(run, 1, 'Person_1', timestamp='2026-01-01 10:00:00')
        zones = self.repo.save_zones(run, [dict(name='Desk', type='restricted', coords=(0,0,10,10))])
        self.repo.log_event(run, 'Intrusion', 'UNAUTHORIZED', 'Person_1', 'Desk',
                            person_id=person, timestamp='2026-01-01 10:00:00', frame_no=8)
        with Repository(self.url) as reader:
            self.assertEqual(reader.get_events(), [])
        self.repo.flush()
        rows = self.repo.get_events(run_id=run, since='2026-01-01', action='Intrusion')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['zone_id'], zones['Desk'])
        self.assertEqual(rows[0]['person_id'], person)
        self.assertEqual(self.repo.get_events(run_id='other'), [])
        self.assertEqual(self.repo.get_events(since='2026-02-01'), [])
        self.assertEqual(self.repo.get_events(action='Theft'), [])
        other = self.run_record()
        self.assertNotEqual(person, self.repo.upsert_person(other, 1))
        self.repo.end_run(run, 'completed', 20, 2)
        record = next(r for r in self.repo.get_runs() if r['id'] == run)
        self.assertEqual((record['status'], record['frames_processed'], record['frames_skipped']),
                         ('completed', 20, 2))

    def test_person_identity_and_first_last_seen(self):
        run = self.run_record()
        first = self.repo.upsert_person(run, 4, timestamp='2026-01-01')
        self.assertEqual(first, self.repo.upsert_person(run, 4, 'Alice', True, '2026-01-02'))
        self.assertIsNone(self.repo.upsert_person(run, -1))
        self.repo.flush()
        people = self.repo.get_people(run)
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]['display_name'], 'Alice')
        self.assertTrue(people[0]['is_enrolled'])
        self.assertEqual(people[0]['first_seen'], datetime(2026,1,1))
        self.assertEqual(people[0]['last_seen'], datetime(2026,1,2))

    def test_activity_deduplicates_polled_results_and_drains_new_results(self):
        run = self.run_record()
        observer = ActivityObserver(self.repo, run)
        result = SimpleNamespace(track_id=3, activity='walking', description='Walking by desk',
                                 timestamp='2026-01-01T10:00:00', confidence=.8, frame_ids=[2,4])
        observer.log_activities([result, result])
        result.timestamp = '2026-01-01T10:00:01'
        observer.log_activity(result)
        self.repo.flush()
        self.assertEqual(len(self.repo.get_activities(run)), 2)
        self.assertEqual(self.repo.get_activities(run)[0]['frame_no'], 4)

    def test_logger_preserves_debounce_and_person_attribution(self):
        run = self.run_record()
        logger = EventLogger(self.repo, run)
        logger.observe_person(4, 'Alice', True)
        logger.frame_no = 12
        logger.log_event('Alice', 'Access', 'AUTHORIZED', True)
        logger.log_event('Alice', 'Access', 'AUTHORIZED', True)
        self.repo.flush()
        rows = self.repo.get_events()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['person_id'], self.repo.get_people(run)[0]['id'])
        self.assertEqual(rows[0]['frame_no'], 12)

    def test_all_fields_survive_reopening_database(self):
        camera = self.repo.ensure_camera('uploads/office.mp4', 'Office camera')
        self.assertEqual(camera, self.repo.ensure_camera('uploads/office.mp4'))
        run = self.repo.start_run('office.mp4', camera_id=camera,
                                  started_at='2026-01-01T10:00:00', metadata={'fps': 30})
        zones = self.repo.save_zones(run, [
            dict(name='Desk', type='restricted', coords=(1,2,30,40), orig_coords=(1,2,30,40)),
            dict(name='Lobby', type='passive', polygon=np.array([[0,0],[10,0],[10,10]])),
        ])
        person = self.repo.upsert_person(run, 7, 'Alice', True, '2026-01-01T10:00:01')
        for action in ('Identity','Intrusion','Access','Theft','Removal'):
            self.repo.log_event(run, action, 'test status', 'Alice', 'Desk',
                person_id=person, timestamp='2026-01-01T10:00:02', frame_no=60,
                details={'confidence': .95}, snapshot_path='snapshots/example.jpg')
        observer = ActivityObserver(self.repo, run)
        observer.log_activity(SimpleNamespace(track_id=7, activity='walking',
            description='Walking past desk', confidence=.875,
            timestamp='2026-01-01T10:00:03', frame_ids=[60,62,64]))
        self.repo.end_run(run, 'completed', 120, 2)
        with Repository(self.url) as reader:
            cameras = reader.get_cameras()
            self.assertEqual(len(cameras), 1)
            self.assertEqual((cameras[0]['id'], cameras[0]['name'], cameras[0]['source']),
                             (camera, 'Office camera', 'uploads/office.mp4'))
            self.assertIsInstance(cameras[0]['created_at'], datetime)
            saved_run = reader.get_runs()[0]
            self.assertEqual(saved_run['camera_id'], camera)
            self.assertEqual(saved_run['video_name'], 'office.mp4')
            self.assertEqual(saved_run['started_at'], datetime(2026,1,1,10))
            self.assertIsInstance(saved_run['ended_at'], datetime)
            self.assertEqual(saved_run['metadata_json'], {'fps': 30})
            self.assertEqual((saved_run['status'], saved_run['frames_processed'],
                              saved_run['frames_skipped']), ('completed',120,2))
            geometry = {z['name']: z for z in reader.get_zones(run)}
            self.assertEqual(geometry['Desk']['geometry_json'],
                             {'coords':[1,2,30,40], 'orig_coords':[1,2,30,40]})
            self.assertEqual(geometry['Lobby']['geometry_json'],
                             {'polygon':[[0,0],[10,0],[10,10]]})
            saved_person = reader.get_people(run)[0]
            self.assertEqual((saved_person['id'],saved_person['track_id'],
                              saved_person['display_name'],saved_person['is_enrolled']),
                             (person,7,'Alice',True))
            self.assertEqual(saved_person['first_seen'], datetime(2026,1,1,10,0,1))
            self.assertEqual(saved_person['last_seen'], datetime(2026,1,1,10,0,3))
            events = reader.get_events(run)
            self.assertEqual([e['action'] for e in events],
                             ['Identity','Intrusion','Access','Theft','Removal'])
            for event in events:
                self.assertEqual((event['run_id'],event['person_id'],event['zone_id']),
                                 (run,person,zones['Desk']))
                self.assertEqual(event['timestamp'], datetime(2026,1,1,10,0,2))
                self.assertEqual(event['frame_no'], 60)
                self.assertEqual(event['status'], 'test status')
                self.assertEqual(event['snapshot_path'], 'snapshots/example.jpg')
                self.assertEqual(event['details_json'],
                                 {'entity':'Alice','location':'Desk','confidence':.95})
            activity = reader.get_activities(run)[0]
            self.assertEqual((activity['person_id'],activity['track_id']), (person,7))
            self.assertEqual(activity['activity_label'], 'walking')
            self.assertEqual(activity['description'], 'Walking past desk')
            self.assertEqual(activity['confidence'], .875)
            self.assertEqual(activity['timestamp'], datetime(2026,1,1,10,0,3))
            self.assertEqual(activity['frame_no'], 64)
            self.assertEqual(activity['details_json'], {'frame_ids':[60,62,64]})
        docs = LogAnalyzer(run_id=run, database_url=self.url).get_all_logs_formatted()
        self.assertEqual(len(docs), 6)
        self.assertEqual(docs[1], 'At 2026-01-01 10:00:02, Alice was detected intruding at the Desk.')
        self.assertEqual(LogAnalyzer(run_id='other', database_url=self.url).get_all_logs_formatted(), [])

    def test_actual_lock_is_fatal_and_status_recovers(self):
        run = self.run_record()
        locker = sqlite3.connect(self.root/'test.db')
        locker.execute('BEGIN IMMEDIATE')
        try:
            self.repo.log_event(run, 'Theft', 'STOLEN', 'ASSET')
            with self.assertRaises(StorageFatalError):
                self.repo.flush()
            with self.assertRaises(StorageFatalError):
                self.repo.end_run(run, 'aborted')
            self.assertTrue(list(self.repo.recovery_dir.glob('*.json')))
        finally:
            locker.rollback()
            locker.close()
        with Repository(self.url) as recovered:
            self.assertEqual(recovered.get_runs()[0]['status'], 'aborted')
            self.assertEqual(recovered.get_events(), [])
            self.assertEqual(list(recovered.recovery_dir.glob('*.json')), [])

    def test_real_readonly_connection_failure_is_wrapped(self):
        run = self.run_record()
        connection = self.repo.session.connection().connection.driver_connection
        connection.execute('PRAGMA query_only=ON')
        self.repo.log_event(run, 'Theft', 'STOLEN', 'ASSET')
        with self.assertRaises(StorageFatalError):
            self.repo.flush()
        connection.execute('PRAGMA query_only=OFF')
        self.repo.end_run(run, 'aborted')
        self.assertEqual(self.repo.get_runs()[0]['status'], 'aborted')

    def test_successful_abort_clears_older_pending_completion(self):
        run = self.run_record()
        locker = sqlite3.connect(self.root/'test.db')
        locker.execute('BEGIN IMMEDIATE')
        try:
            with self.assertRaises(StorageFatalError):
                self.repo.end_run(run, 'completed')
        finally:
            locker.rollback()
            locker.close()
        self.repo.end_run(run, 'aborted')
        self.assertEqual(list(self.repo.recovery_dir.glob('*.json')), [])
        with Repository(self.url) as reader:
            self.assertEqual(reader.get_runs()[0]['status'], 'aborted')

    def pipeline(self, mode):
        """Execute the actual function AST, replacing only UI/vision dependencies."""
        source = Path('streamlit_app.py').read_text(encoding='utf-8')
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'run_surveillance')
        ui = UI()
        if mode == 'stop':
            ui.button = lambda *a, **k: True
        state = ui.session_state
        state.zones = [dict(type='restricted', name='Desk', coords=(0,0,1,1))]
        state.video_path = 'test.mp4'
        state.logger = EventLogger()
        detector = Mock()
        detector.guardian.snapshot_root = self.root
        detector.guardian.get_stats.return_value = dict(known_identities=0, active_remaps=0, retired_ids=0)
        state.detector = detector
        activity = SimpleNamespace(enabled=False, activity_cache={}, shutdown=lambda **kw: 0,
                                   get_activity=lambda *a: None, cleanup_inactive=lambda *a: None)
        if mode == 'activity':
            def shutdown(**kwargs):
                activity.activity_cache[3] = SimpleNamespace(track_id=3, activity='sitting',
                    description='Sitting at desk', confidence=.9,
                    timestamp='2026-01-01T10:00:00', frame_ids=[60,62,64])
                return 0
            activity.shutdown = shutdown
        state.activity_analyzer = activity
        cap = Mock()
        cap.get.return_value = 30
        cap.isOpened.return_value = True
        cap.grab.side_effect = [True]*64 + [False]
        cap.retrieve.return_value = (True, np.zeros((4,4,3), dtype=np.uint8))
        analyzer = Mock()
        analyzer.align_zones.side_effect = lambda frame, zones: zones
        analyzer.detect_theft.return_value = False
        analyzer.person_depth_filters = {}
        analyzer.align_failures_total = 0
        detector.track_and_detect.return_value = SimpleNamespace(boxes=None)
        def inject(frame):
            if mode == 'activity-in-loop' and detector.track_and_detect.call_count == 1:
                activity.activity_cache[3] = SimpleNamespace(track_id=3, activity='walking',
                    description='Walking out of view', confidence=.9,
                    timestamp='2026-01-01T10:00:00', frame_ids=[2])
                activity.cleanup_inactive = lambda *a: activity.activity_cache.clear()
            if mode == 'interrupt':
                raise KeyboardInterrupt()
            if mode == 'failures':
                raise RuntimeError('bad decoder')
            if mode == 'readonly':
                connection = self.repo.session.connection().connection.driver_connection
                connection.execute('PRAGMA query_only=ON')
                self.repo.log_event(state.run_id, 'Theft', 'STOLEN', 'ASSET')
            return SimpleNamespace(boxes=None)
        detector.track_and_detect.side_effect = inject
        # Restore writes only after the first failed flush: final status can then persist.
        original_flush = self.repo.flush
        def flush():
            try:
                return original_flush()
            except StorageFatalError:
                self.repo.session.connection().connection.driver_connection.execute('PRAGMA query_only=OFF')
                raise
        self.repo.flush = flush
        import os
        cv = Mock()
        cv.VideoCapture.return_value = cap
        context = dict(st=ui, datetime=datetime_module, RulesEngine=RulesEngine,
                       FrameState=FrameState, TrackState=TrackState, resolve_rules=resolve_rules, time=time, os=os, logging=logging, copy=copy, cv2=cv, np=np,
                       config=config, Repository=lambda: self.repo, StorageFatalError=StorageFatalError,
                       ActivityObserver=ActivityObserver, ensure_activity_analyzer=lambda: activity,
                       setup_surveillance_memory=lambda: (analyzer, {}, {}, {}, {}, {}, {}, 5),
                       draw_surveillance_ui=lambda *a: None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'streamlit_app.py', 'exec'), context)
        with patch.dict('sys.modules', {'vision.identity_guardian': SimpleNamespace(IdentityGuardian=Mock(return_value=detector.guardian))}):
            if mode == 'interrupt':
                with self.assertRaises(KeyboardInterrupt):
                    context['run_surveillance']()
            else:
                context['run_surveillance']()
        cap.release.assert_called_once()
        with Repository(self.url) as reader:
            run = reader.get_runs()[0]
        return state, run, ui, detector

    def test_pipeline_aborts_first_readonly_write_not_thirty_frames(self):
        state, run, ui, detector = self.pipeline('readonly')
        self.assertEqual(detector.track_and_detect.call_count, 1)
        self.assertEqual(run['status'], 'aborted')
        self.assertEqual(run['frames_skipped'], 0)
        self.assertEqual(state.surveillance_metrics['frame_errors'], 0)
        self.assertFalse(state.processing_complete)
        self.assertTrue(any('Fatal storage error' in error for error in ui.errors))

    def test_pipeline_thirty_frame_failures(self):
        state, run, ui, detector = self.pipeline('failures')
        self.assertEqual(detector.track_and_detect.call_count, 30)
        self.assertEqual(run['status'], 'aborted')
        self.assertEqual(run['frames_skipped'], 30)
        self.assertFalse(state.processing_complete)

    def test_pipeline_keyboard_interrupt(self):
        state, run, ui, detector = self.pipeline('interrupt')
        self.assertEqual(run['status'], 'interrupted')
        self.assertFalse(state.processing_complete)

    def test_pipeline_completed(self):
        state, run, ui, detector = self.pipeline('completed')
        self.assertEqual(run['status'], 'completed')
        self.assertEqual(run['frames_processed'], 32)
        self.assertTrue(state.processing_complete)

    def test_pipeline_user_stop(self):
        state, run, ui, detector = self.pipeline('stop')
        self.assertEqual(run['status'], 'interrupted')
        self.assertEqual(run['frames_processed'], 0)
        self.assertFalse(state.processing_complete)

    def test_pipeline_saves_activity_completed_during_shutdown(self):
        state, run, ui, detector = self.pipeline('activity')
        with Repository(self.url) as reader:
            observations = reader.get_activities(run['id'])
            people = reader.get_people(run['id'])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]['person_id'], people[0]['id'])
        self.assertEqual(observations[0]['activity_label'], 'sitting')
        self.assertEqual(observations[0]['frame_no'], 64)
        self.assertEqual(run['status'], 'completed')

    def test_pipeline_saves_activity_for_track_no_longer_detected(self):
        state, run, ui, detector = self.pipeline('activity-in-loop')
        self.assertEqual(state.activity_analyzer.activity_cache, {})
        with Repository(self.url) as reader:
            observations = reader.get_activities(run['id'])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]['activity_label'], 'walking')
        self.assertEqual(observations[0]['frame_no'], 2)


if __name__ == '__main__':
    unittest.main()


