"""Synthetic video-time rules and local SQLite persistence; no paid services."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from alerts.rules_engine import Alert, FrameState, RulesEngine, TrackState, resolve_rules, validate_rule
from storage.repository import Repository
from storage import StorageFatalError
from utils.log_analyzer import LogAnalyzer

START = datetime(2026, 9, 28, 21, 59, 59)  # Monday


def rule(kind, **params):
    defaults = {'zone_schedule': dict(zone_name='Desk', start='22:00', end='06:00'),
                'occupancy': dict(zone_name='Desk', max_people=1, duration_seconds=2.0),
                'unknown_dwell': dict(duration_seconds=2.0),
                'activity': dict(label='running', min_confidence=.8)}
    return dict(id='rule-1', name='Test rule', type=kind, severity='high', enabled=True,
                cooldown_seconds=5.0, zone_id=None, params_json={**defaults[kind], **params})


def person(track=4, zones=('Desk',), enrolled=False, label=None, confidence=0):
    return TrackState(track, f'Person_{track}', enrolled, frozenset(zones), label, confidence)


class RulesEngineTest(unittest.TestCase):
    def test_occupancy_exact_limit_and_duration(self):
        engine = RulesEngine([rule('occupancy')], 10, START)
        self.assertEqual(engine.update(FrameState(0, (person(),))), [])  # Exactly N.
        crowd = (person(), person(5))
        self.assertEqual(engine.update(FrameState(10, crowd)), [])
        self.assertEqual(engine.update(FrameState(29, crowd)), [])
        alerts = engine.update(FrameState(30, crowd))
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].video_offset_s, 3)
        self.assertEqual(alerts[0].details_json['count'], 2)
        self.assertEqual(engine.update(FrameState(500, crowd)), [])  # No continuous spam.

    def test_unknown_dwell_exact_threshold_identity_and_zone(self):
        engine = RulesEngine([rule('unknown_dwell', zone_name='Desk')], 10, START)
        self.assertEqual(engine.update(FrameState(0, (person(), person(5, enrolled=True), person(6,zones=())))), [])
        alerts = engine.update(FrameState(20, (person(), person(5,enrolled=True), person(6,zones=()))))
        self.assertEqual([a.track_id for a in alerts], [4])
        self.assertEqual(engine.update(FrameState(21, (person(enrolled=True),))), [])
        self.assertEqual(engine.active, {})

    def test_cooldown_and_reset_after_condition_clears(self):
        engine = RulesEngine([rule('unknown_dwell', duration_seconds=0)], 10, START)
        self.assertEqual(len(engine.update(FrameState(0, (person(),)))), 1)
        engine.update(FrameState(10))
        self.assertEqual(engine.update(FrameState(20, (person(),))), [])
        self.assertEqual(engine.update(FrameState(49, (person(),))), [])
        self.assertEqual(len(engine.update(FrameState(50, (person(),)))), 1)
        self.assertEqual(engine.update(FrameState(100, (person(),))), [])

    def test_dwell_timer_restarts_on_absence(self):
        engine = RulesEngine([rule('unknown_dwell')], 10, START)
        engine.update(FrameState(0, (person(),)))
        engine.update(FrameState(15))
        engine.update(FrameState(16, (person(),)))
        self.assertEqual(engine.update(FrameState(35, (person(),))), [])
        self.assertEqual(len(engine.update(FrameState(36, (person(),)))), 1)

    def test_activity_inclusive_confidence_label_zone_and_expiry(self):
        engine = RulesEngine([rule('activity', zone_name='Desk')], 10, START)
        tracks = (person(label='running',confidence=.8), person(5,label='running',confidence=.79),
                  person(6,label='walking',confidence=1), person(7,zones=(),label='running',confidence=1))
        alerts = engine.update(FrameState(0, tracks))
        self.assertEqual([a.track_id for a in alerts], [4])
        self.assertEqual(engine.update(FrameState(1, (person(),))), [])
        self.assertEqual(engine.active, {})

    def test_overnight_schedule_and_start_day(self):
        engine = RulesEngine([rule('zone_schedule', days=[0])], 1, START)
        self.assertEqual(engine.update(FrameState(0, (person(),))), [])
        alert = engine.update(FrameState(1, (person(),)))[0]
        self.assertEqual(alert.triggered_at, datetime(2026,9,28,22))
        self.assertEqual(engine.update(FrameState(7201, (person(),))), []) # Tuesday midnight, same episode.
        self.assertEqual(engine.update(FrameState(28801, (person(),))), []) # Tuesday 06:00 excluded.
        self.assertEqual(engine.active, {})
        self.assertEqual(engine.update(FrameState(86401, (person(),))), []) # Tuesday night excluded.
        at_midnight = RulesEngine([rule('zone_schedule', days=[0])], 1, datetime(2026,9,29,1))
        self.assertEqual(len(at_midnight.update(FrameState(0, (person(),)))), 1)

    def test_daytime_schedule_and_zone(self):
        engine = RulesEngine([rule('zone_schedule', start='09:00', end='17:00')], 1, datetime(2026,9,28,9))
        self.assertEqual([a.track_id for a in engine.update(FrameState(0, (person(), person(5,zones=()))))], [4])
        self.assertEqual(engine.update(FrameState(8*3600, (person(),))), [])
        self.assertEqual(engine.active, {})

    def test_ephemeral_ids_never_create_timers_or_alerts(self):
        for kind in ('unknown_dwell','occupancy','activity','zone_schedule'):
            configured = rule(kind)
            if kind == 'occupancy':
                configured['params_json'].update(max_people=0, duration_seconds=0)
            engine = RulesEngine([configured], 1, datetime(2026,9,28,23))
            self.assertEqual(engine.update(FrameState(0,(person(-1,label='running',confidence=1),))), [])
            self.assertEqual(engine.update(FrameState(20,(person(-2,label='running',confidence=1),))), [])
            self.assertEqual(engine.active, {})

    def test_fps_and_frame_skip_preserve_video_seconds(self):
        for fps in (10,30,60):
            for skip in (1,2,5):
                for kind in ('unknown_dwell','occupancy'):
                    with self.subTest(fps=fps, skip=skip, kind=kind):
                        engine = RulesEngine([rule(kind)], fps, START)
                        observed = []
                        for frame in range(0, 3*fps+1, skip):
                            observed.extend(engine.update(FrameState(frame,(person(),person(5)))))
                        self.assertTrue(observed)
                        self.assertTrue(all(a.video_offset_s == 2.0 for a in observed))

    def test_rule_resolution_and_disabled_rules(self):
        source = rule('occupancy')
        self.assertEqual(resolve_rules([source], [])[0], [])
        self.assertEqual(len(resolve_rules([source], [])[1]), 1)
        bound, warnings = resolve_rules([source], [dict(id='zone-new',name='Desk')])
        self.assertEqual(bound[0]['zone_id'], 'zone-new')
        self.assertIsNone(source['zone_id'])
        self.assertEqual(warnings, [])
        source['enabled'] = False
        self.assertEqual(resolve_rules([source], []), ([], []))

    def test_invalid_parameters(self):
        for configured in [rule('occupancy',max_people=-1), rule('unknown_dwell',duration_seconds=-1),
                           rule('activity',min_confidence=1.1), rule('zone_schedule',start='06:00'),
                           rule('zone_schedule',days=[7]), rule('occupancy',zone_name=None)]:
            with self.assertRaises(ValueError):
                validate_rule(configured)


class AlertRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.url = 'sqlite:///' + str(Path(self.tmp.name)/'alerts.db')
        self.repo = Repository(self.url)
        self.addCleanup(self.repo.close)

    def test_rule_crud_and_alert_readback_acknowledgement_history(self):
        args = dict(name='Dwell', type='unknown_dwell', params_json={'duration_seconds':2},
                    severity='high', enabled=True, cooldown_seconds=3)
        rule_id = self.repo.save_rule(**args)
        self.assertEqual(len(self.repo.get_rules(enabled_only=True)), 1)
        run = self.repo.start_run('test.mp4', metadata={'recording_start_time': START.isoformat()})
        person_id = self.repo.upsert_person(run,4)
        engine = RulesEngine(self.repo.get_rules(), 10, START)
        engine.update(FrameState(0,(person(),)))
        alert = engine.update(FrameState(20,(person(),)))[0]
        self.repo.log_alert(run,alert,person_id)
        self.repo.flush_frame(20)  # Alert-only frames commit too.
        other_run = self.repo.start_run('other.mp4')
        with Repository(self.url) as reader:
            rows = reader.get_alerts(run,'high')
            self.assertEqual(len(rows),1)
            saved = rows[0]
            self.assertEqual((saved['rule_id'],saved['person_id'],saved['frame_no']), (rule_id,person_id,20))
            self.assertEqual(saved['triggered_at'], datetime(2026,9,28,22,0,1))
            self.assertEqual(saved['video_offset_s'],2)
            self.assertEqual(saved['message'],alert.message)
            self.assertEqual(saved['details_json'],alert.details_json)
            self.assertFalse(saved['acknowledged'])
            self.assertEqual(reader.get_alerts(other_run),[])
            self.assertEqual(reader.get_alerts(run,'low'),[])
        self.repo.acknowledge_alert(saved['id'])
        args.update(name='Edited', severity='low', enabled=False)
        self.repo.save_rule(**args, rule_id=rule_id)
        self.assertEqual(self.repo.get_rules(enabled_only=True),[])
        self.assertEqual(self.repo.get_alerts()[0]['severity'],'high')
        self.repo.delete_rule(rule_id)
        self.assertEqual(self.repo.get_rules(),[])
        row = self.repo.get_alerts()[0]
        self.assertIsNone(row['rule_id'])
        self.assertEqual(row['details_json']['rule_name'],'Dwell')
        self.assertTrue(row['acknowledged'])
        docs = LogAnalyzer(run_id=run,database_url=self.url).get_all_logs_formatted()
        self.assertEqual(len(docs),1)
        self.assertIn("high severity alert from rule 'Dwell'", docs[0])

    def test_alerts_merge_without_changing_event_and_activity_text(self):
        run = self.repo.start_run('test.mp4')
        self.repo.log_event(run,'Access','AUTHORIZED','Alice','Desk',timestamp='2026-09-27 10:00:00')
        self.repo.log_activity(run,4,'sitting','At desk',.9,'2026-09-27 10:00:02')
        self.repo.flush()
        analyzer = LogAnalyzer(run_id=run,database_url=self.url)
        before = analyzer.get_all_logs_formatted()
        rule_id = self.repo.save_rule('Test','unknown_dwell',dict(duration_seconds=0))
        engine = RulesEngine(self.repo.get_rules(),1,datetime(2026,9,27,10))
        alert = engine.update(FrameState(1,(person(),)))[0]
        self.repo.log_alert(run,alert,self.repo.get_people(run)[0]['id'])
        self.repo.flush()
        after = analyzer.get_all_logs_formatted()
        self.assertEqual([after[0],after[2]],before)
        self.assertIn('At 2026-09-27 10:00:01',after[1])

    def test_alert_only_frame_storage_failure_is_fatal(self):
        run = self.repo.start_run('test.mp4')
        self.repo.save_rule('Dwell','unknown_dwell',dict(duration_seconds=0))
        engine = RulesEngine(self.repo.get_rules(),1,START)
        alert = engine.update(FrameState(1,(person(),)))[0]
        self.repo.log_alert(run,alert)
        connection = self.repo.session.connection().connection.driver_connection
        connection.execute('PRAGMA query_only=ON')
        try:
            with self.assertRaises(StorageFatalError):
                self.repo.flush_frame(1)
        finally:
            connection.execute('PRAGMA query_only=OFF')
        self.assertEqual(self.repo.get_alerts(),[])


if __name__ == '__main__':
    unittest.main()
