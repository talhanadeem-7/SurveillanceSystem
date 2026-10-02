"""Exercise rule forms/history with Streamlit's local test runner."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
from alerts.rules_engine import RulesEngine, FrameState, TrackState
from storage.repository import Repository


class AlertUITest(unittest.TestCase):
    def test_forms_crud_validation_and_history_acknowledgement(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config, 'DATABASE_URL', 'sqlite:///' + str(Path(directory)/'ui.db')):
            app = AppTest.from_string(
                'from alerts.ui import alert_rules_tab, recording_start_input\n'
                'recording_start_input()\nalert_rules_tab()').run()
            def widget(kind, label):
                return next(w for w in getattr(app,kind) if w.label == label)
            widget('button','Save rule').click().run()
            self.assertTrue(app.error)
            for kind in ('zone_schedule','occupancy','unknown_dwell','activity'):
                widget('selectbox','Rule type').select(kind).run()
                widget('text_input','Rule name').set_value(kind)
                widget('text_input','Zone name (required for schedule/occupancy; blank means anywhere otherwise)').set_value('Desk')
                widget('button','Save rule').click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
            with Repository() as repository:
                rules = repository.get_rules()
                self.assertEqual(len(rules),4)
                activity_rule = next(r for r in rules if r['type']=='activity')
                selected_id = activity_rule['id']
                run = repository.start_run('UI test.mp4')
                zones = repository.save_zones(run,[dict(name='Desk',type='restricted',coords=(0,0,1,1))])
                activity_rule['zone_id'] = zones['Desk']
                pid = repository.upsert_person(run,4)
                engine = RulesEngine([activity_rule],30,datetime(2026,9,27,10))
                alert = engine.update(FrameState(30,(TrackState(4,'Person_4',False,frozenset({'Desk'}),'running',1),)))[0]
                repository.log_alert(run,alert,pid)
                repository.flush()
            app.run()
            self.assertEqual(len(app.dataframe),1)
            widget('selectbox','Filter by severity').select('low').run()
            self.assertEqual(len(app.dataframe),0)
            widget('selectbox','Filter by severity').select('medium').run()
            widget('selectbox','Filter by run').select(run).run()
            widget('button','Acknowledge alert').click().run()
            with Repository() as repository:
                self.assertTrue(repository.get_alerts()[0]['acknowledged'])
            widget('selectbox','Rule to edit').select(selected_id).run()
            widget('text_input','Rule name').set_value('Edited activity rule')
            widget('checkbox','Enabled').uncheck()
            widget('button','Save rule').click().run()
            with Repository() as repository:
                edited = next(r for r in repository.get_rules() if r['id']==selected_id)
                self.assertFalse(edited['enabled'])
                self.assertEqual(edited['name'],'Edited activity rule')
            widget('button','Delete rule').click().run()
            with Repository() as repository:
                self.assertEqual(len(repository.get_rules()),3)
                self.assertEqual(len(repository.get_alerts()),1)
            self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
