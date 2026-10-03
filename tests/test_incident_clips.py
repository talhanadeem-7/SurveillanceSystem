"""Incident provenance, no-API citations, request gating, and real local H.264 cuts."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from langchain_core.documents import Document

import config
from agents.footage import footage_intent, resolve_footage
from agents.reasoning_agent import SecurityAnalyst
from alerts.rules_engine import Alert
from storage.clips import clip_window, get_clip, unavailable_reason
from storage.repository import Repository
from utils.log_analyzer import LogAnalyzer


def document(number=1):
    return Document(page_content=f'At 2026-10-03 10:32:00, Person_{number} was running at Desk.',
                    metadata=dict(source_type='event', source_id=number, run_id='run', frame_no=101,
                                  time='2026-10-03 10:32:00', person=f'Person_{number}',
                                  action='Access', location='Desk'))


class CitationIntentTest(unittest.TestCase):
    def test_only_model_selected_retrieved_sources_capped_and_validated(self):
        analyst = SecurityAnalyst.__new__(SecurityAnalyst)
        docs = [document(i) for i in range(1, 16)]
        analyst.retriever = MagicMock()
        analyst.retriever.query_relevant_logs.return_value = docs
        analyst.chain = MagicMock()
        analyst.chain.invoke.return_value.content = json.dumps(dict(answer_text='Observed running.',
            cited_sources=[15, 15, 0, 99, '2', True, 3, 4, 5, 6, 7]))
        answer, citations = analyst.consult('Who was running?')
        self.assertIn('Observed running.', answer)
        self.assertEqual(citations, [docs[i-1] for i in (15, 3, 4, 5, 6)])
        self.assertIn('Record 15:', analyst.chain.invoke.call_args.args[0]['context'])
        analyst.chain.invoke.return_value.content = 'Hello!'
        self.assertEqual(analyst.consult('hello'), ('Hello!', []))
        analyst.chain.invoke.return_value.content = '{"answer_text":"Hello!","cited_sources":[]}'
        self.assertEqual(analyst.consult('hello'), ('Hello!', []))

    def test_intent_positive_and_negative(self):
        for text in ('show me the footage', 'show me that', 'play it', 'show the last one', 'Show clip of Person_4',
                     'Please show me footage of Person_4 running', 'video of Person_4',
                     'Can you play the video of Person_4?', 'clip of running', 'footage of Desk'):
            with self.subTest(text=text):
                self.assertTrue(footage_intent(text))
        for text in ('What was Person_4 doing at 10:32?', 'Was anyone running?',
                     'What does the video show?', 'Describe the footage of Person_4',
                     'Show me that Python code', 'Show me that report',
                     'Show me the events', 'Is there a clip of running?',
                     "Don't show me video", 'Explain it without video', 'How do I play a clip?'):
            with self.subTest(text=text):
                self.assertFalse(footage_intent(text))

    def test_references_empty_ambiguous_and_new_retrieval(self):
        retriever = MagicMock()
        one, two = document(4), document(40)
        self.assertEqual(resolve_footage('show me that', [one], retriever), (one, None))
        self.assertEqual(resolve_footage('show me the footage', [one], retriever), (one, None))
        activity = document(9)
        activity.metadata['source_type'] = 'activity'
        identity = document(10)
        identity.metadata['action'] = 'Identity'
        self.assertEqual(resolve_footage('show me the footage', [activity, one, identity], retriever), (one, None))
        self.assertIsNone(resolve_footage('show me the footage', [one, two], retriever)[0])
        self.assertEqual(resolve_footage('show video of that', [one], retriever), (one, None))
        self.assertEqual(resolve_footage('play the last one', [one, two], retriever), (two, None))
        self.assertIsNone(resolve_footage('play it', [], retriever)[0])
        self.assertIsNone(resolve_footage('play it', [one, two], retriever)[0])
        retriever.query_relevant_logs.assert_not_called()
        retriever.query_relevant_logs.return_value = [two, one]
        self.assertEqual(resolve_footage('show clip of Person_4 running at 10:32', [], retriever), (one, None))
        self.assertIsNone(resolve_footage('video of running', [], retriever)[0])
        self.assertIsNone(resolve_footage('video of Person_9', [], retriever)[0])
        retriever.query_relevant_logs.return_value = []
        self.assertIsNone(resolve_footage('video of Person_4', [], retriever)[0])

    def test_access_removal_names_and_abbreviations(self):
        retriever = MagicMock()
        for action, phrase in [('Access','auth acess'), ('Intrusion','unauth access'),
                               ('Removal','auth removal'), ('Theft','unauth removal')]:
            doc = document(4)
            doc.metadata['action'] = action
            retriever.query_relevant_logs.return_value = [doc]
            self.assertEqual(resolve_footage('show footage of ' + phrase, [], retriever), (doc,None))


class IncidentStorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, value in [('DATABASE_URL', 'sqlite:///' + str(self.root/'test.db')),
                            ('CLIP_CACHE_DIR', str(self.root/'clips'))]:
            p = patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.repo = Repository()
        self.addCleanup(self.repo.close)

    def test_metadata_all_types_first_activity_observation_and_text_parity(self):
        run = self.repo.start_run('test.mp4')
        self.repo.log_event(run, 'Access', 'AUTHORIZED', 'Person_4', 'Desk',
                            timestamp='2026-10-03 10:32:01', frame_no=11)
        for second, frame in [(2, 21), (3, 31)]:
            self.repo.log_activity(run, 4, 'running', 'At desk', .9,
                                   f'2026-10-03 10:32:0{second}', frame_no=frame)
        alert = Alert('deleted', 'high', 4, None, datetime(2026,10,3,10,32,4), .4, 41,
                      'Person_4 running', dict(rule_name='Running', track_id=4, zone_name='Desk'))
        self.repo.log_alert(run, alert)
        self.repo.flush()
        analyzer = LogAnalyzer()
        records = analyzer.get_narration_records()
        expected = [
            'At 2026-10-03 10:32:01, Person_4 accessed the Desk.',
            'From 2026-10-03 10:32:02 to 2026-10-03 10:32:03, Person_4 was running (confidence 0.90): At desk.',
            "At 2026-10-03 10:32:04, a high severity alert from rule 'Running' was triggered: Person_4 running."]
        self.assertEqual([text.encode() for text, _ in records], [text.encode() for text in expected])
        self.assertEqual(analyzer.get_all_logs_formatted(), expected)
        for (_, m), kind, frame in zip(records, ['event','activity','alert'], [11,21,41]):
            self.assertEqual((m['source_type'],m['run_id'],m['frame_no']), (kind,run,frame))
            source = self.repo.get_clip_source(kind,m['source_id'])
            self.assertEqual(source['frame_no'], frame)
        from agents.retriever import LogRetriever
        with patch('agents.retriever.OpenAIEmbeddings'), patch('agents.retriever.Chroma') as chroma, \
             patch.object(config,'VECTOR_DB_PATH',str(self.root/'vectors')):
            LogRetriever().ingest_logs()
            docs = chroma.from_documents.call_args.kwargs['documents']
            self.assertEqual([(d.page_content,d.metadata) for d in docs], records)

    def test_windows_native_fps_no_frame_skip_and_legacy(self):
        self.assertEqual(clip_window(1, 30, 600, 5, 10), (0, 10))
        self.assertEqual(clip_window(301, 30, 600, 5, 10), (5, 20))
        self.assertEqual(clip_window(600, 30, 600, 5, 10), (599/30-5, 20))
        with patch.object(config, 'FRAME_SKIP', 99):
            self.assertEqual(clip_window(301, 30, 600, 5, 10), (5, 20))
        for frame, fps in [(0,30), (601,30), (5,0), (5,float('nan'))]:
            with self.assertRaises(ValueError):
                clip_window(frame, fps, 600, 5, 10)
        run = self.repo.start_run('missing.mp4')
        self.repo.log_event(run,'Access','AUTHORIZED','Person_4')
        self.repo.flush()
        self.assertEqual(LogAnalyzer().get_narration_records()[0][1]['frame_no'], -1)
        self.assertIsNone(get_clip('event', self.repo.get_events()[0]['id']))
        self.assertIn('missing', unavailable_reason())
        self.assertFalse(Path(config.CLIP_CACHE_DIR).exists())

    def test_real_cut_cache_and_missing_replaced_corrupt_source(self):
        path = self.root/'source.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 10, (96,64))
        self.assertTrue(writer.isOpened())
        for i in range(200):
            frame = np.full((64,96,3), (i, 255-i, i//2), np.uint8)
            writer.write(frame)
        writer.release()
        run = self.repo.start_run(path.name,source=str(path),metadata={'source_fps':10,'frame_number_base':1})
        for frame in (1, 101, 200):
            self.repo.log_event(run,'Access','AUTHORIZED','Person_4',frame_no=frame)
        self.repo.flush()
        events = self.repo.get_events()
        with patch('storage.clips.subprocess.run', side_effect=OSError('encoder unavailable')):
            self.assertIsNone(get_clip('event',events[0]['id']))
            self.assertIn('encoder unavailable', unavailable_reason())
            self.assertEqual(list(Path(config.CLIP_CACHE_DIR).glob('*.mp4')), [])
        for event, expected_frames in zip(events, [50,80,31]):
            result = get_clip('event',event['id'])
            self.assertIsNotNone(result, unavailable_reason())
            cap = cv2.VideoCapture(result)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), expected_frames)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),10)
            ok, first = cap.read()
            self.assertTrue(ok)
            source_index = max(0, event['frame_no'] - 1 - 30)
            self.assertLess(abs(float(first[:,:,0].mean()) - source_index), 4)
            cap.release()
            modified = Path(result).stat().st_mtime_ns
            with patch('storage.clips.subprocess.run',side_effect=AssertionError('must reuse cache')):
                self.assertEqual(get_clip('event',event['id']),result)
            self.assertEqual(Path(result).stat().st_mtime_ns,modified)
        path.unlink()
        self.assertIsNone(get_clip('event',events[0]['id']))
        self.assertIn('missing', unavailable_reason())
        path.write_bytes(b'corrupt')
        self.assertIsNone(get_clip('event',events[0]['id']))
        self.assertIn('replaced', unavailable_reason())
        run2 = self.repo.start_run(path.name,source=str(path))
        self.repo.log_event(run2,'Access','AUTHORIZED','Person_4',frame_no=1)
        self.repo.flush()
        self.assertIsNone(get_clip('event',self.repo.get_events(run2)[0]['id']))
        self.assertIn('corrupt', unavailable_reason())


class IncidentUITest(unittest.TestCase):
    def test_only_latest_requested_clip_is_displayed(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('from agents.analyst_ui import analyst_chat\n'
                                 'import streamlit as st\nanalyst_chat(st.session_state.test_analyst)')
        app.session_state['test_analyst'] = MagicMock()
        app.session_state['analyst_history'] = [dict(question='show footage', answer='Footage',
            citations=[document(i)], requested_clip={'path':str(i),'reason':None}) for i in range(5)]
        with patch('agents.analyst_ui.display_clip') as display:
            app.run()
            self.assertFalse(app.exception)
            display.assert_called_once_with({'path':'4','reason':None})
            self.assertFalse(app.expander)
            self.assertFalse(app.button)

    def test_ambiguous_footage_never_cuts_or_consults_llm(self):
        from streamlit.testing.v1 import AppTest
        analyst = MagicMock()
        analyst.retriever.query_relevant_logs.return_value = [document(4),document(40)]
        app = AppTest.from_string('from agents.analyst_ui import analyst_chat\n'
                                 'import streamlit as st\nanalyst_chat(st.session_state.test_analyst)')
        app.session_state['test_analyst'] = analyst
        with patch('agents.analyst_ui.get_clip') as cut:
            app.run()
            app.chat_input[0].set_value('show video of running').run()
            self.assertFalse(app.exception)
            self.assertIn('Which access or removal event',app.markdown[-1].value)
            cut.assert_not_called()
            analyst.consult.assert_not_called()

    def test_text_only_then_explicit_footage_and_reference_no_extra_llm(self):
        from streamlit.testing.v1 import AppTest
        doc = document(4)
        analyst = MagicMock()
        analyst.consult.return_value = ('Person_4 was running.', [doc])
        app = AppTest.from_string('from agents.analyst_ui import analyst_chat\n'
                                 'import streamlit as st\nanalyst_chat(st.session_state.test_analyst)')
        app.session_state['test_analyst'] = analyst
        with patch('agents.analyst_ui.get_clip', return_value=None) as cut, \
             patch('agents.analyst_ui.unavailable_reason', return_value='Source video is missing.'):
            app.run()
            app.chat_input[0].set_value('What was Person_4 doing?').run()
            self.assertFalse(app.exception)
            cut.assert_not_called()
            self.assertEqual(app.session_state['last_answer_citations'], [doc])
            self.assertEqual(len(app.expander), 0)
            self.assertEqual(len(app.button), 0)
            app.chat_input[0].set_value('show me the footage').run()
            cut.assert_called_once_with('event',4)
            self.assertIn('Clip unavailable',app.warning[0].value)
            app.run()
            self.assertEqual(cut.call_count,1)
            app.chat_input[0].set_value('show me that').run()
            self.assertEqual(cut.call_count,2)
            analyst.consult.assert_called_once()
            analyst.retriever.query_relevant_logs.assert_not_called()
            self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
