"""Local database and mocked RAG checks; never invoke paid APIs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
from storage.repository import Repository
from utils.log_analyzer import LogAnalyzer


class ActivityNarrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.url = 'sqlite:///' + str(self.root / 'logs.db')
        self.repo = Repository(self.url)
        self.addCleanup(self.repo.close)
        self.run = self.repo.start_run('office.mp4')
        self.other = self.repo.start_run('other.mp4')
        floor = patch.object(config, 'VLM_ACTIVITY_MIN_LOG_CONFIDENCE', .5)
        floor.start()
        self.addCleanup(floor.stop)

    def activity(self, second, label='sitting', confidence=.9, track=4, run=None,
                 description='At desk'):
        self.repo.log_activity(run or self.run, track, label, description, confidence,
                               f'2026-09-27 10:32:{second:02d}', frame_no=second)

    def docs(self, run=None):
        self.repo.flush()
        return LogAnalyzer(run_id=run, database_url=self.url).get_all_logs_formatted()

    def test_event_only_fixture_is_byte_identical(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/event_narration.json').read_text())
        for timestamp, entity, action, status, location, _ in fixture:
            self.repo.log_event(self.run, action, status, entity, location, timestamp=timestamp)
        self.assertEqual(self.docs(), [row[-1] for row in fixture])

    def test_activity_only_names_fallback_and_exact_text(self):
        self.repo.upsert_person(self.run, 4, 'Alice', True)
        self.activity(5)
        self.activity(6, track=8, description='Near door.')
        self.assertEqual(self.docs(), [
            'At 2026-09-27 10:32:05, Alice was sitting (confidence 0.90): At desk.',
            'At 2026-09-27 10:32:06, Person_8 was sitting (confidence 0.90): Near door.',
        ])

    def test_timestamp_order_including_out_of_order_inserts(self):
        self.activity(20, label='walking')
        self.repo.log_event(self.run, 'Access', 'AUTHORIZED', 'Alice', 'Desk',
                             timestamp='2026-09-27 10:32:10')
        self.activity(5)
        self.assertEqual([d.split(',')[0] for d in self.docs()],
                         ['At 2026-09-27 10:32:05','At 2026-09-27 10:32:10',
                          'At 2026-09-27 10:32:20'])

    def test_floor_inclusive_configurable_and_rejected_row_breaks_span(self):
        self.activity(1, confidence=.5)
        self.activity(2, confidence=.49)
        self.activity(3, confidence=.8)
        self.assertEqual(len(self.docs()), 2)
        self.assertTrue(all(d.startswith('At ') for d in self.docs()))
        with patch.object(config, 'VLM_ACTIVITY_MIN_LOG_CONFIDENCE', .81):
            self.assertEqual(self.docs(), [])

    def test_collapse_per_track_preserves_label_changes_and_descriptions(self):
        self.activity(1)
        self.activity(2, track=5, label='walking')
        self.activity(3, confidence=.8, description='Still at desk.')
        self.activity(4, label='walking')
        self.activity(5)
        docs = self.docs()
        self.assertEqual(len(docs), 4)
        self.assertEqual(docs[0], 'From 2026-09-27 10:32:01 to 2026-09-27 10:32:03, '
                         'Person_4 was sitting (confidence 0.80): At desk. Still at desk.')
        self.assertIn('Person_5 was walking', docs[1])
        self.assertIn('Person_4 was walking', docs[2])
        self.assertIn('Person_4 was sitting', docs[3])

    def test_run_filter_and_no_collapse_across_runs(self):
        self.activity(1)
        self.activity(2, run=self.other)
        self.repo.log_event(self.other, 'Access', 'AUTHORIZED', 'Other', 'Desk',
                             timestamp='2026-09-27 10:32:03')
        self.assertEqual(len(self.docs()), 3)
        self.assertEqual(len(self.docs(self.run)), 1)
        self.assertEqual(len(self.docs(self.other)), 2)
        self.assertEqual(self.docs('missing'), [])
        self.assertEqual(len(self.repo.get_activities(self.run)), 1)

    def test_retriever_and_analyst_ingest_filtered_activity_without_apis(self):
        from langchain_core.runnables import RunnableLambda
        from agents.reasoning_agent import SecurityAnalyst
        self.activity(5)
        self.activity(6, run=self.other, label='walking')
        expected = self.docs(self.run)
        with patch.object(config, 'DATABASE_URL', self.url), \
             patch.object(config, 'VECTOR_DB_PATH', str(self.root/'vectors')), \
             patch('agents.retriever.OpenAIEmbeddings'), \
             patch('agents.retriever.Chroma') as chroma, \
             patch('agents.reasoning_agent.ChatOpenAI', return_value=RunnableLambda(lambda x: x)):
            analyst = SecurityAnalyst(run_id=self.run)
            documents = chroma.from_documents.call_args.kwargs['documents']
            self.assertEqual([d.page_content for d in documents], expected)
            self.assertEqual(analyst.retriever.log_analyzer.run_id, self.run)
            self.assertIn(self.run, analyst.retriever.db_path)
            self.assertIn('ACTIVITY OBSERVATIONS', analyst.prompt.template)


if __name__ == '__main__':
    unittest.main()
