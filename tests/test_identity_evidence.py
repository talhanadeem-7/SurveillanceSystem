"""Late recognition must survive retrieval without rewriting the audit trail."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from storage.repository import Repository
from utils.log_analyzer import LogAnalyzer
from agents.retriever import LogRetriever
from agents.reasoning_agent import SecurityAnalyst


class IdentityEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.url = 'sqlite:///' + str(Path(self.tmp.name)/'identity.db')
        self.repo = Repository(self.url)
        self.addCleanup(self.repo.close)
        self.run = self.repo.start_run('sample.mp4')
        self.person = self.repo.upsert_person(self.run, 1)
        self.repo.log_event(self.run, 'Intrusion', 'UNAUTHORIZED', 'Person_1',
                            person_id=self.person, frame_no=100)
        self.repo.flush()
        self.analyzer = LogAnalyzer(database_url=self.url)
        self.doc = Document(page_content=self.analyzer.get_narration_records()[0][0],
                            metadata=self.analyzer.get_narration_records()[0][1])

    def recognize(self, name='Talha', status=None):
        self.repo.upsert_person(self.run, 1, name, True)
        self.repo.log_event(self.run, 'Identity', status or 'Recognized as '+name, 'Person_1',
                            person_id=self.person, frame_no=424)
        self.repo.flush()

    def evidence(self, doc=None):
        return json.loads(self.analyzer.enrich_identity([doc or self.doc])[0].metadata['identity_evidence'])

    def test_late_recognition_refresh_preserves_text_and_original_rows(self):
        before = self.repo.get_events()[0]
        self.assertIsNone(self.evidence()['resolved_name'])
        self.recognize()
        enriched = self.analyzer.enrich_identity([self.doc])[0]
        evidence = json.loads(enriched.metadata['identity_evidence'])
        self.assertEqual(evidence['resolved_name'], 'Talha')
        self.assertEqual(evidence['identity_events'][0]['frame_no'],424)
        self.assertEqual(enriched.page_content.encode(),self.doc.page_content.encode())
        self.assertEqual(self.repo.get_events()[0], before)

    def test_run_isolation_missing_link_and_stale_evidence(self):
        self.recognize()
        other = self.repo.start_run('other.mp4')
        person = self.repo.upsert_person(other,1,'Alice',True)
        self.repo.log_event(other,'Identity','Recognized as Alice','Person_1',person_id=person)
        self.repo.log_event(other,'Intrusion','UNAUTHORIZED','Person_1',person_id=person)
        self.repo.log_event(self.run,'Theft','UNAUTHORIZED','ASSET',frame_no=200)
        self.repo.flush()
        self.assertEqual(self.evidence()['resolved_name'],'Talha')
        records=self.analyzer.get_narration_records()
        unlinked=next(Document(page_content=t,metadata=m) for t,m in records if m['action']=='Theft')
        unlinked.metadata['identity_evidence']='stale name'
        self.assertNotIn('identity_evidence',self.analyzer.enrich_identity([unlinked])[0].metadata)
        tampered=Document(page_content=self.doc.page_content,metadata=dict(self.doc.metadata,run_id=other))
        self.assertNotIn('identity_evidence',self.analyzer.enrich_identity([tampered])[0].metadata)

    def test_revocation_or_conflicting_names_prevent_retroactive_mapping(self):
        self.recognize()
        self.recognize(status='Revoked Talha (mismatch)')
        self.assertEqual(self.evidence()['identity_state'],'conflicting_or_revoked')
        self.assertIsNone(self.evidence()['resolved_name'])
        self.recognize('Alice')
        self.assertIsNone(self.evidence()['resolved_name'])
        self.assertEqual(len(self.evidence()['identity_events']),3)

    def test_retrieval_always_carries_evidence_even_without_identity_hit(self):
        self.recognize()
        retriever=LogRetriever.__new__(LogRetriever)
        retriever.vector_store=MagicMock()
        retriever.vector_store.similarity_search.return_value=[self.doc]
        retriever.log_analyzer=self.analyzer
        analyst=SecurityAnalyst.__new__(SecurityAnalyst)
        analyst.retriever=retriever
        analyst.chain=MagicMock()
        analyst.chain.invoke.return_value.content='{"answer_text":"mocked","cited_sources":[1]}'
        answer,citations=analyst.consult('Any unauthorized accesses?')
        context=analyst.chain.invoke.call_args.args[0]['context']
        self.assertIn('Talha',context)
        self.assertIn('424',context)
        self.assertIn('UNAUTHORIZED',self.repo.get_events()[0]['status'])
        self.assertEqual(len(citations),1)
        self.assertEqual(citations[0].metadata['source_id'],self.doc.metadata['source_id'])
        self.assertEqual(citations[0].page_content,self.doc.page_content)


if __name__ == '__main__':
    unittest.main()
