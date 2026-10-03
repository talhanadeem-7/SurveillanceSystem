import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from agents.conversation import route_footage, ordered
from agents.reasoning_agent import SecurityAnalyst


def doc(number, run='run'):
    return Document(page_content=f'Talha accessed Desk at 04:17:{number:02d}.',metadata=dict(
        source_type='event',source_id=number,run_id=run,frame_no=number*25,
        time=f'2026-10-04 04:17:{number:02d}',person='Talha',action='Access',location='Desk'))


class ConversationTest(unittest.TestCase):
    def setUp(self):
        self.analyst=MagicMock()
        self.analyst.retriever.query_relevant_logs.return_value=[]
        self.analyst.classify_footage.return_value={'intent':'clarify','candidates':[]}
        self.docs=[doc(2),doc(1)]

    def test_request_then_earlier_and_explanation(self):
        handled,selected,pending,reply=route_footage('Can I see what happened?',self.docs,[],self.analyst)
        self.assertTrue(handled); self.assertIsNone(selected)
        self.assertEqual(pending,ordered(self.docs))
        self.assertIn('1.',reply)
        self.assertEqual(route_footage('the earlier one',self.docs,pending,self.analyst)[1],doc(1))
        self.assertFalse(route_footage('Why was he marked unauthorized?',self.docs,pending,self.analyst)[0])
        self.analyst.classify_footage.assert_not_called()

    def test_cancellation_invalid_selection_and_cross_run(self):
        self.assertEqual(route_footage('cancel',[],self.docs,self.analyst)[2],[])
        self.assertIsNone(route_footage('99',[],self.docs,self.analyst)[1])
        self.assertIsNone(route_footage('earlier one',[],[doc(1,'a'),doc(2,'b')],self.analyst)[1])
        self.assertIsNone(route_footage('the earlier one',self.docs,[],self.analyst)[1])

    def test_field_selection_and_unknown_person_not_locally_guessed(self):
        self.assertEqual(route_footage('show footage of Talha at 4:17:01 am',self.docs,[],self.analyst)[1],doc(1))
        self.assertIsNone(route_footage('show footage of access by Bob',[doc(1)],[],self.analyst)[1])
        self.analyst.classify_footage.assert_called_once()

    def test_model_selection_validated_and_negative_not_sent(self):
        self.analyst.classify_footage.return_value={'intent':'footage','candidates':[99]}
        self.assertIsNone(route_footage('pull it up',self.docs,[],self.analyst)[1])
        self.analyst.classify_footage.return_value={'intent':'footage','candidates':[1,2]}
        self.assertIsNone(route_footage('pull it up',self.docs,[],self.analyst)[1])
        self.analyst.classify_footage.return_value={'intent':'footage','candidates':[1]}
        self.assertEqual(route_footage('pull it up',self.docs,[],self.analyst)[1],doc(1))
        self.assertFalse(route_footage("don't show video",self.docs,[],self.analyst)[0])

    def test_fallback_api_failure_fails_closed(self):
        analyst=SecurityAnalyst.__new__(SecurityAnalyst)
        analyst.llm=MagicMock(); analyst.llm.invoke.side_effect=RuntimeError('offline')
        self.assertEqual(analyst.classify_footage('pull it up',self.docs)['intent'],'clarify')
        analyst.llm.invoke.side_effect=None
        analyst.llm.invoke.return_value.content='not JSON'
        self.assertEqual(analyst.classify_footage('pull it up',self.docs)['intent'],'clarify')

    def test_structured_answer_validates_events_and_orders_them(self):
        result=dict(answer_text='Two accesses.',events=[{'source':1,'text':'Later access.'},
            {'source':99,'text':'Invented event.'},{'source':2,'text':'Earlier access.'}],identity_note='Same recognized track.')
        text=SecurityAnalyst.format_answer(result,self.docs,self.docs)
        self.assertLess(text.index('Earlier access'),text.index('Later access'))
        self.assertNotIn('Invented',text)
        for heading in ('**Summary**','**Events**','**Identity note**'): self.assertIn(heading,text)

    def test_ui_end_to_end_pending_choice_cuts_once(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_string('import streamlit as st\nfrom agents.analyst_ui import analyst_chat\nanalyst_chat(st.session_state.agent)')
        self.analyst.consult.return_value=('Two accesses.',self.docs)
        app.session_state['agent']=self.analyst
        with patch('agents.analyst_ui.request_clip',return_value={'path':None,'reason':'test'}) as cut:
            app.run()
            app.chat_input[0].set_value('Did anything happen today?').run()
            app.chat_input[0].set_value('Can I see what happened?').run()
            cut.assert_not_called()
            self.assertEqual(len(app.session_state['pending_footage']),2)
            app.chat_input[0].set_value('The earlier one').run()
            cut.assert_called_once_with(doc(1))
            app.chat_input[0].set_value('Why was he marked unauthorized?').run()
            self.assertEqual(cut.call_count,1)
            self.assertFalse(app.exception)
            self.assertFalse(app.get('video'))


if __name__=='__main__': unittest.main()
