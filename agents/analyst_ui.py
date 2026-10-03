"""Streamlit Analyst conversation and strictly on-demand incident media."""
from pathlib import Path
import streamlit as st

from agents.conversation import route_footage
from storage.clips import get_clip, unavailable_reason


def incident_label(doc):
    m = doc.metadata
    return ' | '.join(str(m.get(key, 'Not recorded')) for key in ('time', 'person', 'action', 'location'))


def request_clip(doc):
    path = get_clip(doc.metadata.get('source_type'), doc.metadata.get('source_id'))
    return {'path': path, 'reason': None if path else unavailable_reason()}


def display_clip(result):
    try:
        if result['path'] and Path(result['path']).is_file():
            # Streamlit defaults to paused. Never opt into autoplay.
            st.video(result['path'], format='video/mp4', autoplay=False)
        else:
            st.warning('Clip unavailable: ' + (result['reason'] or 'Cached clip was deleted; request it again.'))
    except Exception as exc:
        st.warning(f'Clip unavailable: {exc}')


def analyst_chat(analyst):
    state = st.session_state
    state.setdefault('analyst_history', [])
    state.setdefault('last_answer_citations', [])
    state.setdefault('pending_footage', [])
    prompt = st.chat_input('Ask Shelby about the footage...')
    if prompt:
        entry = dict(question=prompt, answer='', citations=[], clips={})
        try:
            handled, doc, pending, clarification = route_footage(
                prompt, state.last_answer_citations, state.pending_footage, analyst)
            state.pending_footage = pending
            if handled:
                if doc is None:
                    entry['answer'] = clarification
                else:
                    entry['answer'] = 'Footage for: ' + incident_label(doc)
                    entry['citations'] = [doc]
                    entry['requested_clip'] = request_clip(doc)
                    state.last_answer_citations = [doc]
            else:
                state.last_answer_citations = []
                entry['answer'], entry['citations'] = analyst.consult(prompt)
                entry['citations'] = entry['citations'][:5]
                state.last_answer_citations = entry['citations']
        except Exception as exc:
            entry['answer'] = f'Unable to answer: {exc}'
        state.analyst_history.append(entry)

    for index, entry in enumerate(state.analyst_history):
        with st.chat_message('user'):
            st.markdown(entry['question'])
        with st.chat_message('assistant'):
            st.markdown(entry['answer'])
            if index == len(state.analyst_history) - 1 and 'requested_clip' in entry:
                display_clip(entry['requested_clip'])
