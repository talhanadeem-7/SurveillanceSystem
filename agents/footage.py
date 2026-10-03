"""Conservative deterministic footage intent and source selection (no LLM)."""
import re

_REFERENCE = re.compile(r'\b(?:that|it|the last one)\b', re.I)
_NEGATIVE = re.compile(r"\b(?:don'?t|do not|never|without|no need to)\b", re.I)
_REQUEST = re.compile(
    r'^(?:(?:please|can you|could you|would you|i want to|i would like to)\s+)*'
    r'(?:show|play|open|watch)\b|'
    r'^(?:(?:please|i want|i would like|give me|find|find me)\s+)*'
    r'(?:a\s+|the\s+)?(?:clip|footage|video)\s+(?:of|for|from)\b', re.I)
_MEDIA = re.compile(r'\b(?:clip|footage|video)\b', re.I)
_EVENT_LABELS = {'Access': 'authorized access', 'Intrusion': 'unauthorized access',
                 'Removal': 'authorized removal', 'Theft': 'unauthorized removal'}


def security_sources(docs):
    unique = {}
    for doc in docs:
        m = doc.metadata
        if m.get('source_type') == 'event' and m.get('action') in _EVENT_LABELS:
            unique[(m.get('run_id'), m.get('source_id'))] = doc
    return list(unique.values())


def _clarify(docs):
    message = 'Which access or removal event do you mean? Please ask for footage with the person and event time.'
    if docs:
        message += '\n\n' + '\n\n'.join(
            f"{d.metadata.get('time', 'Unknown time')} — {d.metadata.get('person', 'Unknown person')}: "
            f"{_EVENT_LABELS[d.metadata['action']]} at {d.metadata.get('location', 'unknown location')}"
            for d in docs[:5])
    return message


def _reference_only(query):
    remaining = re.sub(r'\b(?:please|can|could|would|you|show|play|open|watch|me|a|the|clip|footage|video|of|for|from|that|it|last|one)\b', '', query, flags=re.I)
    return bool(_REFERENCE.search(query)) and not re.search(r'\w', remaining)


def footage_intent(query):
    query = query.strip()
    if _NEGATIVE.search(query) or not _REQUEST.search(query):
        return False
    return bool(_MEDIA.search(query) or _reference_only(query) or
                re.match(r'^(?:please\s+)?play\b', query, re.I))


def resolve_footage(query, previous, retriever):
    """Select a unique exact lexical match among retrieved incidents, otherwise clarify."""
    previous = security_sources(previous)
    clarify = _clarify(previous)
    if _reference_only(query):
        if re.search(r'\bthe last one\b', query, re.I):
            return (previous[-1], None) if previous else (None, clarify)
        return (previous[0], None) if len(previous) == 1 else (None, clarify)
    words = re.findall(r'[\w:]+', query.lower())
    stop = set('please can could would you i want like to show play open watch me a the clip footage video of for from at in on incident'.split())
    terms = set(words) - stop
    if not terms:
        return (previous[0], None) if len(previous) == 1 else (None, clarify)
    normalized = ' '.join(words)
    normalized = re.sub(r'\bunauth\b', 'unauthorized', normalized)
    normalized = re.sub(r'\bauth\b', 'authorized', normalized)
    normalized = re.sub(r'\bacess\b', 'access', normalized)
    terms = set(normalized.split()) - stop
    docs = security_sources(retriever.query_relevant_logs(query, k=15))
    matches = {}
    for doc in docs:
        metadata = doc.metadata
        haystack = set(re.findall(r'[\w:]+', (doc.page_content + ' ' + ' '.join(map(str, metadata.values()))
                                           + ' ' + _EVENT_LABELS[metadata['action']]).lower()))
        haystack.update(token[:5] for token in list(haystack) if re.fullmatch(r'\d{2}:\d{2}:\d{2}', token))
        # Exact token checks prevent Person_4 matching Person_40 and reject unrelated nearest neighbors.
        if terms <= haystack:
            key = (metadata.get('source_type'), metadata.get('source_id'), metadata.get('run_id'))
            matches[key] = doc
    if len(matches) == 1:
        return next(iter(matches.values())), None
    return None, _clarify(list(matches.values()))
