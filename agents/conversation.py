"""Request/selection state for on-demand footage. Candidates always come from stored documents."""
import json
import re
from datetime import datetime

from agents.footage import security_sources, footage_intent, _EVENT_LABELS


def cancelled(text):
    return bool(re.search(r"\b(cancel|never mind|nevermind|don't|do not|no thanks)\b", text, re.I))


def is_request(text):
    return footage_intent(text) or bool(re.search(
        r'\b(?:can|could|may) i (?:see|watch)\b|\b(?:show|play|watch|see) (?:me )?(?:what happened|that incident)\b', text, re.I))


def selection_reply(text):
    return bool(re.fullmatch(r'\s*(?:(?:please|show|play|me|the|option|event|clip)\s+)*'
                            r'(?:\d+|first|second|third|fourth|fifth|earlier|earliest|later|latest|last)'
                            r'(?:\s+one)?[.!?]?\s*',text,re.I))


def ordered(docs):
    return sorted(security_sources(docs),key=lambda d:(str(d.metadata.get('run_id','')),
        int(d.metadata.get('frame_no',-1)),str(d.metadata.get('time',''))))


def choose(text, docs):
    docs = ordered(docs)
    if not selection_reply(text):
        return None
    words = re.findall(r'\w+', text.lower())
    positions = dict(first=0,second=1,third=2,fourth=3,fifth=4)
    for word in words:
        if word.isdigit() or word in positions:
            index = int(word)-1 if word.isdigit() else positions[word]
            return docs[index] if 0 <= index < len(docs) else None
    # Relative time is meaningful only within a single run.
    if len({d.metadata.get('run_id') for d in docs}) == 1 and docs:
        return docs[0] if {'earlier','earliest'} & set(words) else docs[-1]
    return None


def choices_text(docs):
    if not docs:
        return 'Which access or removal event do you want to see? Please give the person, time or location.'
    lines = ['Which access or removal event would you like to see? Reply with its number or “the earlier one”.']
    for i,d in enumerate(ordered(docs),1):
        m=d.metadata
        lines.append(f"{i}. {m.get('time','Unknown time')} — {m.get('person','Unknown person')}, "
                     f"{_EVENT_LABELS[m['action']]} at {m.get('location','unknown location')} (run {str(m.get('run_id',''))[:8]}).")
    return '\n\n'.join(lines)


def match_details(text, docs):
    """Match meaningful fields rather than requiring every conversational word in a record."""
    docs = ordered(docs)
    lower = re.sub(r'\bunauth\b','unauthorized',text.lower())
    lower = re.sub(r'\bauth\b','authorized',lower).replace('acess','access')
    constraints = []
    if re.search(r'\b(unauthorized|intrusion|flagged)\b',lower):
        constraints.append(lambda m:m['action'] in ('Intrusion','Theft'))
    elif re.search(r'\bauthorized\b',lower):
        constraints.append(lambda m:m['action'] in ('Access','Removal'))
    if re.search(r'\b(removal|removed|theft|took)\b',lower):
        constraints.append(lambda m:m['action'] in ('Removal','Theft'))
    elif re.search(r'\b(access|accessed|entered|intrusion)\b',lower):
        constraints.append(lambda m:m['action'] in ('Access','Intrusion'))
    person = re.search(r'\bperson_\d+\b',lower)
    if person:
        constraints.append(lambda m:person.group() == str(m.get('person','')).lower())
    stamp = re.search(r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b',lower)
    if stamp:
        hour=int(stamp[1]); hour=(hour%12+(12 if stamp[4]=='pm' else 0)) if stamp[4] else hour
        def matches_time(m):
            try:
                instant=datetime.fromisoformat(m['time'])
                return instant.hour==hour and instant.minute==int(stamp[2]) and (stamp[3] is None or instant.second==int(stamp[3]))
            except (ValueError,KeyError):
                return False
        constraints.append(matches_time)
    date = re.search(r'\b\d{4}-\d{2}-\d{2}\b',lower)
    if date:
        constraints.append(lambda m:str(m.get('time','')).startswith(date.group()))
    # Restrict known names/locations when explicitly mentioned, including resolved identity aliases.
    for field in ('person','location','run_id'):
        values={str(d.metadata.get(field,'')) for d in docs}
        if field=='person':
            for d in docs:
                try:
                    name=json.loads(d.metadata.get('identity_evidence','{}')).get('resolved_name')
                    if name: values.add(name)
                except (ValueError,TypeError): pass
        mentioned={v for v in values if v and re.search(r'(?<!\w)'+re.escape(v.lower())+r'(?!\w)',lower)}
        if mentioned:
            def has_value(m,field=field,mentioned=mentioned):
                if m.get(field) in mentioned: return True
                if field=='person':
                    try: return json.loads(m.get('identity_evidence','{}')).get('resolved_name') in mentioned
                    except (ValueError,TypeError): pass
                return False
            constraints.append(has_value)
    return [d for d in docs if all(check(d.metadata) for check in constraints)], bool(constraints)


def route_footage(text, previous, pending, analyst):
    """Return (handled, selected document, pending candidates, reply). Never cuts media."""
    if cancelled(text):
        return (True,None,[], 'Footage request cancelled.') if pending else (False,None,[],None)
    if pending and selection_reply(text):
        doc=choose(text,pending)
        return True,doc,([] if doc else pending),None if doc else choices_text(pending)
    explicit=is_request(text)
    # Explanation questions cannot consume a pending footage selection.
    if re.match(r'\s*(why|what|who|when|where|how|describe|explain|was|were|did|is|are)\b',text,re.I):
        return False,None,[],None
    if not explicit and not pending and not re.search(r'\b(see|watch|look|pull|bring|replay|footage|clip|video)\b',text,re.I):
        return False,None,[],None
    pool=ordered(pending or previous)
    matches, constrained=match_details(text,pool)
    if explicit:
        # Explicitly named details may refer to an event beyond the previous answer.
        if constrained:
            pool=ordered(pool + analyst.retriever.query_relevant_logs(text,k=15))
            matches,_=match_details(text,pool)
        elif not pool:
            pool=ordered(analyst.retriever.query_relevant_logs(text,k=15))
            matches=pool
        generic=bool(re.fullmatch(r'\s*(?:please |can you |could you )?(?:show|play)(?: me)? (?:the )?(?:footage|video|clip|that|it)[.!?]?\s*',text,re.I)
                     or re.search(r'\b(?:see|watch) what happened\b',text,re.I))
        known=' '.join(d.page_content+' '+json.dumps(d.metadata)+' '+_EVENT_LABELS[d.metadata['action']] for d in pool).lower()
        filler=set('please can could would i you me like want to see show play watch footage clip video of for from the a an at in on happened what by that it earlier later access unauthorized authorized removal flagged unauth auth acess'.split())
        unexplained=[w for w in re.findall(r'\w+',text.lower()) if not w.isdigit() and w not in ('am','pm')
                     and w not in filler and w not in re.findall(r'\w+',known)]
        if (constrained and not unexplained) or generic or selection_reply(text):
            doc=choose(text,matches) or (matches[0] if len(matches)==1 else None)
            return True,doc,([] if doc else matches[:5]),None if doc else choices_text(matches[:5])
    # Unclear wording: the model can select only numbered stored candidates, never invent an ID.
    if not pool:
        pool=ordered(analyst.retriever.query_relevant_logs(text,k=15))
    decision=analyst.classify_footage(text,pool,pending=bool(pending))
    if decision.get('intent')=='answer': return False,None,[],None
    indices=decision.get('candidates',[])
    valid=isinstance(indices,list) and all(type(i) is int and 1<=i<=len(pool) for i in indices)
    selected=[pool[i-1] for i in dict.fromkeys(indices)] if valid else []
    selected_matches,_=match_details(text,selected)
    if decision.get('intent')=='footage' and len(selected)==1 and len(selected_matches)==1:
        return True,selected[0],[],None
    options=selected or pool[:5]
    return True,None,options,choices_text(options)
