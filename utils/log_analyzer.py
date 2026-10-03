"""Database-backed event narration and statistics for the analyst."""

import pandas as pd
import json
import config
from storage.repository import Repository

class LogAnalyzer:
    def __init__(self, run_id=None, database_url=None):
        self.run_id = run_id
        self.database_url = database_url

    def load_data(self):
        """Expose the historical five-column view without reading legacy CSVs."""
        columns = ["Timestamp", "Entity", "Action", "Status", "Location"]
        with Repository(self.database_url) as repository:
            events = repository.get_events(run_id=self.run_id)
        rows = [{"Timestamp": e["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                 "Entity": e["details_json"]["entity"], "Action": e["action"],
                 "Status": e["status"], "Location": e["details_json"]["location"]}
                for e in events]
        data = pd.DataFrame(rows, columns=columns)
        data.attrs["timestamps"] = [e["timestamp"] for e in events]
        data.attrs["sources"] = events
        return data

    def get_recent_logs_as_text(self, limit=20):
        # takes the last 20 things that happened and turns them into simple English sentences.
        """
        Converts the last N rows into natural language sentences for the LLM.
        Example Output:
        "- At 2023-10-25 14:00, Person_1 caused an Intrusion (Status: UNAUTHORIZED)."
        """
        df = self.load_data()
        if df.empty:
            return "No events recorded yet."

        # Get last N rows
        recent_df = df.tail(limit)

        narrative = []
        for index, row in recent_df.iterrows():
            timestamp = row['Timestamp']
            entity = row['Entity']
            action = row['Action']
            status = row['Status']

            # Construct a sentence based on the data
            if action == "Intrusion":
                sentence = f"- At {timestamp}, an unauthorized intrusion was detected by {entity}."
            elif action == "Theft":
                sentence = f"- At {timestamp}, a theft attempt was detected involving {entity}."
            elif action == "Removal":
                sentence = f"- At {timestamp}, {entity} removed a secured asset (Authorized Action)."
            elif action == "Access":
                sentence = f"- At {timestamp}, {entity} accessed the zone (Authorized)."
            else:
                sentence = f"- At {timestamp}, {entity} triggered event '{action}' with status {status}."

            narrative.append(sentence)

        return "\n".join(narrative)

    def get_statistics(self):
        # counts everything up to give you a quick report.
        """
        Returns high-level stats for the Agent.
        """
        df = self.load_data()
        if df.empty:
            return "No stats available."

        total_events = len(df)
        unique_entities = df['Entity'].unique().tolist()

        # Count specific events
        thefts = len(df[df['Action'] == 'Theft'])
        intrusions = len(df[df['Action'] == 'Intrusion'])
        authorized = len(df[df['Status'] == 'AUTHORIZED'])

        stats = (
            f"Total Events Recorded: {total_events}\n"
            f"Unique People Seen: {', '.join(unique_entities)}\n"
            f"Total Thefts Detected: {thefts}\n"
            f"Total Intrusions: {intrusions}\n"
            f"Authorized Actions: {authorized}"
        )
        return stats

    def _event_documents(self):
        """Keep the established security-event wording byte-identical."""
        df = self.load_data()
        if df.empty:
            return []

        documents = []

        for index, row in df.iterrows():
            timestamp = row['Timestamp']
            entity = row['Entity']
            action_type = row['Action']
            verb = row['Status']
            loc = row['Location']

            # PREPOSITION LOGIC
            prep = "in the" if loc == "General Area" else "at the"

            # CONSTRUCT SENTENCE
            if action_type == "Identity":
                text = f"At {timestamp}, {entity} was {verb}."
            elif action_type == "Intrusion":
                text = f"At {timestamp}, {entity} was detected intruding {prep} {loc}."
            elif action_type == "Access":
                text = f"At {timestamp}, {entity} accessed the {loc}."
            else:
                text = f"At {timestamp}, {entity} triggered {action_type} {prep} {loc}."

            documents.append((df.attrs["timestamps"][index], text,
                              self._metadata('event', df.attrs['sources'][index],
                                             timestamp, entity, action_type, loc)))

        return documents
    @staticmethod
    def _metadata(kind, row, timestamp, person, action, location):
        # Chroma rejects null metadata. -1 explicitly means unavailable legacy provenance.
        return dict(source_type=kind, source_id=row['id'], run_id=row['run_id'],
                    frame_no=row['frame_no'] if row['frame_no'] is not None else -1,
                    time=timestamp, person=person, action=action, location=location)

    def get_all_logs_formatted(self):
        return [text for text, _ in self.get_narration_records()]

    def enrich_identity(self, documents):
        """Refresh identity evidence from SQLite, even for a previously built Chroma index."""
        with Repository(self.database_url) as repository:
            evidence = repository.get_identity_evidence([d.metadata for d in documents])
        enriched = []
        for doc in documents:
            metadata = dict(doc.metadata)
            metadata.pop('identity_evidence', None)
            key = (metadata.get('source_type'), metadata.get('source_id'), metadata.get('run_id'))
            identity = evidence.get(key)
            if identity:
                metadata['identity_evidence'] = json.dumps(identity, ensure_ascii=False)
            enriched.append(type(doc)(page_content=doc.page_content, metadata=metadata))
        return enriched

    def get_narration_records(self):
        """Merge security events and confidence-filtered activity spans by time.

        Spans are consecutive observations within one (run, track). A changed
        label or rejected-confidence row breaks the span. Other tracks and
        security events do not. Span confidence is the minimum of its rows;
        distinct descriptions are retained in observation order.
        """
        documents = self._event_documents()
        with Repository(self.database_url) as repository:
            observations = repository.get_activities(run_id=self.run_id)
            alerts = repository.get_alerts(run_id=self.run_id)
        groups = []
        current = {}
        for row in observations:
            key = (row['run_id'], row['track_id'])
            if row['confidence'] < config.VLM_ACTIVITY_MIN_LOG_CONFIDENCE:
                current.pop(key, None)
                continue
            group = current.get(key)
            if group is None or group[0]['activity_label'] != row['activity_label']:
                group = []
                groups.append(group)
                current[key] = group
            group.append(row)

        for group in groups:
            first, last = group[0], group[-1]
            start = first['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            end = last['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            when = f'From {start} to {end}' if len(group) > 1 else f'At {start}'
            name = first.get('display_name') or f"Person_{first['track_id']}"
            confidence = min(row['confidence'] for row in group)
            text = f"{when}, {name} was {first['activity_label']} (confidence {confidence:.2f})"
            descriptions = list(dict.fromkeys(row['description'].strip() for row in group
                                             if row['description'].strip()))
            if descriptions:
                text += ': ' + ' '.join(d if d.endswith(('.', '!', '?')) else d + '.'
                                       for d in descriptions)
            else:
                text += '.'
            documents.append((first['timestamp'], text,
                              self._metadata('activity', first, start, name,
                                             first['activity_label'], 'Location not recorded')))
        for alert in alerts:
            timestamp = alert['triggered_at'].strftime('%Y-%m-%d %H:%M:%S')
            text = (f"At {timestamp}, a {alert['severity']} severity alert from rule "
                    f"'{alert['details_json']['rule_name']}' was triggered: {alert['message']}.")
            details = alert['details_json']
            track_id = details.get('track_id')
            person = f'Person_{track_id}' if track_id is not None else 'Multiple / unspecified people'
            documents.append((alert['triggered_at'], text,
                              self._metadata('alert', alert, timestamp, person,
                                             alert['message'], details.get('zone_name') or 'In view')))
        return [(text, metadata) for _, text, metadata in sorted(documents, key=lambda item: item[0])]
