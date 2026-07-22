# csv_utils.py

import pandas as pd
import os
import config

class LogAnalyzer:
    def __init__(self): 
        # remembers where the CSV file (the logbook) is located on your computer.
        self.csv_path = config.LOG_PATH

    def load_data(self):
        #  It opens the CSV file and loads it into a table (DataFrame) that the code can understand.
        """
        Safely loads the CSV file into a pandas DataFrame.
        """
        # Define the exact columns written by event_logger.py
        required_columns = ["Timestamp", "Entity", "Action", "Status", "Location"]

        if not os.path.exists(self.csv_path):
            # Return empty DF with ALL 5 correct columns
            return pd.DataFrame(columns=required_columns)

        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return pd.DataFrame(columns=required_columns)
            
            df.columns = [c.strip() for c in df.columns]
            
            if 'Location' not in df.columns:
                 df['Location'] = "General Area"
                 
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return pd.DataFrame(columns=required_columns)

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

    def get_all_logs_formatted(self):
        # It reads everything that ever happened and turns it into a detailed story.
        """
        Reads ALL logs and returns a list of natural language sentences.
        Uses state-tracking to describe transitions (e.g., Sitting -> Walking = 'got up').
        """
        df = self.load_data()
        if df.empty:
            return []

        documents = []
        # tracker to remember the last verb seen for each person
        # Format: { "Talha": "Sitting", "Person_1": "Walking" }
        last_action_per_entity = {}

        for index, row in df.iterrows():
            timestamp = row['Timestamp']
            entity = row['Entity']
            action_type = row['Action']
            verb = row['Status']
            loc = row['Location']

            # 1. PREPOSITION LOGIC (From previous step)
            prep = "at the"
            if loc == "General Area": prep = "in the"
            elif verb in ["Sitting", "Lying Down"] and loc.lower() in ["sofa", "chair", "bed", "couch"]:
                prep = "on the"
            elif verb in ["Walking", "Standing", "Running", "Bending", "Picking Up"]:
                prep = "near the"

            # 2. TRANSITION LOGIC (State Awareness)
            last_verb = last_action_per_entity.get(entity, None)
            desc = f"was seen {verb.lower()}" # Default

            if action_type == "Behavior":
                if verb == "Walking":
                    if last_verb in ["Sitting", "Lying Down"]:
                        desc = "got up and moved"
                    elif last_verb == "Bending":
                        desc = "straightened up and moved"
                    else:
                        desc = "approached" if loc != "General Area" else "was seen walking"

                elif verb == "Sitting":
                    desc = "sat down" if last_verb == "Walking" else "was seen sitting"

                elif verb == "Lying Down":
                    desc = "lay down" if last_verb == "Walking" else "was seen lying down"

                elif verb == "Bending":
                    desc = "stooped down"

                elif verb == "Picking Up":
                    desc = "picked something up"

                elif verb == "Running":
                    desc = "started running"

                # Update the tracker for the next row
                last_action_per_entity[entity] = verb

            # 3. CONSTRUCT SENTENCE
            if action_type == "Behavior":
                text = f"At {timestamp}, {entity} {desc} {prep} {loc}."
            elif action_type == "Identity":
                text = f"At {timestamp}, {entity} was {verb}."
            elif action_type == "Intrusion":
                text = f"At {timestamp}, {entity} was detected intruding {prep} {loc}."
            elif action_type == "Access":
                text = f"At {timestamp}, {entity} accessed the {loc}."
            else:
                text = f"At {timestamp}, {entity} triggered {action_type} {prep} {loc}."

            documents.append(text)

        return documents