"""Streamlit controls for additive rule templates and persisted alert history."""
from datetime import datetime, time

import streamlit as st
import config
from alerts.rules_engine import RULE_TYPES, SEVERITIES
from storage import StorageFatalError
from storage.repository import Repository


def recording_start_input():
    current = st.session_state.get('recording_start_time', datetime.now())
    date = st.date_input('Recording start date', value=current.date(), key='recording_date')
    clock = st.time_input('Recording start time', value=current.time().replace(microsecond=0),
                          key='recording_clock')
    st.session_state.recording_start_time = datetime.combine(date, clock)
    st.caption('Local time shown in the recording. Rule times use this start plus video time; '
               'processing speed does not affect rule durations.')


def alert_rules_tab():
    st.header('Alert Rules')
    st.caption('Changes apply to the next run. Rules add alerts; existing intrusion/theft events '
               'remain unchanged. Deleting a rule keeps its historical alerts.')
    try:
        with Repository() as repository:
            rules = repository.get_rules()
            by_id = {r['id']: r for r in rules}
            choices = ['New rule'] + list(by_id)
            current_rule = st.session_state.get('rule_selector', 'New rule')
            selected = st.selectbox('Rule to edit', choices,
                index=choices.index(current_rule) if current_rule in choices else 0,
                format_func=lambda value: by_id[value]['name'] if value in by_id else value, key='rule_selector')
            old = by_id.get(selected, {})
            prefix = 'alert_rule_' + selected
            kind = st.selectbox('Rule type', RULE_TYPES,
                index=RULE_TYPES.index(old.get('type', RULE_TYPES[0])), key=prefix+'_type')
            params = old.get('params_json', {}) if old.get('type', kind) == kind else {}
            with st.form(prefix + '_' + kind):
                name = st.text_input('Rule name', value=old.get('name', ''))
                severity = st.selectbox('Severity', SEVERITIES,
                    index=SEVERITIES.index(old.get('severity', 'medium')))
                enabled = st.checkbox('Enabled', value=old.get('enabled', True))
                cooldown = st.number_input('Cooldown (video seconds)', min_value=0.0,
                    value=float(old.get('cooldown_seconds', config.ALERT_DEFAULT_COOLDOWN_SECONDS)))
                zone = st.text_input('Zone name (required for schedule/occupancy; blank means anywhere otherwise)',
                                     value=params.get('zone_name') or '').strip()
                values = {'zone_name': zone or None}
                if kind == 'zone_schedule':
                    start = st.time_input('Restricted from', value=time.fromisoformat(params.get('start', '22:00')))
                    end = st.time_input('Restricted until', value=time.fromisoformat(params.get('end', '06:00')))
                    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                    days = st.multiselect('Days the window starts (empty = every day)', weekdays,
                                         default=[weekdays[d] for d in params.get('days', [])])
                    st.caption('Overnight windows belong to their starting day. Start is included; end is excluded.')
                    values.update(start=start.isoformat(), end=end.isoformat(),
                                  days=[weekdays.index(day) for day in days])
                if kind == 'occupancy':
                    values['max_people'] = st.number_input('Alert when people exceed', min_value=0,
                                                           value=int(params.get('max_people', 1)), step=1)
                if kind in ('occupancy', 'unknown_dwell'):
                    values['duration_seconds'] = st.number_input('Minimum duration (video seconds)', min_value=0.0,
                        value=float(params.get('duration_seconds', config.ALERT_DEFAULT_DURATION_SECONDS)))
                if kind == 'activity':
                    values['label'] = st.text_input('VLM activity label', value=params.get('label', 'running')).strip()
                    values['min_confidence'] = st.number_input('Minimum activity confidence', min_value=0.0,
                        max_value=1.0, value=float(params.get('min_confidence', config.ALERT_DEFAULT_ACTIVITY_CONFIDENCE)))
                    st.caption('Requires a fresh VLM result. No running detector is added.')
                submitted = st.form_submit_button('Save rule')
            if submitted:
                repository.save_rule(name, kind, values, severity, enabled, cooldown,
                                     rule_id=old.get('id'))
                st.rerun()
            if old and st.button('Delete rule', key=prefix+'_delete'):
                repository.delete_rule(old['id'])
                st.rerun()

            st.subheader('Alerts history')
            runs = repository.get_runs()
            run_labels = {r['id']: f"{r['video_name']} | {r['started_at']} | {r['id'][:8]}" for r in runs}
            run_filter = st.selectbox('Filter by run', ['All runs'] + list(run_labels),
                format_func=lambda value: run_labels.get(value, value), key='alert_run_filter')
            severity_filter = st.selectbox('Filter by severity', ['All severities'] + list(SEVERITIES),
                                           key='alert_severity_filter')
            rows = repository.get_alerts(
                None if run_filter == 'All runs' else run_filter,
                None if severity_filter == 'All severities' else severity_filter)
            if rows:
                st.dataframe([{k: row[k] for k in ('id','run_id','severity','triggered_at',
                             'video_offset_s','message','acknowledged')} for row in rows], use_container_width=True)
                pending = {row['id']: row for row in rows if not row['acknowledged']}
                alert_id = st.selectbox('Alert to acknowledge', list(pending) or [None],
                    format_func=lambda value: (f"#{value}: {pending[value]['message']}"
                                               if value in pending else 'No pending alerts'),
                    key='acknowledge_alert_id', disabled=not pending)
                if st.button('Acknowledge alert', disabled=not pending):
                    repository.acknowledge_alert(alert_id)
                    st.rerun()
            else:
                st.info('No alerts match these filters.')
    except (StorageFatalError, ValueError) as exc:
        st.error(str(exc))
