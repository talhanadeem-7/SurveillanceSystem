"""Pure video-time rule evaluation. Returned alerts are the future notifier seam.

One alert per continuous condition; after it clears, a new episode may fire once
the cooldown from the last alert has elapsed. Days on overnight schedules refer
to the day the window starts. Windows are start-inclusive and end-exclusive.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import math

RULE_TYPES = ('zone_schedule', 'occupancy', 'unknown_dwell', 'activity')
SEVERITIES = ('low', 'medium', 'high')


def validate_rule(rule):
    rule = deepcopy(rule)
    if not str(rule.get('name', '')).strip():
        raise ValueError('A rule name is required.')
    if rule.get('type') not in RULE_TYPES:
        raise ValueError('Unknown rule type.')
    if rule.get('severity') not in SEVERITIES:
        raise ValueError('Severity must be low, medium or high.')
    def number(value, name, minimum=0, maximum=None):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number.')
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f'{name} is outside the allowed range.')
    number(rule.get('cooldown_seconds', 0), 'Cooldown')
    params = rule.get('params_json', {})
    if not isinstance(params, dict):
        raise ValueError('Rule parameters must be an object.')
    if not isinstance(rule.get('enabled', True), bool):
        raise ValueError('Enabled must be true or false.')
    zone = params.get('zone_name')
    if zone is not None and (not isinstance(zone, str) or not zone.strip()):
        raise ValueError('Zone name must be nonempty or unset.')
    if rule['type'] in ('zone_schedule', 'occupancy') and not zone:
        raise ValueError('This rule requires a zone name.')
    if rule['type'] == 'zone_schedule':
        try:
            start, end = time.fromisoformat(params['start']), time.fromisoformat(params['end'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Schedule requires valid start and end times.') from exc
        if start == end or start.tzinfo or end.tzinfo:
            raise ValueError('Schedule endpoints must differ and use local recording time.')
        days = params.get('days', [])
        if not isinstance(days, list) or any(type(d) is not int or d not in range(7) for d in days):
            raise ValueError('Days must be weekday numbers 0 (Monday) through 6.')
    if rule['type'] in ('occupancy', 'unknown_dwell'):
        number(params.get('duration_seconds'), 'Duration')
    if rule['type'] == 'occupancy':
        if type(params.get('max_people')) is not int or params['max_people'] < 0:
            raise ValueError('Maximum people must be a nonnegative integer.')
    if rule['type'] == 'activity':
        if not isinstance(params.get('label'), str) or not params['label'].strip():
            raise ValueError('An activity label is required.')
        number(params.get('min_confidence'), 'Confidence', 0, 1)
    rule['name'] = rule['name'].strip()
    return rule


@dataclass(frozen=True)
class TrackState:
    track_id: int
    display_name: str = ''
    is_enrolled: bool = False
    zones: frozenset[str] = field(default_factory=frozenset)
    activity_label: str | None = None
    activity_confidence: float = 0.0


@dataclass(frozen=True)
class FrameState:
    frame_no: int
    tracks: tuple[TrackState, ...] = ()


@dataclass(frozen=True)
class Alert:
    rule_id: str
    severity: str
    track_id: int | None
    zone_id: str | None
    triggered_at: datetime
    video_offset_s: float
    frame_no: int
    message: str
    details_json: dict


def resolve_rules(rules, zones):
    """Bind name-based templates to this run's zones; exclude missing/ambiguous names."""
    names = {}
    for zone in zones:
        names.setdefault(zone['name'], []).append(zone['id'])
    resolved, warnings = [], []
    for source in rules:
        if not source.get('enabled', True):
            continue
        rule = validate_rule(source)
        name = rule['params_json'].get('zone_name')
        if name and len(names.get(name, [])) != 1:
            warnings.append(f"Rule '{rule['name']}' skipped: zone '{name}' is missing or ambiguous in this run.")
            continue
        rule['zone_id'] = names[name][0] if name else None
        resolved.append(rule)
    return resolved, warnings


class RulesEngine:
    def __init__(self, rules, fps, start_time):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Source FPS must be positive.')
        self.rules = [validate_rule(r) for r in rules if r.get('enabled', True)]
        self.fps, self.start_time = fps, start_time
        self.active = {}
        self.last_fired = {}
        self.last_frame = -1

    @staticmethod
    def _scheduled(params, instant):
        start, end = time.fromisoformat(params['start']), time.fromisoformat(params['end'])
        clock = instant.time().replace(tzinfo=None)
        day = instant.weekday()
        if start < end:
            inside = start <= clock < end
        else:
            inside = clock >= start or clock < end
            if clock < end:
                day = (day - 1) % 7
        return inside and day in (params.get('days') or list(range(7)))

    def update(self, frame_state):
        if frame_state.frame_no <= self.last_frame:
            raise ValueError('Frame numbers must increase within a run.')
        self.last_frame = frame_state.frame_no
        offset = frame_state.frame_no / self.fps
        instant = self.start_time + timedelta(seconds=offset)
        tracks = {t.track_id: t for t in frame_state.tracks if t.track_id >= 0}
        alerts, current_keys = [], set()
        for rule in self.rules:
            params, kind = rule['params_json'], rule['type']
            zone = params.get('zone_name')
            members = [t for t in tracks.values() if zone is None or zone in t.zones]
            candidates = []
            if kind == 'occupancy':
                if len(members) > params['max_people']:
                    candidates = [(None, f"{len(members)} people in {zone} (limit {params['max_people']})",
                                   {'count': len(members), 'track_ids': sorted(t.track_id for t in members)})]
            else:
                for track in members:
                    name = track.display_name or f'Person_{track.track_id}'
                    place = f' in {zone}' if zone else ' in view'
                    if kind == 'zone_schedule' and self._scheduled(params, instant):
                        candidates.append((track.track_id, f'{name} entered scheduled restricted zone {zone}', {}))
                    elif kind == 'unknown_dwell' and not track.is_enrolled:
                        candidates.append((track.track_id, f'Unknown person {name} remained{place}', {}))
                    elif (kind == 'activity' and track.activity_label == params['label']
                          and track.activity_confidence >= params['min_confidence']):
                        candidates.append((track.track_id, f'{name} was {params["label"]}{place}',
                                           {'confidence': track.activity_confidence}))
            for track_id, message, details in candidates:
                key = (rule['id'], track_id if track_id is not None else zone)
                current_keys.add(key)
                started, fired = self.active.setdefault(key, (offset, False))
                duration = params.get('duration_seconds', 0) if kind in ('occupancy', 'unknown_dwell') else 0
                elapsed = offset - started
                if fired or elapsed + 1e-9 < duration:
                    continue
                if offset - self.last_fired.get(key, -math.inf) + 1e-9 < rule.get('cooldown_seconds', 0):
                    continue
                self.active[key] = (started, True)
                self.last_fired[key] = offset
                if kind in ('occupancy', 'unknown_dwell'):
                    message += f' for {elapsed:.2f} seconds'
                alerts.append(Alert(rule['id'], rule['severity'], track_id, rule.get('zone_id'),
                    instant, offset, frame_state.frame_no, message,
                    dict(details, rule_name=rule['name'], rule_type=kind, zone_name=zone,
                         params=deepcopy(params), duration_seconds=elapsed, track_id=track_id)))
        self.active = {key: value for key, value in self.active.items() if key in current_keys}
        return alerts
