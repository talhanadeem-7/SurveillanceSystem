"""Run-scoped unit of work. Queue writes, commit once at the frame boundary."""
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import config
from storage import StorageFatalError
from storage.db import (ActivityObservation, Camera, Event, Person,
                        Run, Zone, AlertRule, AlertRecord, build_engine, new_id)
from alerts.rules_engine import validate_rule


def as_datetime(value=None):
    result = datetime.now() if value is None else (
        value if isinstance(value, datetime) else datetime.fromisoformat(value))
    return (result.astimezone(timezone.utc).replace(tzinfo=None)
            if result.tzinfo else result)


def storage_operation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (SQLAlchemyError, OSError) as exc:
            if getattr(self, 'session', None) is not None:
                try:
                    self.session.rollback()
                except (SQLAlchemyError, OSError):
                    pass  # Preserve the original storage failure if the connection is lost.
            self.people = {}
            raise StorageFatalError(f'{method.__name__}: {exc}') from exc
    return guarded


class Repository:
    @storage_operation
    def __init__(self, url=None):
        self.engine = build_engine(url)
        self.session = Session(self.engine, expire_on_commit=False, autoflush=False)
        self.people = {}
        self.zone_ids = {}
        # Persist finalization intent separately when the database itself is unwritable.
        key = hashlib.sha256(str(self.engine.url).encode()).hexdigest()[:16]
        self.recovery_dir = Path(config.BASE_DIR) / 'storage' / 'pending_runs' / key
        self.recover_runs()

    @storage_operation
    def recover_runs(self):
        for path in self.recovery_dir.glob('*.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            run = self.session.get(Run, data.pop('run_id'))
            if run is not None:
                data['ended_at'] = as_datetime(data['ended_at'])
                for field, value in data.items():
                    setattr(run, field, value)
                self.session.commit()
            path.unlink()

    @storage_operation
    def ensure_camera(self, source, name=None):
        camera = self.session.scalar(select(Camera).where(Camera.source == str(source)))
        if camera is None:
            camera = Camera(id=new_id(), name=name or Path(source).name, source=str(source))
            self.session.add(camera)
            self.session.commit()
        return camera.id

    @storage_operation
    def start_run(self, video_name, source=None, camera_id=None, started_at=None,
                  metadata=None, commit=True):
        if camera_id is None:
            camera_id = self.ensure_camera(source or video_name, video_name)
        run = Run(id=new_id(), camera_id=camera_id, video_name=video_name,
                  started_at=as_datetime(started_at), status='running',
                  metadata_json=metadata or {})
        self.session.add(run)
        self.session.flush()
        if commit:
            self.session.commit()
        return run.id

    @storage_operation
    def save_zones(self, run_id, zones):
        result = {}
        for zone in zones:
            geometry = {}
            for key in ('coords', 'orig_coords', 'polygon'):
                if key in zone:
                    value = zone[key]
                    geometry[key] = value.tolist() if hasattr(value, 'tolist') else list(value)
            item = Zone(id=new_id(), run_id=run_id, name=zone['name'],
                        type=zone['type'], geometry_json=geometry)
            self.session.add(item)
            result[zone['name']] = item.id
        self.zone_ids[run_id] = result
        self.session.flush()
        return result

    @storage_operation
    def upsert_person(self, run_id, track_id, display_name=None, is_enrolled=False,
                      timestamp=None):
        track_id = int(track_id)
        if track_id < 0:
            return None
        key = (run_id, track_id)
        person = self.people.get(key)
        if person is None:
            person = self.session.scalar(select(Person).where(
                Person.run_id == run_id, Person.track_id == track_id))
        now = as_datetime(timestamp)
        if person is None:
            person = Person(id=new_id(), run_id=run_id, track_id=track_id,
                            display_name=display_name or f'Person_{track_id}',
                            is_enrolled=is_enrolled, first_seen=now, last_seen=now)
            self.session.add(person)
        else:
            person.last_seen = max(person.last_seen, now)
            person.first_seen = min(person.first_seen, now)
            if display_name is not None:
                person.display_name = display_name
                person.is_enrolled = is_enrolled
        self.people[key] = person
        return person.id

    @storage_operation
    def log_event(self, run_id, action, status, entity, location='General Area',
                  person_id=None, zone_id=None, timestamp=None, frame_no=None,
                  details=None, snapshot_path=None):
        self.session.add(Event(run_id=run_id, action=action, status=status,
            timestamp=as_datetime(timestamp), frame_no=frame_no, person_id=person_id,
            zone_id=zone_id or self.zone_ids.get(run_id, {}).get(location),
            details_json={**(details or {}), 'entity': entity, 'location': location},
            snapshot_path=snapshot_path))

    @storage_operation
    def log_activity(self, run_id, track_id, activity_label, description, confidence,
                     timestamp, frame_no=None, details=None):
        person_id = self.upsert_person(run_id, track_id, timestamp=timestamp)
        self.session.add(ActivityObservation(run_id=run_id, person_id=person_id,
            track_id=int(track_id), activity_label=activity_label, description=description,
            confidence=float(confidence), timestamp=as_datetime(timestamp),
            frame_no=frame_no, details_json=details or {}))

    @storage_operation
    def flush(self):
        # Explicit parent ordering because rows use foreign-key IDs, not ORM relationships.
        parents = [row for row in self.session.new if isinstance(row, (Person, Zone))]
        if parents:
            self.session.flush(parents)
        self.session.commit()

    def flush_frame(self, frame_no):
        # Events are durable at the end of their frame. Person last_seen-only
        # updates can share a transaction across 30 source frames.
        if frame_no % 30 == 0 or any(isinstance(row, (Event, ActivityObservation, AlertRecord))
                                   for row in self.session.new):
            self.flush()

    @storage_operation
    def end_run(self, run_id, status, frames_processed=0, frames_skipped=0):
        if status not in ('completed', 'aborted', 'interrupted'):
            raise ValueError(f'Invalid terminal status: {status}')
        values = dict(status=status, frames_processed=int(frames_processed),
                      frames_skipped=int(frames_skipped), ended_at=datetime.now())
        try:
            # A failed frame has already rolled back; healthy pending work is flushed.
            self.flush()
            run = self.session.get(Run, run_id)
            if run is None:
                raise ValueError(f'Unknown run: {run_id}')
            for field, value in values.items():
                setattr(run, field, value)
            self.session.commit()
            (self.recovery_dir / f'{run_id}.json').unlink(missing_ok=True)
        except StorageFatalError:
            self._record_pending(run_id, values)
            raise
        except (SQLAlchemyError, OSError):
            self._record_pending(run_id, values)
            raise

    def _record_pending(self, run_id, values):
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        path = self.recovery_dir / f'{run_id}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(run_id=run_id, **values), default=str), encoding='utf-8')
        temporary.replace(path)

    @storage_operation
    def get_events(self, run_id=None, since=None, action=None):
        query = select(Event).order_by(Event.id)
        if run_id is not None:
            query = query.where(Event.run_id == run_id)
        if since is not None:
            query = query.where(Event.timestamp >= as_datetime(since))
        if action is not None:
            query = query.where(Event.action == action)
        return [self._dict(row) for row in self.session.scalars(query)]

    @storage_operation
    def get_runs(self):
        return [self._dict(row) for row in self.session.scalars(select(Run).order_by(Run.started_at))]

    @storage_operation
    def get_cameras(self):
        return [self._dict(row) for row in self.session.scalars(select(Camera).order_by(Camera.created_at))]

    @storage_operation
    def get_zones(self, run_id):
        return [self._dict(row) for row in self.session.scalars(select(Zone).where(Zone.run_id == run_id))]

    @storage_operation
    def get_people(self, run_id):
        return [self._dict(row) for row in self.session.scalars(select(Person).where(Person.run_id == run_id))]

    @storage_operation
    def get_activities(self, run_id=None):
        """Read timestamp-ordered observations with the current known display name."""
        query = (select(ActivityObservation, Person.display_name)
                 .outerjoin(Person, ActivityObservation.person_id == Person.id)
                 .order_by(ActivityObservation.timestamp, ActivityObservation.id))
        if run_id is not None:
            query = query.where(ActivityObservation.run_id == run_id)
        return [dict(self._dict(row), display_name=name)
                for row, name in self.session.execute(query)]

    @storage_operation
    def save_rule(self, name, type, params_json, severity='medium', enabled=True,
                  cooldown_seconds=None, rule_id=None):
        values = validate_rule(dict(name=name, type=type, params_json=params_json,
            severity=severity, enabled=enabled,
            cooldown_seconds=(config.ALERT_DEFAULT_COOLDOWN_SECONDS
                              if cooldown_seconds is None else cooldown_seconds)))
        rule = self.session.get(AlertRule, rule_id) if rule_id else AlertRule(id=new_id())
        if rule is None:
            raise ValueError('Rule no longer exists.')
        for key, value in values.items():
            setattr(rule, key, value)
        rule.zone_id = None  # Templates are name-based, never tied to an old run.
        self.session.add(rule)
        self.flush()
        return rule.id

    @storage_operation
    def get_rules(self, enabled_only=False):
        query = select(AlertRule).order_by(AlertRule.created_at, AlertRule.id)
        if enabled_only:
            query = query.where(AlertRule.enabled.is_(True))
        return [self._dict(row) for row in self.session.scalars(query)]

    @storage_operation
    def delete_rule(self, rule_id):
        rule = self.session.get(AlertRule, rule_id)
        if rule is None:
            raise ValueError('Rule no longer exists.')
        self.session.execute(update(AlertRecord).where(AlertRecord.rule_id == rule_id).values(rule_id=None))
        self.session.delete(rule)
        self.flush()

    @storage_operation
    def log_alert(self, run_id, alert, person_id=None):
        # A template may have been deleted in another UI session since run start.
        rule_id = alert.rule_id if self.session.get(AlertRule, alert.rule_id) else None
        self.session.add(AlertRecord(run_id=run_id, rule_id=rule_id, person_id=person_id,
            zone_id=alert.zone_id, triggered_at=as_datetime(alert.triggered_at),
            video_offset_s=alert.video_offset_s, frame_no=alert.frame_no,
            message=alert.message, details_json=alert.details_json,
            severity=alert.severity, acknowledged=False))

    @storage_operation
    def get_alerts(self, run_id=None, severity=None):
        query = select(AlertRecord).order_by(AlertRecord.triggered_at, AlertRecord.id)
        if run_id is not None:
            query = query.where(AlertRecord.run_id == run_id)
        if severity is not None:
            query = query.where(AlertRecord.severity == severity)
        return [self._dict(row) for row in self.session.scalars(query)]

    @storage_operation
    def acknowledge_alert(self, alert_id):
        alert = self.session.get(AlertRecord, alert_id)
        if alert is None:
            raise ValueError('Alert no longer exists.')
        alert.acknowledged = True
        self.flush()

    @staticmethod
    def _dict(row):
        return {column.name: getattr(row, column.name) for column in row.__table__.columns}

    def rollback(self):
        self.session.rollback()
        self.people.clear()

    def close(self):
        self.session.close()
        self.engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
