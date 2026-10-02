"""Database schema and engine setup. SQLAlchemy stays inside storage."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (Boolean, CheckConstraint, DateTime, Float, ForeignKey,
                        Index, Integer, JSON, String, Text, UniqueConstraint,
                        create_engine, event)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

import config


def new_id():
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = 'cameras'
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Run(Base):
    __tablename__ = 'runs'
    __table_args__ = (CheckConstraint("status IN ('running','completed','aborted','interrupted')"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    camera_id: Mapped[str] = mapped_column(ForeignKey('cameras.id'))
    video_name: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16), default='running')
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    frames_skipped: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class Zone(Base):
    __tablename__ = 'zones'
    __table_args__ = (CheckConstraint("type IN ('restricted','passive')"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey('runs.id'))
    name: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(16))
    geometry_json: Mapped[dict] = mapped_column(JSON)


class Person(Base):
    __tablename__ = 'persons'
    __table_args__ = (UniqueConstraint('run_id', 'track_id'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey('runs.id'))
    track_id: Mapped[int] = mapped_column(Integer)
    display_name: Mapped[str] = mapped_column(Text)
    is_enrolled: Mapped[bool] = mapped_column(Boolean, default=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime)
    last_seen: Mapped[datetime] = mapped_column(DateTime)


class Event(Base):
    __tablename__ = 'events'
    __table_args__ = (Index('ix_events_run_timestamp', 'run_id', 'timestamp'),
                      Index('ix_events_action', 'action'))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey('runs.id'))
    person_id: Mapped[str | None] = mapped_column(ForeignKey('persons.id'))
    zone_id: Mapped[str | None] = mapped_column(ForeignKey('zones.id'))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    frame_no: Mapped[int | None] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(Text)
    details_json: Mapped[dict] = mapped_column(JSON)
    snapshot_path: Mapped[str | None] = mapped_column(Text)


class ActivityObservation(Base):
    __tablename__ = 'activity_observations'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey('runs.id'))
    person_id: Mapped[str | None] = mapped_column(ForeignKey('persons.id'))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    track_id: Mapped[int] = mapped_column(Integer)
    activity_label: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    frame_no: Mapped[int | None] = mapped_column(Integer)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict)


class AlertRule(Base):
    __tablename__ = 'alert_rules'
    __table_args__ = (CheckConstraint("severity IN ('low','medium','high')"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(32))
    # Templates keep zone_name in params_json; zone_id is resolved per run in memory.
    zone_id: Mapped[str | None] = mapped_column(ForeignKey('zones.id'))
    params_json: Mapped[dict] = mapped_column(JSON)
    severity: Mapped[str] = mapped_column(String(8))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    cooldown_seconds: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AlertRecord(Base):
    __tablename__ = 'alerts'
    __table_args__ = (Index('ix_alerts_run_time', 'run_id', 'triggered_at'),
                     CheckConstraint("severity IN ('low','medium','high')"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey('runs.id'))
    rule_id: Mapped[str | None] = mapped_column(ForeignKey('alert_rules.id', ondelete='SET NULL'))
    person_id: Mapped[str | None] = mapped_column(ForeignKey('persons.id'))
    zone_id: Mapped[str | None] = mapped_column(ForeignKey('zones.id'))
    triggered_at: Mapped[datetime] = mapped_column(DateTime)
    video_offset_s: Mapped[float] = mapped_column(Float)
    frame_no: Mapped[int] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(Text)
    details_json: Mapped[dict] = mapped_column(JSON)
    # Snapshot severity so editing/deleting a template never rewrites history.
    severity: Mapped[str] = mapped_column(String(8))
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)


def build_engine(url=None):
    url = make_url(url or config.DATABASE_URL)
    if url.get_backend_name() == 'sqlite' and url.database not in (None, '', ':memory:'):
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **({'connect_args': {'timeout': 0.25}}
                                  if url.get_backend_name() == 'sqlite' else {}))
    if engine.dialect.name == 'sqlite':
        @event.listens_for(engine, 'connect')
        def configure_sqlite(connection, _):
            cursor = connection.cursor()
            # SQLite-specific connection settings only; all data SQL is portable.
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA foreign_keys=ON')
            cursor.close()
    Base.metadata.create_all(engine)
    return engine
