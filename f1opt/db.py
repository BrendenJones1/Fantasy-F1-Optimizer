"""SQLAlchemy schema and helpers. Schema changes go through Alembic migrations."""
from datetime import date as date_type, datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import (Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, Numeric, String,
                        Text, UniqueConstraint, create_engine)
from sqlalchemy.dialects import mysql, sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Driver(Base):
    __tablename__ = "drivers"
    driver_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    code: Mapped[Optional[str]] = mapped_column(String(8))
    name: Mapped[str] = mapped_column(String(128))


class Constructor(Base):
    __tablename__ = "constructors"
    constructor_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))


class Race(Base):
    """One row per calendar race, including upcoming ones (``completed`` False)."""
    __tablename__ = "races"
    __table_args__ = (UniqueConstraint("season", "round", name="uq_races_season_round"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer)
    round: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(128))
    date: Mapped[date_type] = mapped_column(Date)
    circuit_id: Mapped[str] = mapped_column(String(64))
    is_sprint_weekend: Mapped[bool] = mapped_column(Boolean, default=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)


class Result(Base):
    """A driver's realised weekend, with the Fantasy points derived from it."""
    __tablename__ = "results"
    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_results_race_driver"),
        Index("ix_results_driver", "driver_id"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"))
    constructor_id: Mapped[str] = mapped_column(ForeignKey("constructors.constructor_id"))
    grid: Mapped[int] = mapped_column(Integer)
    finish_position: Mapped[int] = mapped_column(Integer)
    position_text: Mapped[str] = mapped_column(String(8))
    quali_position: Mapped[Optional[int]] = mapped_column(Integer)
    quali_pts: Mapped[float] = mapped_column(Float)
    sprint_pts: Mapped[float] = mapped_column(Float)
    race_pts: Mapped[float] = mapped_column(Float)
    fantasy_points: Mapped[float] = mapped_column(Float)


class DriverPrice(Base):
    """Fantasy price ($M) of a driver ahead of a race, with where it came from."""
    __tablename__ = "driver_prices"
    __table_args__ = (UniqueConstraint("race_id", "driver_id", "source", name="uq_driver_price"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"))
    source: Mapped[str] = mapped_column(String(32))
    price: Mapped[float] = mapped_column(Numeric(6, 2))
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ConstructorPrice(Base):
    __tablename__ = "constructor_prices"
    __table_args__ = (UniqueConstraint("race_id", "constructor_id", "source", name="uq_constructor_price"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    constructor_id: Mapped[str] = mapped_column(ForeignKey("constructors.constructor_id"))
    source: Mapped[str] = mapped_column(String(32))
    price: Mapped[float] = mapped_column(Numeric(6, 2))
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class IngestionRun(Base):
    """Audit trail of scheduled ingestion attempts."""
    __tablename__ = "ingestion_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16))  # running | success | failed
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    seasons: Mapped[Optional[str]] = mapped_column(String(64))
    new_races: Mapped[int] = mapped_column(Integer, default=0)
    results_upserted: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text)


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)


def upsert(session: Session, model, rows: Iterable[dict], key_columns: list, update_columns: list) -> int:
    """Insert rows, updating ``update_columns`` when the unique key already exists.

    Uses the dialect's native upsert (MySQL ON DUPLICATE KEY UPDATE, SQLite ON CONFLICT) so that
    re-running ingestion never duplicates rows. ``key_columns`` must match a unique constraint.
    """
    rows = list(rows)
    if not rows:
        return 0
    dialect = session.get_bind().dialect.name
    table = model.__table__
    if dialect == "mysql":
        stmt = mysql.insert(table).values(rows)
        stmt = stmt.on_duplicate_key_update({c: stmt.inserted[c] for c in update_columns})
    elif dialect == "sqlite":
        stmt = sqlite.insert(table).values(rows)
        stmt = stmt.on_conflict_do_update(index_elements=key_columns,
                                          set_={c: stmt.excluded[c] for c in update_columns})
    else:
        raise NotImplementedError(f"upsert not implemented for {dialect}")
    session.execute(stmt)
    return len(rows)
