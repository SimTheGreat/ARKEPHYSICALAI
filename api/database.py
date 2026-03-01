"""
Database connection and models for production state tracking
"""
import os
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Integer, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

logger = logging.getLogger(__name__)

# Database connection
# Default to local SQLite for dev so the API can run without Postgres/psycopg2.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./arke_production.db")

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ProductionState(Base):
    """
    Tracks the current state of production for each production order
    """
    __tablename__ = "production_state"
    
    # Primary key
    active_po = Column(String, primary_key=True, index=True, comment="Active Production Order ID")
    
    # SMT (Surface Mount Technology)
    op_smt_required = Column(Boolean, default=False, comment="Is SMT required for this order")
    op_smt_finished_at = Column(DateTime, nullable=True, comment="SMT operation completed timestamp")
    
    # Reflow Soldering
    op_reflow_required = Column(Boolean, default=False, comment="Is Reflow required for this order")
    op_reflow_finished_at = Column(DateTime, nullable=True, comment="Reflow operation completed timestamp")
    
    # THT (Through-Hole Technology)
    op_tht_required = Column(Boolean, default=False, comment="Is THT required for this order")
    op_tht_finished_at = Column(DateTime, nullable=True, comment="THT operation completed timestamp")
    
    # AOI (Automated Optical Inspection)
    op_aoi_required = Column(Boolean, default=False, comment="Is AOI required for this order")
    op_aoi_finished_at = Column(DateTime, nullable=True, comment="AOI operation completed timestamp")
    
    # Test
    op_test_required = Column(Boolean, default=False, comment="Is Test required for this order")
    op_test_finished_at = Column(DateTime, nullable=True, comment="Test operation completed timestamp")
    
    # Coating
    op_coating_required = Column(Boolean, default=False, comment="Is Coating required for this order")
    op_coating_finished_at = Column(DateTime, nullable=True, comment="Coating operation completed timestamp")
    
    # Pack
    op_pack_required = Column(Boolean, default=False, comment="Is Pack required for this order")
    op_pack_finished_at = Column(DateTime, nullable=True, comment="Pack operation completed timestamp")


class ScheduleEntry(Base):
    """
    Persisted schedule – one row per order in the current schedule.
    Replaced in bulk each time the schedule is (re)calculated.
    """
    __tablename__ = "schedule_entry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    schedule_version = Column(Integer, nullable=False, index=True, comment="Monotonic version counter")
    position = Column(Integer, nullable=False, comment="1-based position in schedule")
    order_number = Column(String, nullable=False)
    customer = Column(String, nullable=True)
    product = Column(String, nullable=True)
    quantity = Column(Integer, nullable=True)
    priority = Column(Integer, nullable=True)
    starts_at = Column(String, nullable=True)
    ends_at = Column(String, nullable=True)
    deadline = Column(String, nullable=True)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProductionLog(Base):
    """
    Append-only log of schedule changes, warnings, and events.
    """
    __tablename__ = "production_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String, nullable=False, comment="info | warning | change | error")
    category = Column(String, nullable=False, comment="schedule | conflict | order | operation")
    title = Column(String, nullable=False)
    detail = Column(Text, nullable=True)
    order_number = Column(String, nullable=True, index=True)
    schedule_version = Column(Integer, nullable=True)


class ArkePushRecord(Base):
    """
    Tracks which orders have been pushed to Arke (idempotency guard).
    One row per sales-order number.  Re-triggering the push skips
    orders that already have status='pushed'.
    """
    __tablename__ = "arke_push_record"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_number = Column(String, nullable=False, unique=True, index=True)
    status = Column(String, nullable=False, default="pending",
                    comment="pending | pushing | pushed | failed")
    arke_production_order_id = Column(String, nullable=True,
                                     comment="ID returned by Arke after creation")
    error_message = Column(Text, nullable=True)
    pushed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Arke push helpers
# ---------------------------------------------------------------------------

def get_push_status(db: Session) -> list:
    """Return all push records."""
    return db.query(ArkePushRecord).order_by(ArkePushRecord.created_at).all()


def get_push_record(db: Session, order_number: str) -> Optional[ArkePushRecord]:
    """Return the push record for a specific order, or None."""
    return db.query(ArkePushRecord).filter(
        ArkePushRecord.order_number == order_number
    ).first()


def upsert_push_record(
    db: Session,
    order_number: str,
    status: str,
    arke_id: str = None,
    error: str = None,
) -> ArkePushRecord:
    """Create or update a push record for an order."""
    record = get_push_record(db, order_number)
    if record is None:
        record = ArkePushRecord(order_number=order_number)
        db.add(record)
    record.status = status
    if arke_id:
        record.arke_production_order_id = arke_id
    if error is not None:
        record.error_message = error
    if status == "pushed":
        record.pushed_at = datetime.utcnow()
    record.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(record)
    return record


# ---------------------------------------------------------------------------
# Schedule persistence helpers
# ---------------------------------------------------------------------------

def _get_current_schedule_version(db: Session) -> int:
    """Return the latest schedule version or 0 if none exists."""
    row = db.query(ScheduleEntry.schedule_version).order_by(
        ScheduleEntry.schedule_version.desc()
    ).first()
    return row[0] if row else 0


def get_persisted_schedule(db: Session) -> list:
    """Return the latest schedule entries ordered by position."""
    version = _get_current_schedule_version(db)
    if version == 0:
        return []
    return (
        db.query(ScheduleEntry)
        .filter(ScheduleEntry.schedule_version == version)
        .order_by(ScheduleEntry.position)
        .all()
    )


def save_schedule(db: Session, plans: list, conflicts: list) -> int:
    """
    Persist a new schedule, compare with previous, log diffs.

    Args:
        db:        Database session
        plans:     List of dicts (the production_plans payload)
        conflicts: List of conflict dicts from scheduler

    Returns:
        Schedule version number (existing if unchanged, new if changed)
    """
    old_version = _get_current_schedule_version(db)

    # ── Fetch old schedule for comparison ──
    old_entries = (
        db.query(ScheduleEntry)
        .filter(ScheduleEntry.schedule_version == old_version)
        .order_by(ScheduleEntry.position)
        .all()
    ) if old_version > 0 else []

    old_by_order = {e.order_number: e for e in old_entries}
    old_order_list = [e.order_number for e in old_entries]
    new_order_list = [p["order_number"] for p in plans]

    # ── Check if schedule actually changed ──
    schedule_changed = _schedule_has_changed(old_entries, plans, old_order_list, new_order_list)

    if not schedule_changed and old_version > 0:
        logger.info(f"Schedule unchanged, staying at v{old_version}")
        return old_version

    new_version = old_version + 1

    # ── Write new entries ──
    for idx, p in enumerate(plans, start=1):
        db.add(ScheduleEntry(
            schedule_version=new_version,
            position=idx,
            order_number=p["order_number"],
            customer=p.get("customer"),
            product=p.get("product"),
            quantity=p.get("quantity"),
            priority=p.get("priority"),
            starts_at=p.get("starts_at"),
            ends_at=p.get("ends_at"),
            deadline=p.get("deadline"),
            reasoning=p.get("reasoning"),
        ))

    # ── Diff & log ──
    if old_version == 0:
        # First schedule ever
        _add_log(db, "info", "schedule", "Schedule created",
                 f"Initial schedule with {len(plans)} orders (v{new_version})",
                 schedule_version=new_version)
        # Log conflicts once for the initial schedule
        for c in conflicts:
            _add_log(db, "warning", "conflict", "EDF vs Priority conflict",
                     c.get("resolution", ""),
                     order_number=c.get("edf_first", {}).get("order"),
                     schedule_version=new_version)
    else:
        _diff_schedules(db, old_by_order, old_order_list, plans, new_order_list, new_version)
        # Only log *new* conflicts that weren't present in the previous version
        _log_new_conflicts(db, conflicts, old_version, new_version)

    db.commit()
    logger.info(f"Saved schedule v{new_version} with {len(plans)} entries")
    return new_version


def _schedule_has_changed(old_entries, new_plans, old_order_list, new_order_list):
    """Return True if the schedule differs from the previous version."""
    if len(old_entries) != len(new_plans):
        return True
    if old_order_list != new_order_list:
        return True
    # Check priorities changed
    for old_e, new_p in zip(old_entries, new_plans):
        if old_e.priority != new_p.get("priority"):
            return True
        if old_e.starts_at != new_p.get("starts_at"):
            return True
        if old_e.ends_at != new_p.get("ends_at"):
            return True
        if old_e.deadline != new_p.get("deadline"):
            return True
    return False


def _log_new_conflicts(db, conflicts, old_version, new_version):
    """Only log conflicts whose resolution text wasn't already logged."""
    # Gather resolutions already logged for the previous version
    existing = set()
    prev_logs = (
        db.query(ProductionLog.detail)
        .filter(
            ProductionLog.schedule_version == old_version,
            ProductionLog.category == "conflict",
        )
        .all()
    )
    for row in prev_logs:
        if row[0]:
            existing.add(row[0])

    for c in conflicts:
        resolution = c.get("resolution", "")
        if resolution not in existing:
            _add_log(db, "warning", "conflict", "EDF vs Priority conflict",
                     resolution,
                     order_number=c.get("edf_first", {}).get("order"),
                     schedule_version=new_version)


def _diff_schedules(db, old_by_order, old_order_list, new_plans, new_order_list, version):
    """Compare old ↔ new schedule and write change log entries."""
    new_by_order = {p["order_number"]: p for p in new_plans}

    # Orders added
    for on in new_order_list:
        if on not in old_by_order:
            _add_log(db, "change", "schedule", f"Order {on} added to schedule",
                     f"New order appeared in schedule at position {new_order_list.index(on) + 1}",
                     order_number=on, schedule_version=version)

    # Orders removed
    for on in old_order_list:
        if on not in new_by_order:
            _add_log(db, "change", "schedule", f"Order {on} removed from schedule",
                     f"Order no longer present in new schedule",
                     order_number=on, schedule_version=version)

    # Position changes & priority changes
    for idx, p in enumerate(new_plans):
        on = p["order_number"]
        if on in old_by_order:
            old = old_by_order[on]
            old_pos = old_order_list.index(on) + 1
            new_pos = idx + 1
            if old_pos != new_pos:
                direction = "up" if new_pos < old_pos else "down"
                _add_log(db, "change", "schedule",
                         f"Order {on} moved {direction}",
                         f"Position changed from #{old_pos} to #{new_pos}",
                         order_number=on, schedule_version=version)

            old_prio = old.priority
            new_prio = p.get("priority")
            if old_prio is not None and new_prio is not None and old_prio != new_prio:
                _add_log(db, "change", "schedule",
                         f"Order {on} priority changed",
                         f"Priority changed from P{old_prio} to P{new_prio}",
                         order_number=on, schedule_version=version)

    # Detect swaps (adjacent pair that swapped)
    for i in range(len(old_order_list) - 1):
        a, b = old_order_list[i], old_order_list[i + 1]
        if a in new_by_order and b in new_by_order:
            new_a = new_order_list.index(a)
            new_b = new_order_list.index(b)
            if new_b < new_a:  # b now comes before a = swap
                _add_log(db, "change", "schedule",
                         f"Orders {b} and {a} swapped",
                         f"{b} (was #{i+2}) now scheduled before {a} (was #{i+1})",
                         schedule_version=version)


def _add_log(db, level, category, title, detail=None, order_number=None, schedule_version=None):
    db.add(ProductionLog(
        level=level,
        category=category,
        title=title,
        detail=detail,
        order_number=order_number,
        schedule_version=schedule_version,
    ))


def add_production_log(db: Session, level: str, category: str, title: str,
                       detail: str = None, order_number: str = None,
                       schedule_version: int = None):
    """Public helper to append a log entry (auto-commits)."""
    _add_log(db, level, category, title, detail, order_number, schedule_version)
    db.commit()


def get_production_logs(db: Session, limit: int = 100) -> list:
    """Return recent production log entries, newest first."""
    return (
        db.query(ProductionLog)
        .order_by(ProductionLog.timestamp.desc())
        .limit(limit)
        .all()
    )


def init_db():
    """Initialize database tables (additive — never drops existing data)"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialised (create-if-not-exists)")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_production_state(db: Session, production_order_id: str) -> ProductionState:
    """
    Create or replace the production state entry.
    Only ONE record can exist - this enforces single order on line.
    
    Args:
        db: Database session
        production_order_id: Production order ID
        
    Returns:
        Created ProductionState object
    """
    # Delete any existing record (there should only be 0 or 1)
    db.query(ProductionState).delete()
    
    # Create new record
    state = ProductionState(
        active_po=production_order_id,
        op_smt_required=True,
        op_reflow_required=True,
        op_tht_required=True,
        op_aoi_required=True,
        op_test_required=True,
        op_coating_required=True,
        op_pack_required=True
    )
    db.add(state)
    db.commit()
    db.refresh(state)
    logger.info(f"Created production state for PO: {production_order_id} (single record enforced)")
    return state


def get_production_state(db: Session, production_order_id: str) -> Optional[ProductionState]:
    """
    Get production state for a specific order
    
    Args:
        db: Database session
        production_order_id: Production order ID
        
    Returns:
        ProductionState object or None
    """
    return db.query(ProductionState).filter(ProductionState.active_po == production_order_id).first()





# Canonical operation order
OPERATION_ORDER = ['smt', 'reflow', 'tht', 'aoi', 'test', 'coating', 'pack']


def finish_operation(db: Session, order_no: str, phase: str) -> ProductionState:
    """
    Finish (complete) an operation for the active order.

    Rules:
      - The order must exist on the line.
      - The phase must be required; error if not.
      - The phase must not already be finished.
      - All previous *required* phases must already be finished.
        (Non-required phases are skipped in the sequence check.)

    Args:
        db:       Database session
        order_no: Production order ID
        phase:    Operation name (smt, reflow, tht, aoi, test, coating, pack)

    Returns:
        Updated ProductionState object

    Raises:
        ValueError on any validation failure
    """
    phase = phase.lower()
    if phase not in OPERATION_ORDER:
        raise ValueError(f"Unknown operation '{phase}'. Must be one of: {', '.join(OPERATION_ORDER)}")

    state = get_production_state(db, order_no)
    if not state:
        raise ValueError(f"No active order found for '{order_no}'")

    # Phase must be required
    if not getattr(state, f"op_{phase}_required"):
        raise ValueError(f"Operation '{phase}' is not required for order {order_no}")

    # Phase must not already be finished
    if getattr(state, f"op_{phase}_finished_at") is not None:
        raise ValueError(f"Operation '{phase}' is already finished for order {order_no}")

    # All preceding *required* phases must be finished
    for prev in OPERATION_ORDER:
        if prev == phase:
            break  # reached current phase, all prior OK
        if getattr(state, f"op_{prev}_required") and getattr(state, f"op_{prev}_finished_at") is None:
            raise ValueError(
                f"Cannot finish '{phase}': previous required operation '{prev}' is not yet finished"
            )

    setattr(state, f"op_{phase}_finished_at", datetime.utcnow())
    db.commit()
    db.refresh(state)
    logger.info(f"Finished operation {phase} for PO: {order_no}")
    return state


def get_all_production_states(db: Session):
    """Get all production states"""
    return db.query(ProductionState).all()


def get_active_order_on_line(db: Session) -> Optional[ProductionState]:
    """
    Get the currently active order on the production line.
    Since only ONE record exists, return it if it has active operations.
    
    Returns:
        ProductionState object if there's an active order, None otherwise
    """
    state = db.query(ProductionState).first()
    return state  # If a record exists, there's an active order


def clear_production_line(db: Session):
    """
    Clear the production line by deleting the single production state record.
    This allows loading a new order.
    """
    db.query(ProductionState).delete()
    db.commit()
    logger.info("Production line cleared (single record deleted)")
