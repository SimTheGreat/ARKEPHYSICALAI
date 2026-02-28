"""
Database connection and models for production state tracking
"""
import os
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, DateTime, Boolean
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


def init_db():
    """Initialize database tables - drops and recreates to apply schema changes"""
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables recreated successfully")
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
