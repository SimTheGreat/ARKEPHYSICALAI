from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timezone
from uuid import uuid4

from request import ArkeAPI
from scheduler import ProductionScheduler, SchedulingPolicy
from production import ProductionOrderManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ARKE Physical AI API",
    description="FastAPI wrapper for ARKE Physical AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize Arke API client, scheduler, and production manager
arke_client = ArkeAPI()
scheduler = ProductionScheduler(arke_client, policy=SchedulingPolicy.EDF)
production_manager = ProductionOrderManager(arke_client, scheduler)
STATIONS = ["SMT", "Reflow", "THT", "AOI", "Test", "Coating", "Pack"]
DETECTION_STATES = set(STATIONS + ["NOT_PRESENT"])


class PartState(BaseModel):
    part_id: str
    label: Optional[str] = None
    current_station: str = "NOT_PRESENT"
    current_station_since: Optional[str] = None
    progress_index: int = -1
    completed_stations: List[str] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated_at: str


class TransitionEvent(BaseModel):
    event_id: str
    part_id: str
    from_station: str
    to_station: str
    source: str = "manual"
    confidence: Optional[float] = None
    happened_at: str


class CreatePartRequest(BaseModel):
    part_id: Optional[str] = None
    label: Optional[str] = None


class DetectionUpdateRequest(BaseModel):
    station: str
    source: str = "vision"
    confidence: Optional[float] = None


parts_state: Dict[str, PartState] = {}
transition_events: List[TransitionEvent] = []


class ArkeRequest(BaseModel):
    endpoint: str
    method: Optional[str] = "GET"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_part_exists(part_id: str) -> PartState:
    part = parts_state.get(part_id)
    if part is None:
        raise HTTPException(status_code=404, detail=f"Part not found: {part_id}")
    return part


def update_part_station(
    part: PartState,
    to_station: str,
    source: str,
    confidence: Optional[float],
) -> Optional[TransitionEvent]:
    if to_station not in DETECTION_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid station '{to_station}'. Allowed: {sorted(DETECTION_STATES)}",
        )

    from_station = part.current_station
    if from_station == to_station:
        part.last_updated_at = utc_now_iso()
        return None

    now_iso = utc_now_iso()

    if part.history and "exited_at" not in part.history[-1]:
        part.history[-1]["exited_at"] = now_iso

    if to_station != "NOT_PRESENT":
        part.history.append({"station": to_station, "entered_at": now_iso})

    part.current_station = to_station
    part.current_station_since = None if to_station == "NOT_PRESENT" else now_iso
    part.last_updated_at = now_iso

    if to_station in STATIONS:
        station_idx = STATIONS.index(to_station)
        if station_idx > part.progress_index:
            part.progress_index = station_idx
            part.completed_stations = STATIONS[: station_idx + 1]

    event = TransitionEvent(
        event_id=f"evt_{uuid4().hex[:12]}",
        part_id=part.part_id,
        from_station=from_station,
        to_station=to_station,
        source=source,
        confidence=confidence,
        happened_at=now_iso,
    )
    transition_events.append(event)
    return event


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ARKE Physical AI API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/arke/{endpoint:path}")
async def proxy_arke_get(endpoint: str):
    """
    Proxy GET requests to Arke API
    
    Args:
        endpoint: The API endpoint path (without base URL)
    
    Returns:
        JSON response from Arke API
    """
    try:
        # Ensure endpoint starts with /
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        
        logger.info(f"Proxying GET request to: {endpoint}")
        result = arke_client.get(endpoint)
        return result
    except Exception as e:
        logger.error(f"Error proxying request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/arke/refresh-token")
async def refresh_token():
    """
    Manually refresh the Arke API token
    
    Returns:
        Status message
    """
    try:
        arke_client.login()
        return {"message": "Token refreshed successfully"}
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scheduler/orders")
async def get_sales_orders():
    """
    Get all accepted sales orders
    
    Returns:
        List of sales orders with scheduling info
    """
    try:
        logger.info("Fetching active sales orders")
        orders = arke_client.get("/sales/order/_active")
        
        sales_orders = scheduler.parse_sales_orders(orders)
        
        return {
            "count": len(sales_orders),
            "orders": [
                {
                    "id": o.id,
                    "order_number": o.order_number,
                    "customer": o.customer,
                    "product": o.product_name,
                    "quantity": o.quantity,
                    "deadline": o.deadline.strftime("%Y-%m-%d"),
                    "priority": o.priority,
                    "production_days": scheduler.calculate_production_time(o.product_name, o.quantity)[0]
                }
                for o in sales_orders
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching sales orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scheduler/schedule")
async def get_production_schedule():
    """
    Generate production schedule using EDF policy
    
    Returns:
        Production schedule with conflict detection
    """
    try:
        logger.info("Generating production schedule")
        
        orders = arke_client.get("/sales/order/_active")
        sales_orders = scheduler.parse_sales_orders(orders)
        
        # Create EDF schedule
        production_plans = scheduler.create_edf_schedule(sales_orders)
        
        # Detect conflicts
        conflicts = scheduler.detect_conflicts(sales_orders)
        
        # Generate summary
        summary = scheduler.generate_schedule_summary(production_plans, conflicts)
        
        return {
            "policy": "EDF (Earliest Deadline First)",
            "generated_at": scheduler.CURRENT_DATE.isoformat(),
            "production_plans": [
                {
                    "order_number": p.sales_order_number,
                    "customer": p.customer,
                    "product": p.product_name,
                    "quantity": p.quantity,
                    "priority": p.priority,
                    "starts_at": p.starts_at.isoformat(),
                    "ends_at": p.ends_at.isoformat(),
                    "deadline": p.deadline.isoformat(),
                    "reasoning": p.reasoning
                }
                for p in production_plans
            ],
            "conflicts": conflicts,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error generating schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scheduler/conflicts")
async def detect_scheduling_conflicts():
    """
    Detect scheduling conflicts (SO-005 vs SO-003)
    
    Returns:
        List of detected conflicts
    """
    try:
        logger.info("Detecting scheduling conflicts")
        
        orders = arke_client.get("/sales/order/_active")
        sales_orders = scheduler.parse_sales_orders(orders)
        conflicts = scheduler.detect_conflicts(sales_orders)
        
        return {
            "conflicts_found": len(conflicts),
            "conflicts": conflicts
        }
    except Exception as e:
        logger.error(f"Error detecting conflicts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/production/create")
async def create_production_orders():
    """
    Create production orders in Arke from EDF schedule
    
    Returns:
        List of created production orders
    """
    try:
        logger.info("Creating production orders")
        
        orders = arke_client.get("/sales/order/_active")
        sales_orders = scheduler.parse_sales_orders(orders)
        production_plans = scheduler.create_edf_schedule(sales_orders)
        
        # Create production orders in Arke
        results = production_manager.create_full_schedule(production_plans)
        
        return {
            "created": len(results),
            "production_orders": results
        }
    except Exception as e:
        logger.error(f"Error creating production orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/production/{production_order_id}/confirm")
async def confirm_production_order(production_order_id: str):
    """
    Confirm a production order after approval
    Moves to in_progress and unlocks first phase
    
    Args:
        production_order_id: ID of the production order to confirm
        
    Returns:
        Confirmed production order
    """
    try:
        logger.info(f"Confirming production order {production_order_id}")
        result = production_manager.confirm_production_order(production_order_id)
        return result
    except Exception as e:
        logger.error(f"Error confirming production order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/production/phase/{phase_id}/start")
async def start_phase(phase_id: str):
    """
    Start a production phase
    
    Args:
        phase_id: ID of the phase to start
        
    Returns:
        Updated phase data
    """
    try:
        logger.info(f"Starting phase {phase_id}")
        result = production_manager.start_phase(phase_id)
        return result
    except Exception as e:
        logger.error(f"Error starting phase: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/line/parts")
async def create_part(payload: CreatePartRequest):
    part_id = payload.part_id or f"PART-{uuid4().hex[:6].upper()}"
    if part_id in parts_state:
        raise HTTPException(status_code=400, detail=f"Part already exists: {part_id}")

    now_iso = utc_now_iso()
    part = PartState(
        part_id=part_id,
        label=payload.label,
        last_updated_at=now_iso,
    )
    parts_state[part_id] = part
    return {"part": part}


@app.post("/api/line/parts/{part_id}/detection")
async def update_detection(part_id: str, payload: DetectionUpdateRequest):
    part = ensure_part_exists(part_id)
    event = update_part_station(
        part=part,
        to_station=payload.station,
        source=payload.source,
        confidence=payload.confidence,
    )
    return {"part": part, "event": event}


@app.get("/api/line/state")
async def get_line_state():
    by_station: Dict[str, List[PartState]] = {station: [] for station in STATIONS}
    not_present: List[PartState] = []
    for part in parts_state.values():
        if part.current_station in STATIONS:
            by_station[part.current_station].append(part)
        else:
            not_present.append(part)

    for station in STATIONS:
        by_station[station].sort(key=lambda p: p.last_updated_at, reverse=True)
    not_present.sort(key=lambda p: p.last_updated_at, reverse=True)

    return {
        "stations": STATIONS,
        "parts": sorted(parts_state.values(), key=lambda p: p.last_updated_at, reverse=True),
        "by_station": by_station,
        "not_present": not_present,
        "recent_events": transition_events[-30:],
    }


@app.get("/api/line/events")
async def get_line_events(limit: int = 50):
    safe_limit = min(max(limit, 1), 500)
    return {"events": transition_events[-safe_limit:]}


@app.post("/api/line/reset")
async def reset_line_state():
    parts_state.clear()
    transition_events.clear()
    return {"message": "Line state reset."}


@app.post("/api/production/phase/{phase_id}/complete")
async def complete_phase(phase_id: str):
    """
    Complete a production phase
    
    Args:
        phase_id: ID of the phase to complete
        
    Returns:
        Updated phase data
    """
    try:
        logger.info(f"Completing phase {phase_id}")
        result = production_manager.complete_phase(phase_id)
        return result
    except Exception as e:
        logger.error(f"Error completing phase: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
