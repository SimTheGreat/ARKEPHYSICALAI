from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
import logging

from request import ArkeAPI
from scheduler import ProductionScheduler, SchedulingPolicy
from production import ProductionOrderManager
from database import init_db, get_db, ProductionState, create_production_state, get_production_state, finish_operation, get_all_production_states, get_active_order_on_line, clear_production_line, OPERATION_ORDER

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

# Initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    logger.info("Database initialized")

# Initialize Arke API client, scheduler, and production manager
arke_client = ArkeAPI()
scheduler = ProductionScheduler(arke_client, policy=SchedulingPolicy.EDF)
production_manager = ProductionOrderManager(arke_client, scheduler)


class ArkeRequest(BaseModel):
    endpoint: str
    method: Optional[str] = "GET"


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


# Production State Endpoints

@app.get("/api/production-state")
async def get_all_states(db: Session = Depends(get_db)):
    """
    Get all production states
    
    Returns:
        List of all production states
    """
    try:
        states = get_all_production_states(db)
        return {
            "count": len(states),
            "states": [
                {
                    "production_order": state.active_po,
                    "smt": {
                        "required": state.op_smt_required,
                        "finished_at": state.op_smt_finished_at.isoformat() if state.op_smt_finished_at else None
                    },
                    "reflow": {
                        "required": state.op_reflow_required,
                        "finished_at": state.op_reflow_finished_at.isoformat() if state.op_reflow_finished_at else None
                    },
                    "tht": {
                        "required": state.op_tht_required,
                        "finished_at": state.op_tht_finished_at.isoformat() if state.op_tht_finished_at else None
                    },
                    "aoi": {
                        "required": state.op_aoi_required,
                        "finished_at": state.op_aoi_finished_at.isoformat() if state.op_aoi_finished_at else None
                    },
                    "test": {
                        "required": state.op_test_required,
                        "finished_at": state.op_test_finished_at.isoformat() if state.op_test_finished_at else None
                    },
                    "coating": {
                        "required": state.op_coating_required,
                        "finished_at": state.op_coating_finished_at.isoformat() if state.op_coating_finished_at else None
                    },
                    "pack": {
                        "required": state.op_pack_required,
                        "finished_at": state.op_pack_finished_at.isoformat() if state.op_pack_finished_at else None
                    }
                }
                for state in states
            ]
        }
    except Exception as e:
        logger.error(f"Error getting production states: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# IMPORTANT: More specific routes MUST come before general :path routes
# Otherwise FastAPI will match the general route first and never reach the specific ones

@app.post("/api/production-state/{production_order_id:path}/operation/{operation}/start")
async def start_operation(production_order_id: str, operation: str, db: Session = Depends(get_db)):
    """No longer needed - operations are set as required when order is loaded"""
    return {"message": "Operations are now set as required at load time. Use /complete to mark finished."}


@app.post("/api/production-state/{production_order_id:path}/operation/{operation}/complete")
async def complete_operation(production_order_id: str, operation: str, db: Session = Depends(get_db)):
    """
    Finish an operation for the active order.
    Enforces sequential order and required-checks.
    """
    try:
        state = finish_operation(db, production_order_id, operation)
        return {"message": f"Operation {operation} completed for {production_order_id}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error completing operation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# General routes with :path must come AFTER specific routes to avoid conflicts

@app.post("/api/production-state/{production_order_id:path}")
async def create_state(production_order_id: str, db: Session = Depends(get_db)):
    """
    Load an order onto the production line.
    ENFORCES SINGLE ORDER: Only ONE record exists in the table at any time.
    Loading a new order overwrites the existing record.
    
    Args:
        production_order_id: Production order ID
        
    Returns:
        Created production state
    """
    try:
        # Get current order on line (if any)
        existing_state = db.query(ProductionState).first()
        replaced_order = existing_state.active_po if existing_state else None
        
        # Create new state (this will delete existing record and create new one)
        state = create_production_state(db, production_order_id)
        
        return {
            "message": f"Production state created for {production_order_id}",
            "production_order": state.active_po,
            "replaced": replaced_order
        }
    except Exception as e:
        logger.error(f"Error creating production state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/production-state/{production_order_id:path}")
async def get_state(production_order_id: str, db: Session = Depends(get_db)):
    """
    Get production state for a specific order
    
    Args:
        production_order_id: Production order ID
        
    Returns:
        Production state
    """
    try:
        state = get_production_state(db, production_order_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Production state not found for {production_order_id}")
        
        return {
            "production_order": state.active_po,
            "smt": {
                "required": state.op_smt_required,
                "finished_at": state.op_smt_finished_at.isoformat() if state.op_smt_finished_at else None
            },
            "reflow": {
                "required": state.op_reflow_required,
                "finished_at": state.op_reflow_finished_at.isoformat() if state.op_reflow_finished_at else None
            },
            "tht": {
                "required": state.op_tht_required,
                "finished_at": state.op_tht_finished_at.isoformat() if state.op_tht_finished_at else None
            },
            "aoi": {
                "required": state.op_aoi_required,
                "finished_at": state.op_aoi_finished_at.isoformat() if state.op_aoi_finished_at else None
            },
            "test": {
                "required": state.op_test_required,
                "finished_at": state.op_test_finished_at.isoformat() if state.op_test_finished_at else None
            },
            "coating": {
                "required": state.op_coating_required,
                "finished_at": state.op_coating_finished_at.isoformat() if state.op_coating_finished_at else None
            },
            "pack": {
                "required": state.op_pack_required,
                "finished_at": state.op_pack_finished_at.isoformat() if state.op_pack_finished_at else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting production state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
