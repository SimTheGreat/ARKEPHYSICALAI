"""
Production order creation and phase scheduling module
"""
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import logging

from request import ArkeAPI
from scheduler import ProductionPlan, ProductionScheduler

logger = logging.getLogger(__name__)


class ProductionOrderManager:
    """
    Manages creation and scheduling of production orders in Arke
    """
    
    def __init__(self, arke_client: ArkeAPI, scheduler: ProductionScheduler):
        self.arke = arke_client
        self.scheduler = scheduler
    
    def create_production_order(self, plan: ProductionPlan) -> Dict[str, Any]:
        """
        Create a production order in Arke
        
        Args:
            plan: Production plan from scheduler
            
        Returns:
            Created production order data
        """
        try:
            payload = {
                "product_id": plan.product_id,
                "quantity": plan.quantity,
                "starts_at": plan.starts_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ends_at": plan.ends_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            logger.info(f"Creating production order for {plan.sales_order_number}: {payload}")
            result = self.arke.put("/product/production", payload)
            logger.info(f"Created production order: {result.get('id', 'unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create production order for {plan.sales_order_number}: {e}")
            raise
    
    def schedule_phases(self, production_order_id: str) -> Dict[str, Any]:
        """
        Schedule phases for a production order.

        Calls _schedule which tells Arke to generate the phase sequence
        from the product BOM and automatically assign phase dates within
        the production order's starts_at / ends_at window.

        Args:
            production_order_id: ID of the production order

        Returns:
            Production order data with scheduled phases
        """
        try:
            logger.info(f"Scheduling phases for production order {production_order_id}")

            # Arke generates phases + dates from the product BOM
            self.arke.post(f"/product/production/{production_order_id}/_schedule")

            # Fetch the result so we can log / return it
            production_order = self.arke.get(f"/product/production/{production_order_id}")

            phases = production_order.get("phases", [])
            for i, phase in enumerate(phases):
                phase_name = phase.get("phase", {}).get("name", f"phase-{i+1}")
                logger.info(
                    f"Phase {i+1} ({phase_name}): "
                    f"{phase.get('starts_at', '?')} → {phase.get('ends_at', '?')} "
                    f"[{phase.get('status', '?')}]"
                )

            logger.info(f"Scheduled {len(phases)} phases for PO {production_order_id}")
            return production_order

        except Exception as e:
            logger.error(f"Failed to schedule phases: {str(e)}")
            raise

    def assign_phase_dates(self, production_order: Dict[str, Any], plan: 'ProductionPlan') -> None:
        """
        Assign concrete start/end dates to each phase based on BOM durations.

        After _schedule generates the phases, their dates default to the
        production order window.  This method walks through each phase,
        computes the real time-window (duration_per_unit × quantity → days)
        and calls _update_starting_date / _update_ending_date on each.

        Args:
            production_order: Result of GET /product/production/{id}
            plan:             The ProductionPlan with starts_at & product_name
        """
        phases = production_order.get("phases", [])
        if not phases:
            logger.warning("No phases found on production order — skipping date assignment")
            return

        # Get per-phase durations from the BOM cache
        phase_blocks = self.scheduler.get_phase_blocks(
            plan.product_name, plan.quantity, plan.starts_at
        )
        if not phase_blocks:
            logger.warning(f"No BOM phase data for {plan.product_name} — skipping date assignment")
            return

        # Match phases to blocks by sequence position (both are ordered)
        for i, arke_phase in enumerate(phases):
            if i >= len(phase_blocks):
                break

            block = phase_blocks[i]
            phase_id = arke_phase.get("id")
            phase_name = arke_phase.get("phase", {}).get("name", f"phase-{i+1}")

            if not phase_id:
                continue

            try:
                # Set start date
                self.arke.post(
                    f"/product/production-order-phase/{phase_id}/_update_starting_date",
                    {"value": block["starts_at"]},
                )
                # Set end date
                self.arke.post(
                    f"/product/production-order-phase/{phase_id}/_update_ending_date",
                    {"value": block["ends_at"]},
                )
                logger.info(
                    f"  Phase {phase_name}: dates set → "
                    f"{block['starts_at']} – {block['ends_at']} ({block['duration_minutes']} min)"
                )
            except Exception as phase_err:
                logger.warning(
                    f"  Phase {phase_name}: date assignment failed — {phase_err}"
                )
    
    def create_full_schedule(self, production_plans: List[ProductionPlan]) -> List[Dict[str, Any]]:
        """
        Create all production orders and schedule their phases
        
        Args:
            production_plans: List of production plans from scheduler
            
        Returns:
            List of created and scheduled production orders
        """
        created_orders = []
        
        for plan in production_plans:
            try:
                # Create production order
                production_order = self.create_production_order(plan)
                production_order_id = production_order["id"]
                
                # Schedule phases
                scheduled_order = self.schedule_phases(production_order_id)

                # Assign concrete per-phase dates from BOM
                self.assign_phase_dates(scheduled_order, plan)

                created_orders.append(scheduled_order)
                
            except Exception as e:
                logger.error(f"Failed to create/schedule production order for {plan.sales_order_number}: {e}")
                continue
        
        return created_orders
    
    def confirm_production_order(self, production_order_id: str) -> Dict[str, Any]:
        """
        Confirm a production order and move to in_progress
        
        Args:
            production_order_id: ID of the production order
            
        Returns:
            Confirmed production order data
        """
        try:
            logger.info(f"Confirming production order {production_order_id}")
            result = self.arke.post(f"/product/production/{production_order_id}/_confirm")
            return result
        except Exception as e:
            logger.error(f"Failed to confirm production order: {str(e)}")
            raise
    
    def start_phase(self, phase_id: str) -> Dict[str, Any]:
        """
        Start a production phase
        
        Args:
            phase_id: ID of the phase to start
            
        Returns:
            Updated phase data
        """
        try:
            logger.info(f"Starting phase {phase_id}")
            result = self.arke.post(f"/product/production-order-phase/{phase_id}/_start")
            return result
        except Exception as e:
            logger.error(f"Failed to start phase: {str(e)}")
            raise
    
    def complete_phase(self, phase_id: str) -> Dict[str, Any]:
        """
        Complete a production phase
        
        Args:
            phase_id: ID of the phase to complete
            
        Returns:
            Updated phase data
        """
        try:
            logger.info(f"Completing phase {phase_id}")
            result = self.arke.post(f"/product/production-order-phase/{phase_id}/_complete")
            return result
        except Exception as e:
            logger.error(f"Failed to complete phase: {str(e)}")
            raise
