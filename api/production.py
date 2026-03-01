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
