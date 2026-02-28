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
                "starts_at": plan.starts_at.isoformat(),
                "ends_at": plan.ends_at.isoformat()
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
        Schedule phases for a production order
        
        1. Call _schedule to generate phase sequence from BOM
        2. Assign concrete start/end dates to each phase
        
        Args:
            production_order_id: ID of the production order
            
        Returns:
            Scheduled phases data
        """
        try:
            logger.info(f"Scheduling phases for production order {production_order_id}")
            
            # Step 1: Generate phase sequence from BOM
            self.arke.post(f"/product/production/{production_order_id}/_schedule")
            
            # Step 2: Get the production order with phases
            production_order = self.arke.get(f"/product/production/{production_order_id}")
            
            phases = production_order.get("phases", [])
            if not phases:
                logger.warning("No phases found after scheduling")
                return production_order
            
            # Step 3: Assign start/end dates to each phase
            # Phases are sequential - each starts when previous ends
            starts_at_str = production_order.get("starts_at", "")
            if starts_at_str:
                current_start = datetime.fromisoformat(starts_at_str.replace("Z", "+00:00"))
                # Ensure timezone awareness
                if current_start.tzinfo is None:
                    current_start = current_start.replace(tzinfo=timezone.utc)
            else:
                current_start = self.scheduler.CURRENT_DATE
            
            for i, phase in enumerate(phases):
                phase_id = phase["id"]
                
                # Calculate phase duration
                duration_per_unit = phase.get("duration_per_unit", 10)  # minutes
                quantity = production_order["quantity"]
                total_minutes = duration_per_unit * quantity
                duration_days = total_minutes / self.scheduler.WORKING_MINUTES_PER_DAY
                
                phase_end = current_start + timedelta(days=duration_days)
                
                # Update phase start date
                self.arke.post(
                    f"/product/production-order-phase/{phase_id}/_update_starting_date",
                    {"starting_date": current_start.isoformat()}
                )
                
                # Update phase end date
                self.arke.post(
                    f"/product/production-order-phase/{phase_id}/_update_ending_date",
                    {"ending_date": phase_end.isoformat()}
                )
                
                logger.info(f"Phase {i+1} ({phase.get('name', 'unknown')}): {current_start.strftime('%b %d')} → {phase_end.strftime('%b %d')}")
                
                # Next phase starts when this one ends
                current_start = phase_end
            
            # Get updated production order
            updated_order = self.arke.get(f"/product/production/{production_order_id}")
            return updated_order
            
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
