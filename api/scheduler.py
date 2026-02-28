from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    """Scheduling policy options"""
    EDF = "earliest_deadline_first"  # Level 1 - Required
    GROUP_BY_PRODUCT = "group_by_product"  # Level 2 - Optional
    SPLIT_BATCHES = "split_batches"  # Level 2 - Optional


@dataclass
class SalesOrder:
    """Sales order data structure"""
    id: str
    order_number: str
    customer: str
    product_id: str
    product_name: str
    quantity: int
    deadline: datetime
    priority: int  # 1 = highest
    status: str
    
    @property
    def urgency_score(self) -> tuple:
        """Returns (deadline, priority) for sorting - deadline takes precedence"""
        return (self.deadline, self.priority)


@dataclass
class ProductionPlan:
    """Production order plan"""
    sales_order_id: str
    sales_order_number: str
    product_id: str
    product_name: str
    quantity: int
    starts_at: datetime
    ends_at: datetime
    priority: int
    customer: str
    reasoning: str


class ProductionScheduler:
    """
    Production scheduler implementing EDF and other policies
    """
    
    # Factory constraints
    WORKING_MINUTES_PER_DAY = 480  # 8 hours
    CURRENT_DATE = datetime(2026, 2, 28, 8, 0, 0)  # Feb 28, 2026 08:00 AM
    
    # Product BOM data (minutes per unit)
    PRODUCT_BOM = {
        "PCB-IND-100": 147,  # SMT(30)+Reflow(15)+THT(45)+AOI(12)+Test(30)+Coating(9)+Pack(6)
        "MED-300": 279,      # SMT(45)+Reflow(30)+THT(60)+AOI(30)+Test(90)+Coating(15)+Pack(9)
        "IOT-200": 63,       # SMT(18)+Reflow(12)+AOI(9)+Test(18)+Pack(6)
        "AGR-400": 144,      # SMT(30)+Reflow(15)+THT(30)+AOI(12)+Test(45)+Coating(12)
        "PCB-PWR-500": 75,   # SMT(24)+Reflow(12)+AOI(9)+Test(24)+Pack(6)
    }
    
    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.EDF):
        self.policy = policy
        
    def parse_sales_orders(self, orders_data: List[Dict[str, Any]]) -> List[SalesOrder]:
        """Parse raw API data into SalesOrder objects"""
        sales_orders = []
        
        for order in orders_data:
            # Parse the order - handle various possible field names
            sales_orders.append(SalesOrder(
                id=order.get("id", ""),
                order_number=order.get("order_number", order.get("number", "")),
                customer=order.get("customer", {}).get("name", "Unknown") if isinstance(order.get("customer"), dict) else order.get("customer", "Unknown"),
                product_id=order.get("product_id", ""),
                product_name=order.get("product", {}).get("name", "") if isinstance(order.get("product"), dict) else "",
                quantity=order.get("quantity", 0),
                deadline=self._parse_datetime(order.get("expected_shipping_time", order.get("deadline", ""))),
                priority=order.get("priority", 3),
                status=order.get("status", "")
            ))
        
        return sales_orders
    
    def _parse_datetime(self, dt_string: str) -> datetime:
        """Parse datetime string from API"""
        if not dt_string:
            return self.CURRENT_DATE + timedelta(days=30)
        
        try:
            # Handle ISO format
            if "T" in dt_string:
                return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
            # Handle date-only format
            return datetime.strptime(dt_string, "%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Failed to parse datetime {dt_string}: {e}")
            return self.CURRENT_DATE + timedelta(days=30)
    
    def calculate_production_time(self, product_name: str, quantity: int) -> float:
        """Calculate production time in working days"""
        minutes_per_unit = self.PRODUCT_BOM.get(product_name, 100)
        total_minutes = minutes_per_unit * quantity
        working_days = total_minutes / self.WORKING_MINUTES_PER_DAY
        return working_days
    
    def create_edf_schedule(self, sales_orders: List[SalesOrder]) -> List[ProductionPlan]:
        """
        Create production schedule using EDF (Earliest Deadline First)
        Level 1 - Required: One production order per sales order, sorted by deadline
        """
        # Sort by deadline first, then priority (EDF prioritizes deadline over priority)
        sorted_orders = sorted(sales_orders, key=lambda x: x.urgency_score)
        
        production_plans = []
        current_start = self.CURRENT_DATE
        
        for order in sorted_orders:
            # Calculate production time needed
            production_days = self.calculate_production_time(order.product_name, order.quantity)
            ends_at = current_start + timedelta(days=production_days)
            
            # Check if we can meet the deadline
            reasoning = f"EDF: Deadline {order.deadline.strftime('%b %d')}, Priority P{order.priority}"
            if ends_at > order.deadline:
                reasoning += f" ⚠️ LATE: Finishes {ends_at.strftime('%b %d')}"
            
            production_plans.append(ProductionPlan(
                sales_order_id=order.id,
                sales_order_number=order.order_number,
                product_id=order.product_id,
                product_name=order.product_name,
                quantity=order.quantity,
                starts_at=current_start,
                ends_at=min(ends_at, order.deadline),  # Target the deadline
                priority=order.priority,
                customer=order.customer,
                reasoning=reasoning
            ))
            
            # Next order starts when this one ends
            current_start = ends_at
        
        return production_plans
    
    def detect_conflicts(self, sales_orders: List[SalesOrder]) -> List[Dict[str, Any]]:
        """
        Detect scheduling conflicts - especially the SO-005 vs SO-003 case
        SO-005: P1 priority, deadline Mar 8
        SO-003: P2 priority, deadline Mar 4
        EDF should schedule SO-003 first despite lower priority
        """
        conflicts = []
        
        # Sort by priority
        priority_sorted = sorted(sales_orders, key=lambda x: x.priority)
        # Sort by EDF
        edf_sorted = sorted(sales_orders, key=lambda x: x.urgency_score)
        
        # Find cases where priority-first differs from EDF
        for i in range(len(sales_orders)):
            if priority_sorted[i].id != edf_sorted[i].id:
                # Found a difference
                priority_order = priority_sorted[i]
                edf_order = edf_sorted[i]
                
                conflicts.append({
                    "type": "priority_vs_deadline",
                    "priority_first": {
                        "order": priority_order.order_number,
                        "priority": priority_order.priority,
                        "deadline": priority_order.deadline.strftime("%b %d"),
                    },
                    "edf_first": {
                        "order": edf_order.order_number,
                        "priority": edf_order.priority,
                        "deadline": edf_order.deadline.strftime("%b %d"),
                    },
                    "resolution": f"EDF schedules {edf_order.order_number} (deadline {edf_order.deadline.strftime('%b %d')}) before {priority_order.order_number} (deadline {priority_order.deadline.strftime('%b %d')}) to meet tighter deadline, despite lower priority.",
                })
        
        return conflicts
    
    def generate_schedule_summary(self, production_plans: List[ProductionPlan], conflicts: List[Dict]) -> str:
        """Generate human-readable schedule summary for messaging"""
        summary = "📋 PRODUCTION SCHEDULE - EDF (Earliest Deadline First)\n"
        summary += f"Generated: {self.CURRENT_DATE.strftime('%b %d, %Y %H:%M')}\n"
        summary += "=" * 60 + "\n\n"
        
        for i, plan in enumerate(production_plans, 1):
            summary += f"{i}. {plan.sales_order_number} - {plan.product_name}\n"
            summary += f"   Customer: {plan.customer}\n"
            summary += f"   Quantity: {plan.quantity} units\n"
            summary += f"   Priority: P{plan.priority}\n"
            summary += f"   Schedule: {plan.starts_at.strftime('%b %d')} → {plan.ends_at.strftime('%b %d')}\n"
            summary += f"   Reasoning: {plan.reasoning}\n\n"
        
        if conflicts:
            summary += "\n⚠️ SCHEDULING CONFLICTS DETECTED:\n"
            summary += "=" * 60 + "\n"
            for conflict in conflicts:
                summary += f"\n{conflict.get('resolution', 'Unknown conflict')}\n"
        
        summary += "\n" + "=" * 60 + "\n"
        summary += "Please review and approve this schedule.\n"
        
        return summary
