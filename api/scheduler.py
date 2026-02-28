from datetime import datetime, timedelta, timezone
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
    deadline: datetime
    priority: int
    customer: str
    reasoning: str
    total_minutes: float = 0.0


class ProductionScheduler:
    """
    Production scheduler implementing EDF and other policies
    """
    
    # Factory constraints
    WORKING_MINUTES_PER_DAY = 480  # 8 hours
    CURRENT_DATE = datetime(2026, 2, 28, 8, 0, 0, tzinfo=timezone.utc)  # Feb 28, 2026 08:00 AM UTC
    
    def __init__(self, arke_client, policy: SchedulingPolicy = SchedulingPolicy.EDF):
        self.policy = policy
        self.arke = arke_client
        self.product_bom_cache = {}
    
    def load_product_bom_data(self) -> Dict[str, float]:
        """Fetch product BOM data from Arke API"""
        try:
            products = self.arke.get("/product/product")
            bom_data = {}
            
            for product in products:
                product_name = product.get("name", "")
                plan = product.get("plan", {})
                
                # Handle different plan structures
                total_minutes = 0
                if isinstance(plan, dict):
                    phases = plan.get("phases", [])
                    if isinstance(phases, list):
                        for phase in phases:
                            if isinstance(phase, dict):
                                total_minutes += phase.get("duration_per_unit", 0)
                    # Also try direct duration field
                    elif plan.get("total_duration"):
                        total_minutes = plan["total_duration"]
                
                if total_minutes == 0:
                    logger.warning(f"No BOM data found for {product_name}, using 0 min/unit")
                else:
                    logger.info(f"Loaded BOM for {product_name}: {total_minutes} min/unit")
                
                bom_data[product_name] = total_minutes
            
            self.product_bom_cache = bom_data
            return bom_data
            
        except Exception as e:
            logger.error(f"Failed to load product BOM data: {e}")
            raise
        
    def parse_sales_orders(self, orders_data: List[Dict[str, Any]]) -> List[SalesOrder]:
        """Parse raw API data into SalesOrder objects"""
        sales_orders = []
        
        for order in orders_data:
            try:
                customer = "Unknown"
                if isinstance(order.get("customer_attr"), dict):
                    customer = order["customer_attr"].get("name", "Unknown")
                
                order_number = order.get("internal_id", "")
                order_id = order.get("id", "")
                
                deadline_str = order.get("expected_shipping_time", "")
                deadline = self._parse_datetime(deadline_str)
                
                # Map Arke API priority (5=highest) to standard priority (1=highest)
                api_priority = order.get("priority", 3)
                priority = 6 - api_priority  # 5→1, 4→2, 3→3, 2→4, 1→5
                status = order.get("status", "")
                
                products = order.get("products", [])
                
                if not products:
                    logger.warning(f"No products found for order {order_number}")
                    continue
                
                for product_item in products:
                    product_name = product_item.get("name", "")
                    quantity = product_item.get("quantity", 0)
                    
                    if not product_name or quantity == 0:
                        continue
                    
                    sales_orders.append(SalesOrder(
                        id=order_id,
                        order_number=order_number,
                        customer=customer,
                        product_id=product_item.get("extra_id", ""),
                        product_name=product_name,
                        quantity=quantity,
                        deadline=deadline,
                        priority=priority,
                        status=status
                    ))
                
            except Exception as e:
                logger.error(f"Failed to parse order: {e}")
                continue
        
        return sales_orders
    
    def _parse_datetime(self, dt_string: str) -> datetime:
        """Parse datetime string from API"""
        if not dt_string:
            return self.CURRENT_DATE + timedelta(days=30)
        
        try:
            # Handle ISO format
            if "T" in dt_string:
                dt = datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
                # Ensure it has timezone info
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            # Handle date-only format - add time and timezone
            dt = datetime.strptime(dt_string, "%Y-%m-%d")
            return dt.replace(hour=8, minute=0, second=0, tzinfo=timezone.utc)
        except Exception as e:
            logger.warning(f"Failed to parse datetime {dt_string}: {e}")
            return self.CURRENT_DATE + timedelta(days=30)
    
    def calculate_production_time(self, product_name: str, quantity: int) -> tuple:
        """
        Calculate production time in working days and total minutes
        Returns: (working_days, total_minutes)
        """
        if not self.product_bom_cache:
            self.load_product_bom_data()
        
        minutes_per_unit = self.product_bom_cache.get(product_name, 100)
        total_minutes = minutes_per_unit * quantity
        working_days = total_minutes / self.WORKING_MINUTES_PER_DAY
        
        return working_days, total_minutes
    
    def create_edf_schedule(self, sales_orders: List[SalesOrder]) -> List[ProductionPlan]:
        """
        Create production schedule using EDF (Earliest Deadline First)
        """
        # Sort by deadline first, then priority (EDF prioritizes deadline over priority)
        sorted_orders = sorted(sales_orders, key=lambda x: x.urgency_score)
        
        production_plans = []
        current_start = self.CURRENT_DATE
        
        for order in sorted_orders:
            # Calculate production time needed
            production_days, total_minutes = self.calculate_production_time(
                order.product_name, 
                order.quantity
            )
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
                deadline=order.deadline,
                priority=order.priority,
                customer=order.customer,
                reasoning=reasoning,
                total_minutes=total_minutes
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
        
        # Sort by priority, then deadline
        priority_sorted = sorted(sales_orders, key=lambda x: (x.priority, x.deadline))
        # Sort by EDF (deadline, then priority)
        edf_sorted = sorted(sales_orders, key=lambda x: x.urgency_score)
        
        # Find adjacent pairs where a lower-priority order comes before a higher-priority order in EDF
        for i in range(len(edf_sorted) - 1):
            current_order = edf_sorted[i]
            next_order = edf_sorted[i + 1]
            
            # Check if next order has higher priority (lower number = higher priority)
            # but comes after in EDF schedule because current has earlier deadline
            if next_order.priority < current_order.priority:
                conflicts.append({
                    "type": "priority_vs_deadline",
                    "priority_first": {
                        "order": next_order.order_number,
                        "priority": next_order.priority,
                        "deadline": next_order.deadline.strftime("%b %d"),
                    },
                    "edf_first": {
                        "order": current_order.order_number,
                        "priority": current_order.priority,
                        "deadline": current_order.deadline.strftime("%b %d"),
                    },
                    "resolution": f"EDF schedules {current_order.order_number} (deadline {current_order.deadline.strftime('%b %d')}, P{current_order.priority}) before {next_order.order_number} (deadline {next_order.deadline.strftime('%b %d')}, P{next_order.priority}) to meet tighter deadline, despite lower priority.",
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
            summary += f"   Quantity: {plan.quantity} units ({plan.total_minutes:.0f} min)\n"
            summary += f"   Priority: P{plan.priority}\n"
            summary += f"   Schedule: {plan.starts_at.strftime('%b %d %H:%M')} → {plan.ends_at.strftime('%b %d %H:%M')}\n"
            summary += f"   Reasoning: {plan.reasoning}\n\n"
        
        if conflicts:
            summary += "\n⚠️ SCHEDULING CONFLICTS DETECTED:\n"
            summary += "=" * 60 + "\n"
            for conflict in conflicts:
                summary += f"\n{conflict.get('resolution', 'Unknown conflict')}\n"
        
        summary += "\n" + "=" * 60 + "\n"
        summary += "Please review and approve this schedule.\n"
        
        return summary
