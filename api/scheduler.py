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
        self.product_bom_cache = {}          # product_name → total min/unit
        self.product_phases_cache = {}       # product_name → [{name, duration_per_unit}]
        self.product_id_map = {}             # extra_id / internal_id / name → UUID
    
    def load_product_bom_data(self) -> Dict[str, float]:
        """Fetch product BOM data from Arke API.

        The list endpoint (/product/product) does NOT include the BOM plan.
        We must call the detail endpoint (/product/product/{id}) for each
        product to get plan[].processes[].properties.{name, duration}.
        """
        try:
            products = self.arke.get("/product/product")
            bom_data = {}

            for product in products:
                product_name = product.get("name", "")
                product_uuid = product.get("id", "")
                internal_id = product.get("internal_id", "")
                extra_id = product.get("extra_id", "")

                # Build internal_id/extra_id → UUID lookup
                if product_uuid:
                    if internal_id:
                        self.product_id_map[internal_id] = product_uuid
                    if extra_id:
                        self.product_id_map[extra_id] = product_uuid
                    if product_name:
                        self.product_id_map[product_name] = product_uuid
                    logger.info(f"Product map: {internal_id or extra_id} → {product_uuid}")

                # ── Fetch detail to get BOM plan ──
                total_minutes = 0
                phase_list: List[Dict[str, Any]] = []
                if product_uuid:
                    try:
                        detail = self.arke.get(f"/product/product/{product_uuid}")
                        plan = detail.get("plan", [])
                        # plan is a list of step objects:
                        #   [{"operator":"and", "processes":[{"properties":{"name":"SMT","duration":24}}]}]
                        if isinstance(plan, list):
                            for step in plan:
                                for proc in (step.get("processes") or []):
                                    props = proc.get("properties", {})
                                    dur = props.get("duration", 0)
                                    name = props.get("name", "Unknown")
                                    total_minutes += dur
                                    phase_list.append({
                                        "name": name,
                                        "duration_per_unit": dur,
                                    })
                    except Exception as detail_err:
                        logger.warning(f"Could not fetch detail for {product_name}: {detail_err}")

                if total_minutes == 0:
                    logger.warning(f"No BOM data found for {product_name}, using 0 min/unit")
                else:
                    logger.info(f"Loaded BOM for {product_name}: {total_minutes} min/unit, {len(phase_list)} phases")

                bom_data[product_name] = total_minutes
                # Also key by internal_id / extra_id for lookup flexibility
                if internal_id:
                    bom_data[internal_id] = total_minutes
                if extra_id:
                    bom_data[extra_id] = total_minutes
                if phase_list:
                    self.product_phases_cache[product_name] = phase_list
                    if internal_id:
                        self.product_phases_cache[internal_id] = phase_list
                    if extra_id:
                        self.product_phases_cache[extra_id] = phase_list

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
                    
                    # Try multiple field names for the product code
                    raw_pid = (product_item.get("extra_id")
                               or product_item.get("internal_id")
                               or product_item.get("product_id")
                               or product_item.get("id")
                               or "")
                    # Resolve to real UUID via product map
                    if not self.product_id_map:
                        self.load_product_bom_data()
                    pid = self.product_id_map.get(raw_pid, raw_pid)
                    # If still not a UUID, try by product name
                    if len(pid) < 32 and product_name in self.product_id_map:
                        pid = self.product_id_map[product_name]
                    logger.info(f"Product '{product_name}': raw={raw_pid} → uuid={pid}")
                    
                    sales_orders.append(SalesOrder(
                        id=order_id,
                        order_number=order_number,
                        customer=customer,
                        product_id=pid,
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

    def get_phase_blocks(self, product_name: str, quantity: int, order_start: datetime) -> List[Dict[str, Any]]:
        """Break an order into per-phase time blocks for Gantt machine rows."""
        if not self.product_phases_cache:
            self.load_product_bom_data()

        phases = self.product_phases_cache.get(product_name, [])
        if not phases:
            return []

        blocks: List[Dict[str, Any]] = []
        cursor = order_start
        for phase in phases:
            dur_minutes = phase["duration_per_unit"] * quantity
            dur_days = dur_minutes / self.WORKING_MINUTES_PER_DAY
            phase_end = cursor + timedelta(days=dur_days)
            blocks.append({
                "name": phase["name"],
                "starts_at": cursor.isoformat(),
                "ends_at": phase_end.isoformat(),
                "duration_minutes": round(dur_minutes, 1),
            })
            cursor = phase_end
        return blocks

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
                ends_at=ends_at,
                deadline=order.deadline,
                priority=order.priority,
                customer=order.customer,
                reasoning=reasoning,
                total_minutes=total_minutes
            ))
            
            # Next order starts when this one ends
            current_start = ends_at
        
        return production_plans

    # ------------------------------------------------------------------
    # Level 2 – Group by Product
    # ------------------------------------------------------------------
    def create_grouped_schedule(self, sales_orders: List[SalesOrder]) -> List[ProductionPlan]:
        """
        Schedule that groups orders for the *same product* together.

        Within each product group:
          - Use earliest deadline across all orders in the group.
          - Sum the quantities.
          - Use the highest Arke priority (max value).

        Groups are then scheduled with EDF on the merged deadline.
        This reduces setup/changeover overhead on the production line.
        """
        from collections import defaultdict

        groups: Dict[str, List[SalesOrder]] = defaultdict(list)
        for order in sales_orders:
            groups[order.product_name].append(order)

        merged: List[SalesOrder] = []
        for product_name, orders in groups.items():
            earliest = min(orders, key=lambda o: o.deadline)
            merged.append(SalesOrder(
                id=earliest.id,
                order_number=", ".join(o.order_number for o in orders),
                customer=", ".join(dict.fromkeys(o.customer for o in orders)),
                product_id=earliest.product_id,
                product_name=product_name,
                quantity=sum(o.quantity for o in orders),
                deadline=earliest.deadline,
                priority=max(o.priority for o in orders),
                status=earliest.status,
            ))

        return self.create_edf_schedule(merged)

    # ------------------------------------------------------------------
    # Level 2 – Split Batches
    # ------------------------------------------------------------------
    MAX_BATCH_SIZE = 10  # units

    def create_split_schedule(self, sales_orders: List[SalesOrder]) -> List[ProductionPlan]:
        """
        Schedule that splits large orders into smaller batches.

        If an order exceeds MAX_BATCH_SIZE, it is broken into multiple
        sub-orders of at most MAX_BATCH_SIZE units.  Each sub-order
        keeps the original deadline and priority so it is independently
        scheduled by EDF.
        """
        split_orders: List[SalesOrder] = []
        for order in sales_orders:
            if order.quantity <= self.MAX_BATCH_SIZE:
                split_orders.append(order)
            else:
                remaining = order.quantity
                batch_num = 1
                while remaining > 0:
                    batch_qty = min(remaining, self.MAX_BATCH_SIZE)
                    split_orders.append(SalesOrder(
                        id=order.id,
                        order_number=f"{order.order_number}-B{batch_num}",
                        customer=order.customer,
                        product_id=order.product_id,
                        product_name=order.product_name,
                        quantity=batch_qty,
                        deadline=order.deadline,
                        priority=order.priority,
                        status=order.status,
                    ))
                    remaining -= batch_qty
                    batch_num += 1

        return self.create_edf_schedule(split_orders)

    # ------------------------------------------------------------------
    # Schedule dispatch – pick policy at call time
    # ------------------------------------------------------------------
    def create_schedule(self, sales_orders: List[SalesOrder],
                        policy: Optional[SchedulingPolicy] = None) -> List[ProductionPlan]:
        """Route to the right scheduler based on the active policy."""
        pol = policy or self.policy
        if pol == SchedulingPolicy.GROUP_BY_PRODUCT:
            return self.create_grouped_schedule(sales_orders)
        elif pol == SchedulingPolicy.SPLIT_BATCHES:
            return self.create_split_schedule(sales_orders)
        return self.create_edf_schedule(sales_orders)

    def detect_conflicts(self, sales_orders: List[SalesOrder]) -> List[Dict[str, Any]]:
        """
        Detect scheduling conflicts between priority-first and EDF orderings.

        Finds adjacent EDF pairs where a lower-priority order is placed
        before a higher-priority one because its deadline is tighter.
        """
        conflicts = []
        edf_sorted = sorted(sales_orders, key=lambda x: x.urgency_score)

        for i in range(len(edf_sorted) - 1):
            cur = edf_sorted[i]
            nxt = edf_sorted[i + 1]
            # lower number = higher priority; nxt has higher priority but comes later
            if nxt.priority < cur.priority:
                conflicts.append({
                    "type": "priority_vs_deadline",
                    "priority_first": {
                        "order": nxt.order_number,
                        "priority": nxt.priority,
                        "deadline": nxt.deadline.strftime("%b %-d"),
                    },
                    "edf_first": {
                        "order": cur.order_number,
                        "priority": cur.priority,
                        "deadline": cur.deadline.strftime("%b %-d"),
                    },
                    "resolution": (
                        f"{cur.order_number} (deadline {cur.deadline.strftime('%b %-d')}) "
                        f"is scheduled before {nxt.order_number} (deadline {nxt.deadline.strftime('%b %-d')}) "
                        f"despite {nxt.order_number} being P{nxt.priority} "
                        f"— EDF prioritises tighter deadlines."
                    ),
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
