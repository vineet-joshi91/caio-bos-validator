# -*- coding: utf-8 -*-
"""
COO tools module.

Hybrid / auto-detect COO layer that:
- detects operational context (manufacturing, service, logistics, retail, generic)
- computes core ops metrics
- computes context-specific metrics
- returns {"context": ..., "metrics": {...}, "needs": [...]}
"""

from typing import Dict, Any, List, Optional
from .common import safe_div, add_need, clip


def _detect_context(meta: Dict[str, Any]) -> str:
    """
    Detect COO context based on available fields.

    Priority order:
    - explicit meta["coo_context"]
    - manufacturing signals
    - service ops signals
    - logistics/fulfillment signals
    - retail/D2C signals
    - fallback: "generic"
    """
    explicit = meta.get("coo_context")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()

    keys = {str(k).lower() for k in meta.keys()}

    # Manufacturing / production signals
    manuf_signals = [
        "oee",
        "scrap_rate_pct",
        "yield_pct",
        "machine_downtime_hours",
        "production_volume",
        "wip_items",
        "work_in_progress",
    ]
    if any(k in keys for k in manuf_signals):
        return "manufacturing"

    # Service ops / ticketing signals
    service_signals = [
        "ticket_count",
        "sla_breaches",
        "fcr_pct",
        "backlog_age_days",
        "reopen_rate_pct",
        "tickets_open",
        "tickets_closed",
    ]
    if any(k in keys for k in service_signals):
        return "service"

    # Logistics / fulfillment signals
    logistics_signals = [
        "shipments",
        "on_time_delivery_pct",
        "fulfillment_accuracy_pct",
        "late_shipments_pct",
        "warehouse_orders",
    ]
    if any(k in keys for k in logistics_signals):
        return "logistics"

    # Retail / D2C signals
    retail_signals = [
        "inventory_turnover",
        "stockout_rate_pct",
        "shrinkage_pct",
        "picking_accuracy_pct",
        "store_orders",
    ]
    if any(k in keys for k in retail_signals):
        return "retail"

    return "generic"


def _core_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Compute core COO metrics that make sense across multiple contexts.
    """
    m: Dict[str, Any] = {}

    tat_avg = meta.get("tat_avg_days") or meta.get("tat_days_avg")
    tat_p95 = meta.get("tat_p95_days")
    sla_compliance_pct = meta.get("sla_compliance_pct")
    on_time_delivery_pct = meta.get("on_time_delivery_pct")

    defect_rate_pct = meta.get("defect_rate_pct")
    capacity_utilization_pct = meta.get("capacity_utilization_pct")
    operating_cost_per_unit = meta.get("operating_cost_per_unit")

    backlog_size = meta.get("backlog_size") or meta.get("open_items")
    wip_items = meta.get("wip_items") or meta.get("work_in_progress")

    bottleneck_step = meta.get("bottleneck_step")
    bleeding_areas = meta.get("bleeding_areas")

    # Needs
    if tat_avg is None:
        add_need(needs, "tat_avg_days / tat_days_avg for TAT")
    if sla_compliance_pct is None:
        add_need(needs, "sla_compliance_pct for SLA performance")
    if defect_rate_pct is None:
        add_need(needs, "defect_rate_pct for quality tracking")
    if capacity_utilization_pct is None:
        add_need(needs, "capacity_utilization_pct for capacity usage")
    if operating_cost_per_unit is None:
        add_need(needs, "operating_cost_per_unit for cost efficiency")

    # Metrics
    m.update(
        {
            "tat_avg_days": tat_avg,
            "tat_p95_days": tat_p95,
            "sla_compliance_pct": None
            if sla_compliance_pct is None
            else round(clip(float(sla_compliance_pct), 0.0, 100.0), 2),
            "on_time_delivery_pct": None
            if on_time_delivery_pct is None
            else round(clip(float(on_time_delivery_pct), 0.0, 100.0), 2),
            "defect_rate_pct": None
            if defect_rate_pct is None
            else round(clip(float(defect_rate_pct), 0.0, 100.0), 2),
            "capacity_utilization_pct": None
            if capacity_utilization_pct is None
            else round(clip(float(capacity_utilization_pct), 0.0, 150.0), 2),
            "operating_cost_per_unit": operating_cost_per_unit,
            "backlog_size": backlog_size,
            "wip_items": wip_items,
            "bottleneck_step": bottleneck_step,
            "bleeding_areas": bleeding_areas
            if isinstance(bleeding_areas, list)
            else None,
        }
    )

    return m


def _manufacturing_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Manufacturing / production specific COO metrics.
    """
    m: Dict[str, Any] = {}

    yield_pct = meta.get("yield_pct")
    scrap_rate_pct = meta.get("scrap_rate_pct")
    oee = meta.get("oee")
    downtime_hours = meta.get("machine_downtime_hours")

    if yield_pct is None:
        add_need(needs, "yield_pct for production yield")
    if scrap_rate_pct is None:
        add_need(needs, "scrap_rate_pct for scrap/waste tracking")
    if oee is None:
        add_need(needs, "oee for overall equipment effectiveness")

    m.update(
        {
            "yield_pct": None
            if yield_pct is None
            else round(clip(float(yield_pct), 0.0, 100.0), 2),
            "scrap_rate_pct": None
            if scrap_rate_pct is None
            else round(clip(float(scrap_rate_pct), 0.0, 100.0), 2),
            "oee": None if oee is None else round(clip(float(oee), 0.0, 100.0), 2),
            "machine_downtime_hours": downtime_hours,
        }
    )
    return m


def _service_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Service / ticketing / support specific COO metrics.
    """
    m: Dict[str, Any] = {}

    escalation_rate_pct = meta.get("escalation_rate_pct")
    fcr_pct = meta.get("fcr_pct")  # First Contact Resolution
    backlog_age_days = meta.get("backlog_age_days")
    reopen_rate_pct = meta.get("reopen_rate_pct")

    tickets_open = meta.get("tickets_open")
    tickets_closed = meta.get("tickets_closed")

    if escalation_rate_pct is None:
        add_need(needs, "escalation_rate_pct for service escalation tracking")
    if fcr_pct is None:
        add_need(needs, "fcr_pct for first contact resolution quality")
    if backlog_age_days is None:
        add_need(needs, "backlog_age_days for aging analysis")

    m.update(
        {
            "escalation_rate_pct": None
            if escalation_rate_pct is None
            else round(clip(float(escalation_rate_pct), 0.0, 100.0), 2),
            "fcr_pct": None
            if fcr_pct is None
            else round(clip(float(fcr_pct), 0.0, 100.0), 2),
            "backlog_age_days": backlog_age_days,
            "reopen_rate_pct": None
            if reopen_rate_pct is None
            else round(clip(float(reopen_rate_pct), 0.0, 100.0), 2),
            "tickets_open": tickets_open,
            "tickets_closed": tickets_closed,
        }
    )
    return m


def _logistics_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Logistics / fulfillment specific COO metrics.
    """
    m: Dict[str, Any] = {}

    fulfillment_accuracy_pct = meta.get("fulfillment_accuracy_pct")
    shipments_per_day = meta.get("shipments_per_day")
    late_shipments_pct = meta.get("late_shipments_pct")

    if fulfillment_accuracy_pct is None:
        add_need(needs, "fulfillment_accuracy_pct for order accuracy")
    if late_shipments_pct is None:
        add_need(needs, "late_shipments_pct for delivery delays")

    m.update(
        {
            "fulfillment_accuracy_pct": None
            if fulfillment_accuracy_pct is None
            else round(clip(float(fulfillment_accuracy_pct), 0.0, 100.0), 2),
            "shipments_per_day": shipments_per_day,
            "late_shipments_pct": None
            if late_shipments_pct is None
            else round(clip(float(late_shipments_pct), 0.0, 100.0), 2),
        }
    )
    return m


def _retail_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Retail / D2C specific COO metrics.
    """
    m: Dict[str, Any] = {}

    inventory_turnover = meta.get("inventory_turnover")
    stockout_rate_pct = meta.get("stockout_rate_pct")
    shrinkage_pct = meta.get("shrinkage_pct")
    picking_accuracy_pct = meta.get("picking_accuracy_pct")

    if inventory_turnover is None:
        add_need(needs, "inventory_turnover for stock efficiency")
    if stockout_rate_pct is None:
        add_need(needs, "stockout_rate_pct for availability risk")
    if shrinkage_pct is None:
        add_need(needs, "shrinkage_pct for loss/leakage tracking")

    m.update(
        {
            "inventory_turnover": inventory_turnover,
            "stockout_rate_pct": None
            if stockout_rate_pct is None
            else round(clip(float(stockout_rate_pct), 0.0, 100.0), 2),
            "shrinkage_pct": None
            if shrinkage_pct is None
            else round(clip(float(shrinkage_pct), 0.0, 100.0), 2),
            "picking_accuracy_pct": None
            if picking_accuracy_pct is None
            else round(clip(float(picking_accuracy_pct), 0.0, 100.0), 2),
        }
    )
    return m


def run(pkt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for COO tools.

    Returns:
        {
            "context": "<manufacturing|service|logistics|retail|generic>",
            "metrics": {...},
            "needs": [...]
        }
    """
    needs: List[str] = []
    metrics: Dict[str, Any] = {}

    meta = pkt.get("meta", {}) or {}

    context = _detect_context(meta)

    # Core metrics (common across contexts)
    metrics.update(_core_metrics(meta, needs))

    # Context-specific metrics
    if context == "manufacturing":
        metrics.update(_manufacturing_metrics(meta, needs))
    elif context == "service":
        metrics.update(_service_metrics(meta, needs))
    elif context == "logistics":
        metrics.update(_logistics_metrics(meta, needs))
    elif context == "retail":
        metrics.update(_retail_metrics(meta, needs))
    else:
        # generic: nothing extra, but we still return core metrics + needs
        pass

    return {
        "context": context,
        "metrics": metrics,
        "needs": needs,
    }
