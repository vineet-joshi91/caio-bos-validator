# -*- coding: utf-8 -*-
"""
CMO tools module.

Computes core marketing metrics + "needs" from the BOS packet.

This mirrors the pattern used in cfo_tools.run(pkt):
- returns {"metrics": {...}, "needs": [...]}
- uses safe_div / add_need for robustness
"""

from typing import Dict, Any, List, Optional
from .common import safe_div, add_need, clip


def run(pkt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute CMO metrics from the BOS packet.

    Expects marketing-related fields in pkt["meta"], for example:
        marketing_spend_total / ad_spend_total / marketing_spend
        leads_total / leads
        customers_acquired / new_customers
        revenue_marketing_attributed / revenue_from_marketing / revenue

    Returns:
        {
            "metrics": {
                "marketing_spend_total": ...,
                "leads_total": ...,
                "customers_acquired": ...,
                "cost_per_lead": ...,
                "customer_acquisition_cost": ...,
                "conversion_rate_lead_to_customer": ...,
                "revenue_marketing_attributed": ...,
                "marketing_roi": ...,
                "roas": ...
            },
            "needs": [ ... ]
        }
    """
    needs: List[str] = []
    m: Dict[str, Any] = {}

    meta = pkt.get("meta", {}) or {}

    # --- Raw inputs (allow for multiple possible key names) ---

    marketing_spend = (
        meta.get("marketing_spend_total")
        or meta.get("ad_spend_total")
        or meta.get("marketing_spend")
    )

    leads_total = meta.get("leads_total") or meta.get("leads")
    customers_acquired = (
        meta.get("customers_acquired")
        or meta.get("new_customers")
        or meta.get("customers_from_marketing")
    )

    revenue_marketing_attributed = (
        meta.get("revenue_marketing_attributed")
        or meta.get("revenue_from_marketing")
        or meta.get("marketing_attributed_revenue")
        or meta.get("revenue")  # fallback in early MVP
    )

    # --- Basic "needs" if critical inputs are missing ---

    if marketing_spend is None:
        add_need(needs, "marketing_spend_total / ad_spend_total / marketing_spend")
    if leads_total is None:
        add_need(needs, "leads_total / leads for lead volume and CPL")
    if customers_acquired is None:
        add_need(needs, "customers_acquired / new_customers from marketing")
    if revenue_marketing_attributed is None:
        add_need(needs, "revenue_marketing_attributed / revenue_from_marketing")

    # --- Derived metrics ---

    # 1) Cost per lead (CPL)
    cost_per_lead: Optional[float] = None
    if marketing_spend is not None and leads_total not in (None, 0):
        cost_per_lead = safe_div(marketing_spend, leads_total)

    # 2) Customer Acquisition Cost (CAC)
    customer_acquisition_cost: Optional[float] = None
    if marketing_spend is not None and customers_acquired not in (None, 0):
        customer_acquisition_cost = safe_div(marketing_spend, customers_acquired)

    # 3) Conversion rate lead -> customer (%)
    conv_rate: Optional[float] = None
    if leads_total not in (None, 0) and customers_acquired is not None:
        conv_rate_raw = safe_div(customers_acquired, leads_total)
        if conv_rate_raw is not None:
            conv_rate = clip(conv_rate_raw * 100.0, 0.0, 100.0)

    # 4) Marketing ROI = (rev - spend) / spend
    marketing_roi: Optional[float] = None
    if marketing_spend not in (None, 0) and revenue_marketing_attributed is not None:
        marketing_roi = safe_div(
            revenue_marketing_attributed - marketing_spend,
            marketing_spend,
        )

    # 5) ROAS = revenue / spend
    roas: Optional[float] = None
    if marketing_spend not in (None, 0) and revenue_marketing_attributed is not None:
        roas = safe_div(revenue_marketing_attributed, marketing_spend)

    # --- Populate metrics dict (rounded + clipped where sensible) ---

    m.update(
        {
            "marketing_spend_total": marketing_spend,
            "leads_total": leads_total,
            "customers_acquired": customers_acquired,
            "cost_per_lead": None
            if cost_per_lead is None
            else round(cost_per_lead, 2),
            "customer_acquisition_cost": None
            if customer_acquisition_cost is None
            else round(customer_acquisition_cost, 2),
            "conversion_rate_lead_to_customer": None
            if conv_rate is None
            else round(conv_rate, 2),  # in %
            "revenue_marketing_attributed": revenue_marketing_attributed,
            "marketing_roi": None
            if marketing_roi is None
            else round(clip(marketing_roi, -10.0, 10.0), 4),
            "roas": None if roas is None else round(clip(roas, 0.0, 20.0), 4),
        }
    )

    return {"metrics": m, "needs": needs}
