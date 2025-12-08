# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:16:48 2025

@author: Vineet
"""

from typing import Dict, Any, List, Optional
from .common import safe_div, add_need, as_months, clip


def run(pkt: Dict[str, Any]) -> Dict[str, Any]:
    """Returns {'metrics': {...}, 'needs': [...]}"""
    needs: List[str] = []
    m: Dict[str, Any] = {}  # metrics

    meta = pkt.get("meta", {})             # optional if you later pass raw tables
    tables = pkt.get("tables", {}) or {}   # optional (unused for now)

    # --- Basic financials (allow None; we’ll request needs) ---
    cash = meta.get("cash_end") or meta.get("cash")  # allow your pipeline to populate later
    monthly_burn = meta.get("monthly_burn")          # positive number
    revenue = meta.get("revenue")
    cogs = meta.get("cogs")
    net_income = meta.get("net_income")
    debt = meta.get("total_debt")
    equity = meta.get("equity")

    # Runway
    runway_m: Optional[float] = None
    if cash is None:
        add_need(needs, "cash_end (closing cash balance)")
    if monthly_burn is None:
        add_need(needs, "monthly_burn (absolute)")

    if cash is not None and monthly_burn and monthly_burn > 0:
        runway_m = cash / monthly_burn

    # Gross margin (ratio, later we’ll also expose pct)
    gm: Optional[float] = None
    if revenue is None or cogs is None:
        add_need(needs, "revenue, cogs for gross_margin")
    else:
        gm = safe_div((revenue - cogs), revenue)

    # Leverage & current ratio (light)
    current_assets = meta.get("current_assets")
    current_liabilities = meta.get("current_liabilities")
    current_ratio = (
        safe_div(current_assets, current_liabilities)
        if current_assets is not None and current_liabilities is not None
        else None
    )

    # Debt-to-equity
    d_e: Optional[float] = None
    if debt is not None and equity not in (None, 0):
        d_e = safe_div(debt, equity)
    else:
        add_need(needs, "total_debt, equity for D/E")

    # DSO/DPO (needs receivables/payables & sales/cogs)
    ar = meta.get("accounts_receivable")
    ap = meta.get("accounts_payable")
    sales_per_day = safe_div(revenue, 365.0) if revenue is not None else None
    cogs_per_day = safe_div(cogs, 365.0) if cogs is not None else None
    dso = safe_div(ar, sales_per_day) if ar is not None and sales_per_day else None
    dpo = safe_div(ap, cogs_per_day) if ap is not None and cogs_per_day else None

    # Clamp a couple of signals to sane ranges so models don't go wild
    m.update(
        {
            "runway_months": None if runway_m is None else round(runway_m, 2),
            "gross_margin": None if gm is None else round(clip(gm, -1.0, 1.0), 4),
            "current_ratio": None if current_ratio is None else round(
                clip(current_ratio, 0.0, 10.0), 3
            ),
            "debt_to_equity": None if d_e is None else round(
                clip(d_e, 0.0, 10.0), 3
            ),
            "dso_days": None if dso is None else round(dso, 1),
            "dpo_days": None if dpo is None else round(dpo, 1),
        }
    )

    # ------------------------------------------------------------------
    # Headline CFO KPIs you requested
    # ------------------------------------------------------------------

    # 1) Revenue
    if revenue is None:
        add_need(needs, "revenue (for revenue_amount metric)")
    revenue_amount = revenue

    # 2) Gross margin in percentage
    gross_margin_pct: Optional[float] = None
    if gm is not None:
        gross_margin_pct = round(clip(gm * 100.0, -100.0, 100.0), 2)

    # 3) Operating expenses
    operating_expenses = (
        meta.get("operating_expenses")
        or meta.get("opex")
    )
    if operating_expenses is None:
        add_need(needs, "operating_expenses / opex for Operating Expenses metric")

    # 4) Net profit
    net_profit = net_income
    if net_profit is None:
        add_need(needs, "net_income for Net Profit metric")

    # 5) Cashflow
    cashflow = meta.get("operating_cashflow") or meta.get("cashflow")
    if cashflow is None:
        add_need(needs, "operating_cashflow / cashflow for Cashflow metric")

    # 6) Burn rate (monthly)
    burn_rate = monthly_burn
    if burn_rate is None:
        add_need(needs, "monthly_burn for Burn Rate metric")

    # 7) Debt ratio (debt / (debt + equity))
    debt_ratio: Optional[float] = None
    if debt is not None and equity is not None:
        total_cap = debt + equity
        if total_cap > 0:
            debt_ratio = safe_div(debt, total_cap)
    if debt_ratio is None:
        add_need(needs, "total_debt, equity for Debt Ratio metric")

    m.update(
        {
            "revenue_amount": revenue_amount,
            "gross_margin_pct": gross_margin_pct,
            "operating_expenses": operating_expenses,
            "net_profit": net_profit,
            "cashflow": cashflow,
            "burn_rate": burn_rate,
            "debt_ratio": None
            if debt_ratio is None
            else round(clip(debt_ratio, 0.0, 1.0), 4),
        }
    )

    return {"metrics": m, "needs": needs}
