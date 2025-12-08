# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from typing import Dict, Any, List

from slm.core.slm_core import OllamaRunner, PROMPT_SYSTEM
from slm.core.ea_core import build_ea_prompt, coerce_ea_json, ea_output_to_dict


def _normalize_per_brain(per_brain: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Make sure each brain payload is a plain dict with
    keys: plan, recommendation, confidence.
    Accepts either dicts or simple objects with attributes.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in (per_brain or {}).items():
        if isinstance(v, dict):
            out[k] = {
                "plan": v.get("plan", {}) or {},
                "recommendation": v.get("recommendation", {}) or {},
                "confidence": float(v.get("confidence", 0.7)),
            }
        else:
            # object-like fallback
            out[k] = {
                "plan": getattr(v, "plan", {}) or {},
                "recommendation": getattr(v, "recommendation", {}) or {},
                "confidence": float(getattr(v, "confidence", 0.7)),
            }
    return out


# ---------------------------------------------------------------------
# EA-LEVEL VISUALS: BUDGET VS ACTUAL (BY BRAIN) + REVENUE VS PROFIT
# ---------------------------------------------------------------------
def _safe_float(value: Any) -> float | None:
    """Best-effort numeric conversion; returns None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _guess_brain_actual_total(brain: str, pkt: Dict[str, Any]) -> float | None:
    """
    Try to infer an 'actual spend' or 'actual cost' for a given brain
    from the packet. This is intentionally conservative: if we don't
    find anything obvious, we return None and skip that row.

    You can tighten this later once your ETL consistently populates
    <brain>_metrics with specific keys.
    """
    metrics = pkt.get(f"{brain}_metrics") or {}

    # CFO: try to pick up overall expenses from CFO metrics / finance snapshot
    if brain == "cfo":
        val = (
            metrics.get("total_expenses")
            or metrics.get("total_costs")
            or metrics.get("opex_total")
        )
        if val is not None:
            return _safe_float(val)

        # fallback to a P&L snapshot if present
        finance = pkt.get("pnl_snapshot") or pkt.get("finance") or {}
        val = (
            finance.get("total_expenses")
            or finance.get("operating_expenses")
            or finance.get("total_costs")
        )
        return _safe_float(val)

    # CMO: marketing spend
    if brain == "cmo":
        val = (
            metrics.get("marketing_spend_total")
            or metrics.get("spend_total")
            or metrics.get("ad_spend_total")
        )
        return _safe_float(val)

    # CHRO: internal HR spend
    if brain == "chro":
        val = (
            metrics.get("hr_total_spend")
            or metrics.get("people_costs_total")
            or metrics.get("spend_total")
        )
        return _safe_float(val)

    # COO: operating spend (non-people, non-marketing)
    if brain == "coo":
        val = (
            metrics.get("operating_cost_total")
            or metrics.get("ops_spend_total")
            or metrics.get("spend_total")
        )
        return _safe_float(val)

    # CPO (Chief People Officer): external talent cost
    if brain == "cpo":
        val = (
            metrics.get("external_talent_cost_total")
            or metrics.get("contractor_cost_total")
            or metrics.get("outsourcing_cost_total")
            or metrics.get("spend_total")
        )
        return _safe_float(val)

    # Generic fallback
    val = metrics.get("spend_total") or metrics.get("total_cost")
    return _safe_float(val)


def _build_ea_charts(pkt: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build EA-level charts:

    1) Budget vs Actual by function (brain) using:
        pkt["budgets"][<brain>]["total_annual"] as budget
        and inferred actuals from <brain>_metrics or finance snapshot.

    2) Revenue vs Profit (if finance / P&L snapshot exists).
    """
    charts: List[Dict[str, Any]] = []

    budgets = pkt.get("budgets") or {}

    # ----------------------------------------
    # 1) Budget vs Actual by Brain / Function
    # ----------------------------------------
    grouped_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for brain, bdata in budgets.items():
        if not isinstance(bdata, dict):
            continue
        budget_total = _safe_float(
            bdata.get("total_annual") or bdata.get("total") or bdata.get("budget")
        )
        actual_total = _guess_brain_actual_total(brain, pkt)

        # Skip completely empty entries
        if budget_total is None and actual_total is None:
            continue

        label = brain.upper()

        if budget_total is not None:
            grouped_rows.append(
                {"brain": label, "kind": "Budget", "value": budget_total}
            )
        if actual_total is not None:
            grouped_rows.append(
                {"brain": label, "kind": "Actual", "value": actual_total}
            )

        if budget_total is not None and actual_total is not None:
            delta_rows.append(
                {
                    "brain": label,
                    "delta": actual_total - budget_total,
                }
            )

    if grouped_rows:
        charts.append(
            {
                "id": "ea-budget-vs-actual-by-brain",
                "brain": "ea",
                "type": "bar",
                "title": "Budget vs Actual by Function",
                "x": {"field": "brain", "label": "Function"},
                "y": {"field": "value", "label": "Amount", "unit": "currency"},
                # Allows frontend to group Budget/Actual bars per function
                "series_field": "kind",
                "data": grouped_rows,
            }
        )

    if delta_rows:
        charts.append(
            {
                "id": "ea-spend-delta-by-brain",
                "brain": "ea",
                "type": "bar",
                "title": "Spend Gaps vs Budget (Over / Under) by Function",
                "x": {"field": "brain", "label": "Function"},
                "y": {"field": "delta", "label": "Δ vs Budget", "unit": "currency"},
                "data": delta_rows,
            }
        )

    # ----------------------------------------
    # 2) Revenue vs Profit (Finance Summary)
    # ----------------------------------------
    finance = pkt.get("pnl_snapshot") or pkt.get("finance") or {}
    rev = _safe_float(
        finance.get("revenue_total")
        or finance.get("total_revenue")
        or finance.get("revenue")
    )
    profit = _safe_float(
        finance.get("net_profit")
        or finance.get("profit_after_tax")
        or finance.get("ebitda")
    )

    rev_profit_rows: List[Dict[str, Any]] = []
    if rev is not None:
        rev_profit_rows.append({"label": "Revenue", "value": rev})
    if profit is not None:
        rev_profit_rows.append({"label": "Profit", "value": profit})

    if rev_profit_rows:
        charts.append(
            {
                "id": "ea-revenue-vs-profit",
                "brain": "ea",
                "type": "bar",
                "title": "Revenue vs Profit Summary",
                "x": {"field": "label", "label": "Metric"},
                "y": {"field": "value", "label": "Amount", "unit": "currency"},
                "data": rev_profit_rows,
            }
        )

    return charts


def run(
    pkt: Dict[str, Any],
    host: str,
    per_brain: Dict[str, Any],
    model: str,
    timeout_sec: int,
    num_predict: int,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repeat_penalty: float = 1.05,
) -> Dict[str, Any]:
    """
    Executive Assistant (EA) fan-in stage.
    - Builds an EA prompt from validator packet + per-brain SLM outputs.
    - Calls Ollama via the shared runner.
    - Coerces the model text into a strict JSON shape for the UI.
    - Attaches EA-level visual specs under out["tools"]["charts"]:

        * ea-budget-vs-actual-by-brain
        * ea-spend-delta-by-brain
        * ea-revenue-vs-profit

      (Charts are only added if relevant data is present.)
    Returns a dict ready to print/dump as JSON.
    """
    # 1) Normalize inputs
    per_brain_norm = _normalize_per_brain(per_brain)

    # 2) Build prompt
    prompt = build_ea_prompt(pkt, per_brain_norm)

    # 3) Call Ollama
    runner = OllamaRunner(
        model=model,
        host=host,
        timeout_sec=timeout_sec,
        num_predict=num_predict,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    raw = runner.infer(prompt=prompt, system=PROMPT_SYSTEM)

    # 4) Coerce to stable JSON
    ea_obj = coerce_ea_json(raw)
    out = ea_output_to_dict(ea_obj)

    # 5) Attach meta for traceability
    out["_meta"] = {
        "engine": "ollama",
        "model": model,
        "bytes_in": len(prompt),
        "bytes_out": len(raw) if isinstance(raw, str) else 0,
        "confidence": out.get("confidence", 0.8),
    }

    # 6) Attach EA-level charts (budget umbrella + profit comparison)
    tools: Dict[str, Any] = out.setdefault("tools", {})
    charts = tools.setdefault("charts", [])

    existing_ids = {c.get("id") for c in charts if isinstance(c, dict)}
    for chart in _build_ea_charts(pkt):
        cid = chart.get("id")
        if cid and cid not in existing_ids:
            charts.append(chart)

    return out
