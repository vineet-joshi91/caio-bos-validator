# -*- coding: utf-8 -*-
"""
CHRO SLM module.

- Builds CHRO prompt and calls the model.
- Normalises recommendation structure.
- Attaches visualisation specs (charts) for:
  * HR Budget vs Actual by Program
  * HR Spend Gaps (Over / Under Budget) by Program
  * Optional: Overall HR Budget vs Actual (if totals are present)
"""

import json
from typing import Dict, Any, List

from slm.core.slm_core import build_brain_prompt, call_ollama, PROMPT_SYSTEM
from slm.tools.common import ensure_recommendation_shape


def _get_chro_metrics_view(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a metrics dict for CHRO charts.

    Priority:
    - packet["chro_metrics"]  (e.g. forwarded from chro_tools.run()['metrics'])
    - packet["meta"]          (fallback, if metrics are embedded there)
    """
    metrics = packet.get("chro_metrics")
    if isinstance(metrics, dict):
        return metrics

    meta = packet.get("meta")
    if isinstance(meta, dict):
        return meta

    return {}


def _add_value_safe(d: Dict[str, Any], key: str) -> float | None:
    """
    Safely get a numeric value from a dict for the given key.
    Returns None if missing or non-numeric.
    """
    val = d.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _build_chro_charts(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build chart specs for CHRO based on budgets + actuals.

    We assume the BOS packet may contain:
        packet["budgets"]["chro"] = {
            "total_annual": <number, optional>,
            "by_program": {
                "Hiring": <budget>,
                "L&D": <budget>,
                "Engagement / Wellness": <budget>,
                "Compliance & HR Ops": <budget>,
                ...
            }
        }

        packet["chro_metrics"]["spend_by_program"] = {
            "Hiring": <actual>,
            "L&D": <actual>,
            ...
        }

        optional overall actual:
            packet["chro_metrics"]["hr_total_spend"]

    Charts created (only if data exists):
        - chro-budget-vs-actual-programs (grouped bar via series_field="kind")
        - chro-spend-delta-programs (bar, delta vs budget)
        - chro-budget-vs-actual-total (simple bar for total HR budget vs total actual)
    """
    charts: List[Dict[str, Any]] = []

    budgets_root = packet.get("budgets") or {}
    chro_budget = budgets_root.get("chro") or {}
    budgets_by_program = chro_budget.get("by_program") or {}
    total_budget = chro_budget.get("total_annual")

    metrics = _get_chro_metrics_view(packet)
    actual_by_program = metrics.get("spend_by_program") or {}
    total_actual = (
        metrics.get("hr_total_spend")
        or metrics.get("total_hr_spend")
        or metrics.get("spend_total")
    )

    # --------------------------------------------------
    # 1) HR Budget vs Actual by Program (spend map base)
    # --------------------------------------------------
    program_names = sorted(
        set(str(k) for k in budgets_by_program.keys())
        | set(str(k) for k in actual_by_program.keys())
    )

    grouped_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for program in program_names:
        b_val = _add_value_safe(budgets_by_program, program)
        a_val = _add_value_safe(actual_by_program, program)

        # Only create rows if we have at least one side
        if b_val is None and a_val is None:
            continue

        # Grouped bar rows (Budget vs Actual)
        if b_val is not None:
            grouped_rows.append(
                {"program": program, "kind": "Budget", "value": b_val}
            )
        if a_val is not None:
            grouped_rows.append(
                {"program": program, "kind": "Actual", "value": a_val}
            )

        # Delta rows (Actual - Budget) where both exist
        if b_val is not None and a_val is not None:
            delta_rows.append(
                {
                    "program": program,
                    "delta": a_val - b_val,
                }
            )

    if grouped_rows:
        charts.append(
            {
                "id": "chro-budget-vs-actual-programs",
                "brain": "chro",
                "type": "bar",
                "title": "HR Budget vs Actual by Program",
                "x": {"field": "program", "label": "Program"},
                "y": {"field": "value", "label": "Amount", "unit": "INR"},
                # series_field indicates how frontend can group bars
                "series_field": "kind",  # "Budget" vs "Actual"
                "data": grouped_rows,
            }
        )

    if delta_rows:
        charts.append(
            {
                "id": "chro-spend-delta-programs",
                "brain": "chro",
                "type": "bar",
                "title": "HR Spend Gaps vs Budget (Over / Under)",
                "x": {"field": "program", "label": "Program"},
                "y": {"field": "delta", "label": "Δ vs Budget", "unit": "INR"},
                "data": delta_rows,
            }
        )

    # --------------------------------------------------
    # 2) Overall HR Budget vs Actual (summary)
    # --------------------------------------------------
    total_rows: List[Dict[str, Any]] = []
    if total_budget is not None:
        try:
            total_b = float(total_budget)
            total_rows.append({"label": "Budget", "value": total_b})
        except Exception:
            pass
    if total_actual is not None:
        try:
            total_a = float(total_actual)
            total_rows.append({"label": "Actual", "value": total_a})
        except Exception:
            pass

    if total_rows:
        charts.append(
            {
                "id": "chro-budget-vs-actual-total",
                "brain": "chro",
                "type": "bar",
                "title": "Total HR Budget vs Actual Spend",
                "x": {"field": "label", "label": "Category"},
                "y": {"field": "value", "label": "Amount", "unit": "INR"},
                "data": total_rows,
            }
        )

    return charts


def run(
    packet: Dict[str, Any],
    host: str,
    model: str,
    timeout_sec: int,
    num_predict: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
) -> Dict[str, Any]:
    """
    CHRO SLM wrapper.

    - Builds CHRO prompt from BOS packet
    - Calls backend model via call_ollama
    - Parses JSON / falls back if needed
    - Normalises recommendation shape
    - Attaches visualisation spec under obj["tools"]["charts"]
    """
    prompt = build_brain_prompt(packet, "chro")
    resp_text = call_ollama(
        host,
        model,
        prompt,
        timeout_sec,
        num_predict,
        temperature,
        top_p,
        repeat_penalty,
        system=PROMPT_SYSTEM,
    )

    try:
        obj = json.loads(resp_text)
    except Exception:
        obj = {
            "plan": {
                "assumptions": [],
                "priorities": [],
                "queries_to_run": [],
                "data_gaps": [],
            },
            "recommendation": {
                "summary": "Unstructured output",
                "actions_7d": [],
                "actions_30d": [],
                "kpis_to_watch": [],
                "risks": [],
                "forecast_note": "",
            },
            "confidence": 0.5,
            "_meta": {"model": model, "engine": "ollama", "confidence": 0.5},
            "raw_text": resp_text,
        }

    # Ensure metadata exists
    obj.setdefault(
        "_meta",
        {"model": model, "engine": "ollama", "confidence": obj.get("confidence", 0.7)},
    )

    # 1) Normalise recommendation structure (adds actions_quarter, actions_half_year, actions_year, etc.)
    ensure_recommendation_shape(obj)

    # 2) Attach / update CHRO visualisation spec under obj["tools"]["charts"]
    tools: Dict[str, Any] = obj.setdefault("tools", {})
    charts = tools.setdefault("charts", [])

    existing_ids = {c.get("id") for c in charts if isinstance(c, dict)}

    for chart in _build_chro_charts(packet):
        cid = chart.get("id")
        if cid not in existing_ids:
            charts.append(chart)

    return obj
