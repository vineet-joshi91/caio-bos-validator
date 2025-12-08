# -*- coding: utf-8 -*-
"""
CMO SLM module.

- Builds CMO prompt and calls the model.
- Normalises recommendation structure so it matches other brains
  (actions_7d, actions_30d, actions_quarter, actions_half_year, actions_year, etc.).
- Attaches basic CMO visualisation specs under tools["charts"]:
    * cmo-spend-vs-outcomes
    * cmo-efficiency-metrics
  (Only if the underlying metrics are present in the packet.)
"""

import json
from typing import Dict, Any, List

from slm.core.slm_core import build_brain_prompt, call_ollama, PROMPT_SYSTEM
from slm.tools.common import ensure_recommendation_shape


def _get_cmo_metrics_view(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a metrics dict for CMO charts.

    Priority:
    - packet["cmo_metrics"]  (e.g. forwarded from cmo_tools.run()['metrics'])
    - packet["meta"]         (fallback, if metrics are embedded there)
    """
    metrics = packet.get("cmo_metrics")
    if isinstance(metrics, dict):
        return metrics

    meta = packet.get("meta")
    if isinstance(meta, dict):
        return meta

    return {}


def _add_value_safe(rows: List[Dict[str, Any]], label: str, value: Any) -> None:
    """
    Append a row with label/value if value is numeric and not None.
    """
    if value is None:
        return
    try:
        v = float(value)
    except Exception:
        return
    rows.append({"label": label, "value": v})


def _build_cmo_charts(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build chart specs for CMO based on marketing metrics.

    We assume the BOS packet may contain:
        packet["cmo_metrics"] = {
            "marketing_spend_total": ...,
            "revenue_marketing_attributed": ...,
            "cost_per_lead": ...,
            "customer_acquisition_cost": ...,
            "conversion_rate_lead_to_customer": ...,
            "marketing_roi": ...,
            "roas": ...
        }

    Charts created (only if data exists):
        - cmo-spend-vs-outcomes
        - cmo-efficiency-metrics
    """
    charts: List[Dict[str, Any]] = []

    metrics = _get_cmo_metrics_view(packet)

    spend = metrics.get("marketing_spend_total")
    rev = metrics.get("revenue_marketing_attributed")
    cpl = metrics.get("cost_per_lead")
    cac = metrics.get("customer_acquisition_cost")

    conv = metrics.get("conversion_rate_lead_to_customer")
    roi = metrics.get("marketing_roi")
    roas = metrics.get("roas")

    # -----------------------------------------
    # 1) Spend vs Outcomes (money & unit costs)
    # -----------------------------------------
    spend_rows: List[Dict[str, Any]] = []
    _add_value_safe(spend_rows, "Marketing Spend (Total)", spend)
    _add_value_safe(spend_rows, "Revenue Attributed", rev)
    _add_value_safe(spend_rows, "Cost per Lead", cpl)
    _add_value_safe(spend_rows, "Customer Acquisition Cost", cac)

    if spend_rows:
        charts.append(
            {
                "id": "cmo-spend-vs-outcomes",
                "brain": "cmo",
                "type": "bar",
                "title": "Marketing Spend vs Outcomes",
                "x": {"field": "label", "label": "Metric"},
                "y": {
                    "field": "value",
                    "label": "Amount / Unit Cost",
                    "unit": "currency",
                },
                "data": spend_rows,
            }
        )

    # -----------------------------------------
    # 2) Efficiency Metrics (percentages / ratios)
    # -----------------------------------------
    eff_rows: List[Dict[str, Any]] = []

    # Conversion rate is already in %, ROI is ratio, ROAS is ratio (we'll label them clearly)
    _add_value_safe(eff_rows, "Lead→Customer Conversion %", conv)

    if roi is not None:
        try:
            eff_rows.append({"label": "Marketing ROI (x)", "value": float(roi)})
        except Exception:
            pass

    if roas is not None:
        try:
            eff_rows.append({"label": "ROAS (x)", "value": float(roas)})
        except Exception:
            pass

    if eff_rows:
        charts.append(
            {
                "id": "cmo-efficiency-metrics",
                "brain": "cmo",
                "type": "bar",
                "title": "Marketing Efficiency Metrics",
                "x": {"field": "label", "label": "Metric"},
                "y": {"field": "value", "label": "Value / Percent", "unit": ""},
                "data": eff_rows,
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
    CMO SLM wrapper.

    - Builds CMO prompt from BOS packet
    - Calls backend model via call_ollama
    - Parses JSON / falls back if needed
    - Normalises recommendation shape
    - Attaches visualisation spec under obj["tools"]["charts"]
    """
    prompt = build_brain_prompt(packet, "cmo")
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
        # Fallback if the model does not return valid JSON
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

    # 1) Normalise recommendation structure
    ensure_recommendation_shape(obj)

    # 2) Attach / update CMO visualisation spec under obj["tools"]["charts"]
    tools: Dict[str, Any] = obj.setdefault("tools", {})
    charts = tools.setdefault("charts", [])
    existing_ids = {c.get("id") for c in charts if isinstance(c, dict)}

    for chart in _build_cmo_charts(packet):
        cid = chart.get("id")
        if cid not in existing_ids:
            charts.append(chart)

    return obj
