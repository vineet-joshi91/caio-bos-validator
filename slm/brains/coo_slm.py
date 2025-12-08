import json
from typing import Dict, Any, List

from slm.core.slm_core import build_brain_prompt, call_ollama, PROMPT_SYSTEM
from slm.tools.common import ensure_recommendation_shape


def _get_coo_context(packet: Dict[str, Any]) -> str:
    """
    Determine COO context from the packet, with a safe fallback.
    """
    ctx = packet.get("coo_context")
    if isinstance(ctx, str) and ctx.strip():
        return ctx.strip().lower()

    meta = packet.get("meta") or {}
    explicit = meta.get("coo_context")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()

    return "generic"


def _get_metrics_view(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a metrics dict for COO charts.

    Priority:
    - packet["coo_metrics"]  (if you forward coo_tools.run()['metrics'] here later)
    - packet["meta"]         (directly from upstream if present)
    """
    metrics = packet.get("coo_metrics")
    if isinstance(metrics, dict):
        return metrics
    meta = packet.get("meta")
    if isinstance(meta, dict):
        return meta
    return {}


def _add_if_available(data_rows: List[Dict[str, Any]], label: str, value: Any) -> None:
    if value is None:
        return
    try:
        v = float(value)
    except Exception:
        return
    data_rows.append({"label": label, "value": v})


def _build_coo_charts(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build chart specs for COO based on context + metrics.

    Charts (only added if data exists):
    - coo-sla-quality: SLA, on-time %, defect %, capacity %
    - coo-tat-summary: average vs p95 TAT
    - coo-bottleneck: process bottleneck (if step metrics provided)
    - context-specific chart:
        * manufacturing: yield, scrap, OEE
        * service: escalation, FCR, reopen
        * logistics: fulfillment accuracy, late shipments
        * retail: inventory turnover, stockout, shrinkage
    """
    charts: List[Dict[str, Any]] = []

    context = _get_coo_context(packet)
    metrics = _get_metrics_view(packet)

    # -----------------------------
    # 1) SLA & Quality Overview
    # -----------------------------
    sla = metrics.get("sla_compliance_pct")
    on_time = metrics.get("on_time_delivery_pct")
    defect = metrics.get("defect_rate_pct")
    capacity = metrics.get("capacity_utilization_pct")

    sla_rows: List[Dict[str, Any]] = []
    _add_if_available(sla_rows, "SLA Compliance %", sla)
    _add_if_available(sla_rows, "On-time Delivery %", on_time)
    _add_if_available(sla_rows, "Defect Rate %", defect)
    _add_if_available(sla_rows, "Capacity Utilization %", capacity)

    if sla_rows:
        charts.append(
            {
                "id": "coo-sla-quality",
                "brain": "coo",
                "type": "bar",
                "title": "COO SLA & Quality Overview",
                "x": {"field": "label", "label": "Metric"},
                "y": {"field": "value", "label": "Percent / Score", "unit": "%"},
                "data": sla_rows,
            }
        )

    # -----------------------------
    # 2) TAT Summary (avg vs p95)
    # -----------------------------
    tat_avg = metrics.get("tat_avg_days") or metrics.get("tat_days_avg")
    tat_p95 = metrics.get("tat_p95_days")

    tat_rows: List[Dict[str, Any]] = []
    _add_if_available(tat_rows, "TAT Avg (days)", tat_avg)
    _add_if_available(tat_rows, "TAT P95 (days)", tat_p95)

    if tat_rows:
        charts.append(
            {
                "id": "coo-tat-summary",
                "brain": "coo",
                "type": "bar",
                "title": "Turnaround Time Summary",
                "x": {"field": "label", "label": "Metric"},
                "y": {"field": "value", "label": "Days", "unit": "days"},
                "data": tat_rows,
            }
        )

    # -----------------------------
    # 3) Bottleneck view (if steps list is provided)
    # -----------------------------
    # Expected optional shape:
    #   packet["coo_process_steps"] = [
    #       {"step": "Intake", "avg_time_days": 1.2, "wip": 10},
    #       {"step": "QC", "avg_time_days": 3.5, "wip": 5},
    #       ...
    #   ]
    steps = packet.get("coo_process_steps")
    if isinstance(steps, list) and steps:
        step_rows: List[Dict[str, Any]] = []
        for row in steps:
            if not isinstance(row, dict):
                continue
            step_name = row.get("step") or row.get("name")
            avg_time = row.get("avg_time_days")
            if step_name is None or avg_time is None:
                continue
            try:
                v = float(avg_time)
            except Exception:
                continue
            step_rows.append({"step": str(step_name), "value": v})

        if step_rows:
            charts.append(
                {
                    "id": "coo-bottleneck-steps",
                    "brain": "coo",
                    "type": "bar",
                    "title": "Process Steps by Average TAT (Bottlenecks)",
                    "x": {"field": "step", "label": "Step"},
                    "y": {"field": "value", "label": "Avg TAT (days)", "unit": "days"},
                    "data": step_rows,
                }
            )

    # -----------------------------
    # 4) Context-specific chart
    # -----------------------------

    # Manufacturing: yield, scrap, OEE
    if context == "manufacturing":
        rows: List[Dict[str, Any]] = []
        _add_if_available(rows, "Yield %", metrics.get("yield_pct"))
        _add_if_available(rows, "Scrap Rate %", metrics.get("scrap_rate_pct"))
        _add_if_available(rows, "OEE %", metrics.get("oee"))
        if rows:
            charts.append(
                {
                    "id": "coo-manufacturing-quality",
                    "brain": "coo",
                    "type": "bar",
                    "title": "Manufacturing Quality & Efficiency",
                    "x": {"field": "label", "label": "Metric"},
                    "y": {"field": "value", "label": "Percent", "unit": "%"},
                    "data": rows,
                }
            )

    # Service: escalation, FCR, reopen
    elif context == "service":
        rows = []
        _add_if_available(rows, "Escalation Rate %", metrics.get("escalation_rate_pct"))
        _add_if_available(rows, "FCR %", metrics.get("fcr_pct"))
        _add_if_available(rows, "Reopen Rate %", metrics.get("reopen_rate_pct"))
        if rows:
            charts.append(
                {
                    "id": "coo-service-quality",
                    "brain": "coo",
                    "type": "bar",
                    "title": "Service Ops Quality Metrics",
                    "x": {"field": "label", "label": "Metric"},
                    "y": {"field": "value", "label": "Percent", "unit": "%"},
                    "data": rows,
                }
            )

    # Logistics: fulfillment accuracy, late shipments
    elif context == "logistics":
        rows = []
        _add_if_available(
            rows, "Fulfillment Accuracy %", metrics.get("fulfillment_accuracy_pct")
        )
        _add_if_available(
            rows, "Late Shipments %", metrics.get("late_shipments_pct")
        )
        if rows:
            charts.append(
                {
                    "id": "coo-logistics-performance",
                    "brain": "coo",
                    "type": "bar",
                    "title": "Logistics Performance Overview",
                    "x": {"field": "label", "label": "Metric"},
                    "y": {"field": "value", "label": "Percent", "unit": "%"},
                    "data": rows,
                }
            )

    # Retail: inventory turnover, stockout, shrinkage
    elif context == "retail":
        rows = []
        # Turnover is not % but we can still show it in same chart; unit clarified in title
        inv_turnover = metrics.get("inventory_turnover")
        if inv_turnover is not None:
            try:
                v = float(inv_turnover)
                rows.append({"label": "Inventory Turnover (x)", "value": v})
            except Exception:
                pass
        _add_if_available(
            rows, "Stockout Rate %", metrics.get("stockout_rate_pct")
        )
        _add_if_available(rows, "Shrinkage %", metrics.get("shrinkage_pct"))
        if rows:
            charts.append(
                {
                    "id": "coo-retail-ops",
                    "brain": "coo",
                    "type": "bar",
                    "title": "Retail / D2C Ops Metrics",
                    "x": {"field": "label", "label": "Metric"},
                    "y": {"field": "value", "label": "Value / Percent", "unit": ""},
                    "data": rows,
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
    COO SLM wrapper.

    - Builds COO prompt from BOS packet
    - Calls backend model via call_ollama
    - Parses JSON / falls back if needed
    - Normalises recommendation shape
    - Attaches visualisation spec under obj["tools"]["charts"]
    """
    prompt = build_brain_prompt(packet, "coo")
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

    # 1) Normalise recommendation structure (adds actions_quarter, actions_half_year, actions_year, etc.)
    ensure_recommendation_shape(obj)

    # 2) Attach / update COO visualisation spec under obj["tools"]["charts"]
    tools: Dict[str, Any] = obj.setdefault("tools", {})
    charts = tools.setdefault("charts", [])

    existing_ids = {c.get("id") for c in charts if isinstance(c, dict)}

    for chart in _build_coo_charts(packet):
        cid = chart.get("id")
        if cid not in existing_ids:
            charts.append(chart)

    return obj
