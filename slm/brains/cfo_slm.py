import json
from typing import Dict, Any, List

from slm.core.slm_core import build_brain_prompt, call_ollama, PROMPT_SYSTEM
from slm.tools.common import ensure_recommendation_shape


def _add_timeseries_chart(
    charts: List[Dict[str, Any]],
    packet: Dict[str, Any],
    series_key: str,
    chart_id: str,
    title: str,
    y_label: str,
    unit: str,
) -> None:
    """
    Helper: add a line chart from a time series in the packet.

    Expected shape:
        packet[series_key] = [
            {"period": "2025-01", "value": 123.4},
            ...
        ]
    """
    series = packet.get(series_key)
    if not isinstance(series, list) or not series:
        return

    data_rows: List[Dict[str, Any]] = []
    for row in series:
        if not isinstance(row, dict):
            continue
        period = row.get("period") or row.get("month") or row.get("date")
        val = row.get("value")
        if period is None or val is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        data_rows.append({"period": str(period), "value": v})

    if not data_rows:
        return

    charts.append(
        {
            "id": chart_id,
            "brain": "cfo",
            "type": "line",
            "title": title,
            "x": {"field": "period", "label": "Period"},
            "y": {"field": "value", "label": y_label, "unit": unit},
            "data": data_rows,
        }
    )


def _build_cfo_charts(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a list of chart specs for the CFO brain from the BOS packet.

    Currently covered:
    - BOS Index (bar)
    - BOS Dimensional Scores (bar) if 'bos_subscores' present
    - Revenue (line)
    - Gross Margin (line)
    - Operating Expenses (line)
    - Net Profit (line)
    - Cashflow (line)
    - Burn Rate (line)
    - Debt Ratio (gauge + optional line series)
    """
    charts: List[Dict[str, Any]] = []

    # 1) BOS Index as a single bar
    bos_index_raw = packet.get("bos_index", 0.0)
    try:
        bos_index = float(bos_index_raw) if bos_index_raw is not None else 0.0
    except Exception:
        bos_index = 0.0

    charts.append(
        {
            "id": "cfo-bos-index",
            "brain": "cfo",
            "type": "bar",
            "title": "Current BOS Index (CFO)",
            "x": {"field": "label", "label": "Metric"},
            "y": {"field": "value", "label": "Score", "unit": "index"},
            "data": [
                {"label": "BOS Index", "value": bos_index},
            ],
        }
    )

    # 2) Optional BOS dimension scores
    # Expected shape: packet["bos_subscores"] = {"Liquidity": 0.8, "Profitability": 0.7, ...}
    subscores = packet.get("bos_subscores")
    if isinstance(subscores, dict) and subscores:
        data_rows: List[Dict[str, Any]] = []
        for name, value in subscores.items():
            try:
                v = float(value)
            except Exception:
                continue
            data_rows.append({"dimension": str(name), "value": v})

        if data_rows:
            charts.append(
                {
                    "id": "cfo-bos-dimensions",
                    "brain": "cfo",
                    "type": "bar",
                    "title": "BOS Dimensional Scores (CFO)",
                    "x": {"field": "dimension", "label": "Dimension"},
                    "y": {"field": "value", "label": "Score", "unit": "index"},
                    "data": data_rows,
                }
            )

    # 3) Time-series metrics (line charts)

    _add_timeseries_chart(
        charts,
        packet,
        series_key="cfo_revenue_series",
        chart_id="cfo-revenue-trend",
        title="Revenue Trend",
        y_label="Revenue",
        unit="INR",
    )

    _add_timeseries_chart(
        charts,
        packet,
        series_key="cfo_gross_margin_series",
        chart_id="cfo-gross-margin-trend",
        title="Gross Margin Trend",
        y_label="Gross Margin",
        unit="%",
    )

    _add_timeseries_chart(
        charts,
        packet,
        series_key="cfo_opex_series",
        chart_id="cfo-opex-trend",
        title="Operating Expenses Trend",
        y_label="Operating Expenses",
        unit="INR",
    )

    _add_timeseries_chart(
        charts,
        packet,
        series_key="cfo_net_profit_series",
        chart_id="cfo-net-profit-trend",
        title="Net Profit Trend",
        y_label="Net Profit",
        unit="INR",
    )

    _add_timeseries_chart(
        charts,
        packet,
        series_key="cfo_cashflow_series",
        chart_id="cfo-cashflow-trend",
        title="Cashflow Trend",
        y_label="Cashflow",
        unit="INR",
    )

    _add_timeseries_chart(
        charts,
        packet,
        series_key="cfo_burn_rate_series",
        chart_id="cfo-burn-rate-trend",
        title="Burn Rate Trend",
        y_label="Burn Rate",
        unit="INR",
    )

    # 4) Debt Ratio – gauge + optional line series

    debt_ratio_raw = packet.get("cfo_debt_ratio")
    try:
        debt_ratio = float(debt_ratio_raw) if debt_ratio_raw is not None else None
    except Exception:
        debt_ratio = None

    if debt_ratio is not None:
        # Gauge/progress-style chart for latest debt ratio
        charts.append(
            {
                "id": "cfo-debt-ratio",
                "brain": "cfo",
                "type": "gauge",  # frontend can map 'gauge' to a radial/progress chart
                "title": "Debt Ratio (Latest)",
                "x": {"field": "label", "label": "Metric"},
                "y": {"field": "value", "label": "Debt Ratio", "unit": "%"},
                "data": [
                    {"label": "Debt Ratio", "value": debt_ratio},
                ],
            }
        )

    # Optional time-series for debt ratio if available
    _add_timeseries_chart(
        charts,
        packet,
        series_key="cfo_debt_ratio_series",
        chart_id="cfo-debt-ratio-trend",
        title="Debt Ratio Trend",
        y_label="Debt Ratio",
        unit="%",
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
    prompt = build_brain_prompt(packet, "cfo")
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

    # 2) Attach / update CFO visualisation spec under obj["tools"]["charts"]
    tools: Dict[str, Any] = obj.setdefault("tools", {})
    charts = tools.setdefault("charts", [])

    # Avoid duplicating charts if run() is called multiple times
    existing_ids = {c.get("id") for c in charts if isinstance(c, dict)}

    for chart in _build_cfo_charts(packet):
        chart_id = chart.get("id")
        if chart_id not in existing_ids:
            charts.append(chart)

    return obj
