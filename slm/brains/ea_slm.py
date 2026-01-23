# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from typing import Dict, Any, List

import re

from slm.core.slm_core import OllamaRunner, PROMPT_SYSTEM
from slm.core.ea_core import build_ea_prompt, build_ea_doc_prompt, coerce_ea_json, ea_output_to_dict

def _is_empty_ea_obj(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return True
    if (d.get("executive_summary") or "").strip():
        return False
    for k in ["top_priorities", "key_risks", "cross_brain_actions_7d", "cross_brain_actions_30d"]:
        v = d.get(k)
        if isinstance(v, list) and len(v) > 0:
            return False
    om = d.get("owner_matrix")
    if isinstance(om, dict) and any(isinstance(v, list) and v for v in om.values()):
        return False
    return True


def _fallback_nonempty_ea() -> Dict[str, Any]:
    return {
        "executive_summary": "The model returned an empty plan. This is a safe fallback. Re-run after strengthening evidence extraction or prompts.",
        "top_priorities": [
            "Extract key facts (pricing, deliverables, timelines) from the document",
            "Define success KPIs and reporting cadence",
            "Assign owners and dependencies",
        ],
        "key_risks": [
            "Empty model output (Evidence: model returned blank fields)",
            "Insufficient evidence in excerpt (Evidence: missing or unclear document details)",
        ],
        "cross_brain_actions_7d": [
            "CFO: Confirm commercial terms and budget ceiling (Evidence: document terms)",
            "CMO: Convert deliverables into a 30-day content calendar (Evidence: listed deliverables)",
            "COO: Define workflow + approvals + cadence (Evidence: execution requirement)",
            "CHRO: Assign roles/owners and capacity plan (Evidence: resourcing implied)",
            "CPO: Vendor/SLA checklist for external deliverables (Evidence: quotation/proposal)",
        ],
        "cross_brain_actions_30d": [
            "CFO: Define ROI model and tracking (Evidence: revenue/lead outcomes)",
            "CMO: Launch content pipeline and measure engagement baseline (Evidence: content scope)",
            "COO: Implement weekly execution review (Evidence: timeline requirement)",
            "CHRO: Define accountability + incentives (Evidence: execution governance)",
            "CPO: Finalize vendor milestones and acceptance criteria (Evidence: quotation scope)",
        ],
        "owner_matrix": {
            "CFO": ["Confirm terms + ROI model"],
            "CMO": ["Build content calendar + KPI baseline"],
            "COO": ["Execution cadence + operational workflow"],
            "CHRO": ["Resourcing + accountability"],
            "CPO": ["Vendor milestones + acceptance criteria"],
        },
        "confidence": 0.4,
        "tools": {"charts": []},
    }
    

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



def _extract_facts_from_doc(text: str) -> Dict[str, Any]:
    t = text or ""

    # money + percentages
    money = []

    # ₹ amounts (with commas)
    money += re.findall(r"(₹\s*\d[\d,]*(?:\.\d+)?)", t)
    
    # Rs / INR formats
    money += re.findall(r"((?:Rs\.?|INR)\s*\d[\d,]*(?:\.\d+)?)", t, flags=re.IGNORECASE)
    
    # numbers that look like prices (e.g., 130000, 1,30,000) near keywords
    money += re.findall(r"(\d[\d,]{2,})\s*(?:per\s*month|/month|monthly|pm)", t, flags=re.IGNORECASE)
    
    # normalize / dedupe
    money = [m.strip() for m in money]
    money = list(dict.fromkeys(money))[:8]

    perc = re.findall(r"(\d{1,3}\s?%)(?!\w)", t)
    dates = re.findall(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", t)

    # quick deliverable keywords (customize later)
    deliverable_phrases = [
    "podcast", "vodcast", "masterclass", "reels", "shorts",
    "long-form", "long form", "youtube", "instagram", "linkedin",
    "case study", "webinar", "newsletter"
    ]
    deliverable_hits = []
    for kw in deliverable_phrases:
        if re.search(rf"\b{re.escape(kw)}\b", t, flags=re.IGNORECASE):
            deliverable_hits.append(kw)
    
    deliverables = list(dict.fromkeys(deliverable_hits))[:12]


    # take first few meaningful lines as context bullets
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    sample_lines = lines[:12]

    return {
        "money": list(dict.fromkeys(money))[:5],
        "percent": list(dict.fromkeys(perc))[:5],
        "dates": list(dict.fromkeys(dates))[:5],
        "deliverables": list(dict.fromkeys(deliverable_hits))[:12],
        "sample_lines": sample_lines,
    }


def _fallback_from_doc(doc_text: str) -> Dict[str, Any]:
    facts = _extract_facts_from_doc(doc_text)
    money = ", ".join(facts["money"]) if facts["money"] else "pricing not found"
    perc = ", ".join(facts["percent"]) if facts["percent"] else "percent terms not found"
    dels = ", ".join(facts["deliverables"]) if facts["deliverables"] else "deliverables unclear"
    date = facts["dates"][0] if facts["dates"] else "date not found"

    evidence = f"Evidence: {money}; {perc}; deliverables: {dels}; date: {date}"
    preview = " | ".join(facts.get("sample_lines", [])[:4])
    return {
        
        "executive_summary": (
            f"Document-first plan generated via deterministic extraction because the model returned an empty schema. "
            f"This proposal centers on organic content-led growth with defined deliverables and commercial terms. ({evidence})"
        ),
        "top_priorities": [
            f"Confirm commercial terms and scope ({evidence})",
            f"Convert deliverables into a 30-day production calendar (Evidence: deliverables: {dels})",
            "Define KPI baseline + tracking (Evidence: proposal references engagement/conversion goals)",
            "Set governance: owners, cadence, approvals (Evidence: multi-deliverable execution)",
        ],
        "key_risks": [
            "Missing baseline metrics for ROI (Evidence: no CAC/lead baseline in doc text excerpt)",
            "Scope ambiguity (Evidence: deliverables listed but acceptance criteria unclear)",
            "Attribution risk for revenue share terms (Evidence: percent terms detected but attribution rules not specified)",
        ],
        "cross_brain_actions_7d": [
            f"CFO: Confirm pricing + payment cadence + revenue share logic ({evidence})",
            f"CMO: Draft 30-day content calendar from deliverables (Evidence: {dels})",
            "COO: Define workflow: ideation → production → approvals → publishing",
            "CHRO: Identify internal owners + time allocation for execution",
            "CPO: Vendor/SLA checklist + milestone acceptance criteria",
        ],
        "cross_brain_actions_30d": [
            "CFO: Build ROI + attribution model; agree reporting cadence",
            "CMO: Launch first content sprint; baseline engagement + leads",
            "COO: Operationalize weekly review and backlog grooming",
            "CHRO: Accountability + performance expectations for owners",
            "CPO: Lock vendor milestones and enforce quality gates",
        ],
        "owner_matrix": {
            "CFO": ["Terms, ROI model, attribution rules"],
            "CMO": ["Content calendar, launch, KPI baseline"],
            "COO": ["Workflow, cadence, delivery operations"],
            "CHRO": ["Owners, capacity, accountability"],
            "CPO": ["Vendor milestones, acceptance criteria"],
        },
        "confidence": 0.55,
        "tools": {"charts": []},
    }


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
    per_brain_norm = _normalize_per_brain(per_brain)

    # Decide EA mode:
    # - If document_text exists (Upload & Analyze), always use document-first EA prompt.
    # - Otherwise (true validator flow), use fusion prompt.
    doc_text = (pkt.get("document_text") or pkt.get("text") or "").strip()
    doc_text_len = len(doc_text)
    mode = "doc" if doc_text_len > 0 else "fusion"

    if mode == "doc":
        prompt = build_ea_doc_prompt(pkt)
    else:
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

    # Try to parse the JSON directly first (before coerce)
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else {}
    except Exception:
        parsed = {}
    
    # If empty output, retry once with stricter instruction
    if _is_empty_ea_obj(parsed):
        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous output was EMPTY. You MUST fill every field with non-empty content. "
              "If evidence is missing, explicitly write 'Insufficient evidence' items rather than leaving arrays empty."
        )
        runner2 = OllamaRunner(
            model=model,
            host=host,
            timeout_sec=timeout_sec,
            num_predict=num_predict,
            temperature=0.0,   # stricter
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
        raw2 = runner2.infer(prompt=retry_prompt, system=PROMPT_SYSTEM)
        try:
            parsed2 = json.loads(raw2) if isinstance(raw2, str) else {}
        except Exception:
            parsed2 = {}
    
        if not _is_empty_ea_obj(parsed2):
            raw = raw2
            parsed = parsed2
    
    # If still empty after retry, force a non-empty fallback
    if _is_empty_ea_obj(parsed):
        out = _fallback_from_doc(doc_text)
    else:
        ea_obj = coerce_ea_json(raw)
        out = ea_output_to_dict(ea_obj)


    # 5) Attach meta for traceability
    out["_meta"] = {
        "engine": "ollama",
        "model": model,
        "bytes_in": len(prompt),
        "bytes_out": len(raw) if isinstance(raw, str) else 0,
        "confidence": out.get("confidence", 0.8),
        "mode": mode,
        "doc_text_len": doc_text_len,
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
