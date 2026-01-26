# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Dict, Any, List, Tuple, Optional

from slm.core.slm_core import OllamaRunner, PROMPT_SYSTEM
from slm.core.ea_core import (
    build_ea_prompt,
    build_ea_doc_prompt,
    coerce_ea_json,
    ea_output_to_dict,
)

# =============================================================================
# EA schema expectations & validators
# =============================================================================

EA_SYSTEM = (
    PROMPT_SYSTEM
    + "\n\n"
    + "CRITICAL OUTPUT FORMAT:\n"
      "- Output MUST be a single valid JSON object and NOTHING else.\n"
      "- No markdown, no code fences, no explanation.\n"
      "- Do not add any keys outside the required schema.\n"
      "- If a field lacks evidence, write 'Insufficient evidence: <what>' instead of leaving it empty.\n"
)

REQUIRED_EA_KEYS = {
    "executive_summary",
    "top_priorities",
    "key_risks",
    "cross_brain_actions_7d",
    "cross_brain_actions_30d",
    "owner_matrix",
    "confidence",
}

REQUIRED_ROLES = ["CFO", "CMO", "COO", "CHRO", "CPO"]


def _ea_schema_template() -> Dict[str, Any]:
    """
    Minimal schema template used for repair prompts.
    (We keep this small to reduce burden on small models.)
    """
    return {
        "executive_summary": "string",
        "top_priorities": ["string", "string", "string"],
        "key_risks": ["string", "string"],
        "cross_brain_actions_7d": [
            "CFO: ...",
            "CMO: ...",
            "COO: ...",
            "CHRO: ...",
            "CPO: ...",
        ],
        "cross_brain_actions_30d": [
            "CFO: ...",
            "CMO: ...",
            "COO: ...",
            "CHRO: ...",
            "CPO: ...",
        ],
        "owner_matrix": {
            "CFO": ["action"],
            "CMO": ["action"],
            "COO": ["action"],
            "CHRO": ["action"],
            "CPO": ["action"],
        },
        "confidence": 0.7,
    }

def _extract_first_json_object(text: str) -> str:
    if not isinstance(text, str):
        return ""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1].strip()
    return ""


def _try_parse_json(s: Any) -> Dict[str, Any]:
    if not isinstance(s, str) or not s.strip():
        return {}
    candidate = _extract_first_json_object(s) or s
    try:
        j = json.loads(candidate)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}



def _is_empty_ea_obj(d: Dict[str, Any]) -> bool:
    """
    True if "basically empty": no summary and all major lists empty.
    """
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


def _is_valid_ea_schema(obj: Dict[str, Any]) -> bool:
    """
    Strict-enough schema validation to prevent "placeholder" and empty matrices.
    """
    if not isinstance(obj, dict):
        return False

    if not REQUIRED_EA_KEYS.issubset(set(obj.keys())):
        return False

    if not isinstance(obj.get("executive_summary"), str) or not obj["executive_summary"].strip():
        return False

    tp = obj.get("top_priorities")
    if not isinstance(tp, list) or len(tp) < 3:
        return False

    kr = obj.get("key_risks")
    if not isinstance(kr, list) or len(kr) < 2:
        return False

    a7 = obj.get("cross_brain_actions_7d")
    if not isinstance(a7, list) or len(a7) < 5:
        return False

    a30 = obj.get("cross_brain_actions_30d")
    if not isinstance(a30, list) or len(a30) < 5:
        return False

    om = obj.get("owner_matrix")
    if not isinstance(om, dict):
        return False

    for role in REQUIRED_ROLES:
        v = om.get(role)
        if not isinstance(v, list) or len(v) < 1:
            return False

    # confidence should be numeric-ish
    try:
        float(obj.get("confidence", 0.0))
    except Exception:
        return False

    return True


def _needs_repair(obj: Dict[str, Any]) -> bool:
    """
    Repair if empty OR schema-invalid.
    """
    return _is_empty_ea_obj(obj) or (not _is_valid_ea_schema(obj))


# =============================================================================
# Fallbacks (deterministic + generic)
# =============================================================================

def _fallback_nonempty_ea() -> Dict[str, Any]:
    """
    Safe generic fallback (used mainly in fusion mode if we have no doc text).
    """
    return {
        "executive_summary": (
            "The model returned an empty or invalid plan. This is a safe fallback. "
            "Re-run after strengthening evidence extraction or increasing model capacity."
        ),
        "top_priorities": [
            "Extract key facts (pricing, deliverables, timelines) from the input",
            "Define success KPIs and reporting cadence",
            "Assign owners and dependencies",
        ],
        "key_risks": [
            "Empty/invalid model output (Evidence: schema validation failure)",
            "Insufficient evidence in provided inputs (Evidence: missing or unclear details)",
        ],
        "cross_brain_actions_7d": [
            "CFO: Confirm commercial terms and budget ceiling (Evidence: provided inputs)",
            "CMO: Convert deliverables into a 30-day content calendar (Evidence: listed deliverables)",
            "COO: Define workflow + approvals + cadence (Evidence: execution requirement)",
            "CHRO: Assign roles/owners and capacity plan (Evidence: resourcing implied)",
            "CPO: Vendor/SLA checklist for external deliverables (Evidence: proposal context)",
        ],
        "cross_brain_actions_30d": [
            "CFO: Define ROI model and tracking (Evidence: expected outcomes)",
            "CMO: Launch content pipeline and measure engagement baseline (Evidence: scope)",
            "COO: Implement weekly execution review (Evidence: timeline requirement)",
            "CHRO: Define accountability + incentives (Evidence: governance)",
            "CPO: Finalize vendor milestones and acceptance criteria (Evidence: deliverables)",
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
    Ensure each brain payload is a dict with keys: plan, recommendation, confidence.
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
            out[k] = {
                "plan": getattr(v, "plan", {}) or {},
                "recommendation": getattr(v, "recommendation", {}) or {},
                "confidence": float(getattr(v, "confidence", 0.7)),
            }
    return out


# =============================================================================
# Charts helpers (kept from your file; minimal edits for safety)
# =============================================================================

def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _guess_brain_actual_total(brain: str, pkt: Dict[str, Any]) -> Optional[float]:
    metrics = pkt.get(f"{brain}_metrics") or {}

    if brain == "cfo":
        val = metrics.get("total_expenses") or metrics.get("total_costs") or metrics.get("opex_total")
        if val is not None:
            return _safe_float(val)

        finance = pkt.get("pnl_snapshot") or pkt.get("finance") or {}
        val = finance.get("total_expenses") or finance.get("operating_expenses") or finance.get("total_costs")
        return _safe_float(val)

    if brain == "cmo":
        val = metrics.get("marketing_spend_total") or metrics.get("spend_total") or metrics.get("ad_spend_total")
        return _safe_float(val)

    if brain == "chro":
        val = metrics.get("hr_total_spend") or metrics.get("people_costs_total") or metrics.get("spend_total")
        return _safe_float(val)

    if brain == "coo":
        val = metrics.get("operating_cost_total") or metrics.get("ops_spend_total") or metrics.get("spend_total")
        return _safe_float(val)

    if brain == "cpo":
        val = (
            metrics.get("external_talent_cost_total")
            or metrics.get("contractor_cost_total")
            or metrics.get("outsourcing_cost_total")
            or metrics.get("spend_total")
        )
        return _safe_float(val)

    val = metrics.get("spend_total") or metrics.get("total_cost")
    return _safe_float(val)


def _build_ea_charts(pkt: Dict[str, Any]) -> List[Dict[str, Any]]:
    charts: List[Dict[str, Any]] = []
    budgets = pkt.get("budgets") or {}

    grouped_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for brain, bdata in budgets.items():
        if not isinstance(bdata, dict):
            continue
        budget_total = _safe_float(bdata.get("total_annual") or bdata.get("total") or bdata.get("budget"))
        actual_total = _guess_brain_actual_total(brain, pkt)

        if budget_total is None and actual_total is None:
            continue

        label = str(brain).upper()

        if budget_total is not None:
            grouped_rows.append({"brain": label, "kind": "Budget", "value": budget_total})
        if actual_total is not None:
            grouped_rows.append({"brain": label, "kind": "Actual", "value": actual_total})

        if budget_total is not None and actual_total is not None:
            delta_rows.append({"brain": label, "delta": actual_total - budget_total})

    if grouped_rows:
        charts.append(
            {
                "id": "ea-budget-vs-actual-by-brain",
                "brain": "ea",
                "type": "bar",
                "title": "Budget vs Actual by Function",
                "x": {"field": "brain", "label": "Function"},
                "y": {"field": "value", "label": "Amount", "unit": "currency"},
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

    finance = pkt.get("pnl_snapshot") or pkt.get("finance") or {}
    rev = _safe_float(finance.get("revenue_total") or finance.get("total_revenue") or finance.get("revenue"))
    profit = _safe_float(finance.get("net_profit") or finance.get("profit_after_tax") or finance.get("ebitda"))

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


# =============================================================================
# Deterministic doc fallback (used if model fails schema even after repair)
# =============================================================================

def _extract_json_block(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Find the first JSON object-like block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0).strip() if m else ""


def _extract_facts_from_doc(text: str) -> Dict[str, Any]:
    t = text or ""

    # normalize common artifacts
    t = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060]", "", t)
    t = t.replace("\u00a0", " ")
    t = t.replace("ﬁ", "fi")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    sample_lines = lines[:12]

    money: List[str] = []
    money += re.findall(r"(₹\s*\d[\d,]*(?:\.\d+)?)", t)
    money += re.findall(r"((?:Rs\.?|INR)\s*\d[\d,]*(?:\.\d+)?)", t, flags=re.IGNORECASE)
    money += re.findall(r"(\d[\d,]{2,})\s*(?:per\s*month|/month|monthly|pm)", t, flags=re.IGNORECASE)
    money += re.findall(r"(\d[\d,]{2,})\s*(?:\n|\s)*per\s*(?:\n|\s)*month", t, flags=re.IGNORECASE)

    money = [m.strip() for m in money if str(m).strip()]
    money = list(dict.fromkeys(money))[:8]

    perc = re.findall(r"(\d{1,3}\s?%)(?!\w)", t)
    perc = [p.strip() for p in perc if p.strip()]
    perc = list(dict.fromkeys(perc))[:8]

    dates = re.findall(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", t)
    dates = list(dict.fromkeys(dates))[:8]

    deliverable_phrases = [
        "podcast", "vodcast", "masterclass", "reels", "shorts",
        "long-form", "long form", "youtube", "instagram", "linkedin",
        "case study", "webinar", "newsletter",
        "short video", "long video", "content calendar", "content strategy",
    ]
    deliverable_hits: List[str] = []
    for kw in deliverable_phrases:
        if re.search(rf"\b{re.escape(kw)}\b", t, flags=re.IGNORECASE):
            deliverable_hits.append(kw)
    deliverables = list(dict.fromkeys(deliverable_hits))[:12]

    return {
        "money": money[:5],
        "percent": perc[:5],
        "dates": dates[:5],
        "deliverables": deliverables,
        "sample_lines": sample_lines,
    }


def _fallback_from_doc(doc_text: str) -> Dict[str, Any]:
    facts = _extract_facts_from_doc(doc_text)
    money = ", ".join(facts["money"]) if facts["money"] else "pricing not found"
    perc = ", ".join(facts["percent"]) if facts["percent"] else "percent terms not found"
    dels = ", ".join(facts["deliverables"]) if facts["deliverables"] else "deliverables unclear"
    date = facts["dates"][0] if facts["dates"] else "date not found"

    preview = " | ".join(facts.get("sample_lines", [])[:4])
    evidence = f"Evidence: {money}; {perc}; deliverables: {dels}; date: {date}. Preview: {preview}"

    return {
        "executive_summary": (
            "Document-first plan generated via deterministic extraction because the model returned empty/invalid JSON. "
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


# =============================================================================
# Repair prompts
# =============================================================================

def _build_repair_prompt(base_prompt: str, broken_output: str) -> str:
    """
    Strong repair prompt: show schema + broken output; demand ONLY JSON.
    """
    schema = _ea_schema_template()
    return (
        base_prompt
        + "\n\n=== JSON REPAIR MODE ===\n"
          "Your previous output was EMPTY or INVALID JSON, or failed schema checks.\n"
          "Return ONLY valid JSON. No markdown. No commentary.\n"
          "Must include ALL keys in REQUIRED SCHEMA.\n"
          "owner_matrix MUST contain CFO/CMO/COO/CHRO/CPO each with 1–3 actions.\n"
          "If evidence is missing, write 'Insufficient evidence: <what>' instead of leaving fields empty.\n\n"
          "REQUIRED SCHEMA TEMPLATE:\n"
        + "```json\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n```\n\n"
          "BROKEN OUTPUT:\n"
        + "```text\n" + (broken_output or "")[:6000] + "\n```\n"
    )


# =============================================================================
# Main entrypoint
# =============================================================================

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

    Key behaviors:
    - doc mode if pkt has document_text/text
    - fusion mode otherwise
    - 2-pass generation: initial -> repair (if empty/invalid) -> deterministic fallback
    - Always attaches _meta and charts
    """
    per_brain_norm = _normalize_per_brain(per_brain)

    doc_text = (pkt.get("document_text") or pkt.get("text") or "").strip()
    doc_text_len = len(doc_text)
    mode = "doc" if doc_text_len > 0 else "fusion"

    prompt = build_ea_doc_prompt(pkt) if mode == "doc" else build_ea_prompt(pkt, per_brain_norm)

    def _parse_model_output(s: Any) -> Dict[str, Any]:
        """
        Robust parse: try extracted JSON block first, then raw.
        """
        if not isinstance(s, str) or not s.strip():
            return {}
        block = _extract_json_block(s)
        parsed_obj = _try_parse_json(block) if block else {}
        if parsed_obj:
            return parsed_obj
        return _try_parse_json(s)

    # -----------------------------
    # Pass 1: Primary generation
    # -----------------------------
    runner = OllamaRunner(
        model=model,
        host=host,
        timeout_sec=timeout_sec,
        num_predict=num_predict,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )

    raw1 = runner.infer(prompt=prompt, system=EA_SYSTEM)
    parsed1 = _parse_model_output(raw1)

    # Keep these variables so we can debug later
    raw2 = ""
    parsed2: Dict[str, Any] = {}

    # -----------------------------
    # Pass 2: Repair if needed
    # -----------------------------
    parsed = parsed1
    raw = raw1

    if _needs_repair(parsed1):
        repair_prompt = _build_repair_prompt(prompt, raw1 if isinstance(raw1, str) else "")

        runner2 = OllamaRunner(
            model=model,
            host=host,
            timeout_sec=timeout_sec,
            num_predict=num_predict,
            temperature=0.0,   # stricter
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
        raw2 = runner2.infer(prompt=repair_prompt, system=EA_SYSTEM)
        parsed2 = _parse_model_output(raw2)

        if not _needs_repair(parsed2):
            raw = raw2
            parsed = parsed2

    # -----------------------------
    # Final decision + DEBUG
    # -----------------------------
    if _needs_repair(parsed):
        # Debug prints ONLY when we end up falling back
        try:
            print("[EA_DEBUG] Fallback triggered (still empty/invalid after repair).")
            if isinstance(raw1, str):
                print("[EA_DEBUG] raw1_head:", raw1[:400].replace("\n", "\\n"))
                print("[EA_DEBUG] raw1_tail:", raw1[-400:].replace("\n", "\\n"))
            if isinstance(raw2, str) and raw2:
                print("[EA_DEBUG] raw2_head:", raw2[:400].replace("\n", "\\n"))
                print("[EA_DEBUG] raw2_tail:", raw2[-400:].replace("\n", "\\n"))
        except Exception:
            # Never break the pipeline due to debug
            pass

        if mode == "doc" and doc_text_len > 0:
            out = _fallback_from_doc(doc_text)
        else:
            out = _fallback_nonempty_ea()
    else:
        out = parsed

        # Ensure tools/charts exists for downstream consumers (charts + UI)
        if isinstance(out, dict):
            out.setdefault("tools", {"charts": []})
            if isinstance(out["tools"], dict):
                out["tools"].setdefault("charts", [])

    # -----------------------------
    # Attach meta (always)
    # -----------------------------
    if not isinstance(out, dict):
        # safety: never return non-dict
        out = _fallback_nonempty_ea()

    out["_meta"] = {
        "engine": "ollama",
        "model": model,
        "bytes_in": len(prompt) if isinstance(prompt, str) else 0,
        "bytes_out": len(raw) if isinstance(raw, str) else 0,
        "confidence": out.get("confidence", 0.8),
        "mode": mode,
        "doc_text_len": doc_text_len,
    }

    # -----------------------------
    # Attach EA-level charts
    # -----------------------------
    tools: Dict[str, Any] = out.setdefault("tools", {})
    if not isinstance(tools, dict):
        tools = {}
        out["tools"] = tools

    charts = tools.setdefault("charts", [])
    if not isinstance(charts, list):
        charts = []
        tools["charts"] = charts

    existing_ids = {c.get("id") for c in charts if isinstance(c, dict)}
    for chart in _build_ea_charts(pkt):
        cid = chart.get("id")
        if cid and cid not in existing_ids:
            charts.append(chart)

    return out
