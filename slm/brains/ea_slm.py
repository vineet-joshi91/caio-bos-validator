# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Dict, Any, List, Optional

from slm.core.slm_core import OllamaRunner, PROMPT_SYSTEM
from slm.core.ea_core import build_ea_prompt, build_ea_doc_prompt


# =============================================================================
# System prompt (shared)
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

REQUIRED_ROLES = ["CFO", "CMO", "COO", "CHRO", "CPO"]

REQUIRED_EA_KEYS = {
    "executive_summary",
    "top_priorities",
    "key_risks",
    "cross_brain_actions_7d",
    "cross_brain_actions_30d",
    "owner_matrix",
    "confidence",
}

REQUIRED_DR_KEYS = {
    "plan_summary",
    "critical_gaps",
    "missing_metrics",
    "risk_flags",
    "recommendation",
    "owner_matrix",
    "confidence",
}


# =============================================================================
# JSON extraction / parsing
# =============================================================================

def _extract_first_json_object(text: str) -> str:
    """Extract the first brace-balanced JSON object from a string."""
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
    """Parse JSON safely; tolerate extra text around JSON."""
    if not isinstance(s, str) or not s.strip():
        return {}
    candidate = _extract_first_json_object(s) or s
    try:
        j = json.loads(candidate)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}


# =============================================================================
# EA validators
# =============================================================================

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


def _is_valid_ea_schema(obj: Dict[str, Any]) -> bool:
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

    try:
        float(obj.get("confidence", 0.0))
    except Exception:
        return False

    return True


def _needs_repair_ea(obj: Dict[str, Any]) -> bool:
    return _is_empty_ea_obj(obj) or (not _is_valid_ea_schema(obj))


# =============================================================================
# Decision Review normalization + relaxed validator
# =============================================================================

def _normalize_decision_review_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize DR output to stable schema:
    - Ensure keys exist
    - Coerce list fields to list[str]
    - Coerce recommendation object
    - Normalize owner_matrix role lists
    - Add placeholders so UI never blanks
    """
    if not isinstance(d, dict):
        return {}

    out = dict(d)

    # Ensure keys exist
    out.setdefault("plan_summary", "")
    out.setdefault("critical_gaps", [])
    out.setdefault("missing_metrics", [])
    out.setdefault("risk_flags", [])
    out.setdefault("recommendation", {})
    out.setdefault("owner_matrix", {})
    out.setdefault("confidence", 0.6)
    out.setdefault("tools", {"charts": []})

    # Coerce list fields
    for k in ["critical_gaps", "missing_metrics", "risk_flags"]:
        v = out.get(k)
        if not isinstance(v, list):
            out[k] = []
        else:
            out[k] = [str(x).strip() for x in v if str(x).strip()]

    # Recommendation object
    rec = out.get("recommendation")
    if not isinstance(rec, dict):
        rec = {}

    rec.setdefault("verdict", "CAUTION")
    rec.setdefault("why", [])
    rec.setdefault("next_steps", [])

    if rec.get("verdict") not in ("GO", "CAUTION", "NO-GO"):
        rec["verdict"] = "CAUTION"

    for k in ["why", "next_steps"]:
        rv = rec.get(k)
        if not isinstance(rv, list):
            rec[k] = []
        else:
            rec[k] = [str(x).strip() for x in rv if str(x).strip()]

    out["recommendation"] = rec

    # Owner matrix
    om = out.get("owner_matrix")
    if not isinstance(om, dict):
        om = {}
    norm_om: Dict[str, List[str]] = {}
    for role in REQUIRED_ROLES:
        rv = om.get(role)
        if isinstance(rv, list):
            norm_om[role] = [str(x).strip() for x in rv if str(x).strip()]
        elif isinstance(rv, str) and rv.strip():
            norm_om[role] = [rv.strip()]
        else:
            norm_om[role] = []
    out["owner_matrix"] = norm_om

    # Confidence numeric
    try:
        out["confidence"] = float(out.get("confidence", 0.6))
    except Exception:
        out["confidence"] = 0.6

    # Minimal placeholders
    if not out["plan_summary"]:
        out["plan_summary"] = "Insufficient evidence: plan summary not provided"
    if not out["critical_gaps"]:
        out["critical_gaps"] = ["Insufficient evidence: critical gaps not provided"]
    if not out["missing_metrics"]:
        out["missing_metrics"] = ["Insufficient evidence: missing metrics not provided"]
    if not out["risk_flags"]:
        out["risk_flags"] = ["Insufficient evidence: risk flags not provided"]

    if not out["recommendation"]["why"]:
        out["recommendation"]["why"] = ["Insufficient evidence: recommendation rationale not provided"]
    if not out["recommendation"]["next_steps"]:
        out["recommendation"]["next_steps"] = ["Insufficient evidence: next steps not provided"]

    return out


def _is_valid_decision_review_schema(obj: Dict[str, Any]) -> bool:
    """
    Relaxed DR validation (prevents rejecting good reviews):
    - Must have plan_summary (non-empty)
    - Must have >=1 critical gap
    - Must have recommendation.verdict in GO/CAUTION/NO-GO
    - Must have owner_matrix with at least 3 roles populated (>=1 item)
    """
    if not isinstance(obj, dict):
        return False

    if not isinstance(obj.get("plan_summary"), str) or not obj["plan_summary"].strip():
        return False

    cg = obj.get("critical_gaps")
    if not isinstance(cg, list) or len(cg) < 1:
        return False

    rec = obj.get("recommendation")
    if not isinstance(rec, dict):
        return False
    if rec.get("verdict") not in ("GO", "CAUTION", "NO-GO"):
        return False

    om = obj.get("owner_matrix")
    if not isinstance(om, dict):
        return False

    roles_ok = 0
    for role in REQUIRED_ROLES:
        rv = om.get(role)
        if isinstance(rv, list) and len(rv) > 0:
            roles_ok += 1
    if roles_ok < 3:
        return False

    return True


def _needs_repair_dr(obj: Dict[str, Any]) -> bool:
    return (not isinstance(obj, dict)) or (not _is_valid_decision_review_schema(obj))


# =============================================================================
# EA normalization (convert action dicts to strings)
# =============================================================================

def _normalize_model_ea_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}

    out = dict(d)

    def _actions_to_strings(v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        res: List[str] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    res.append(s)
                continue
            if isinstance(item, dict):
                # {"CFO": "..."} style
                if len(item) == 1:
                    k, val = next(iter(item.items()))
                    if isinstance(k, str) and isinstance(val, str):
                        s = f"{k}: {val}".strip()
                        if s:
                            res.append(s)
                        continue

                # {"action":"CFO","description":"..."} style
                act = item.get("action")
                desc = item.get("description") or item.get("detail") or item.get("text")
                owner = item.get("owner")
                if isinstance(act, str) and isinstance(desc, str):
                    s = f"{act}: {desc}".strip()
                    if isinstance(owner, str) and owner.strip():
                        s += f" (Owner: {owner.strip()})"
                    res.append(s)
        return res

    out["cross_brain_actions_7d"] = _actions_to_strings(out.get("cross_brain_actions_7d"))
    out["cross_brain_actions_30d"] = _actions_to_strings(out.get("cross_brain_actions_30d"))

    # owner_matrix normalization
    om = out.get("owner_matrix")
    if not isinstance(om, dict):
        om = {}
    norm_om: Dict[str, List[str]] = {}
    for role in REQUIRED_ROLES:
        v = om.get(role)
        if isinstance(v, list):
            norm_om[role] = [str(x).strip() for x in v if str(x).strip()]
        elif isinstance(v, str) and v.strip():
            norm_om[role] = [v.strip()]
        else:
            norm_om[role] = []
    out["owner_matrix"] = norm_om

    # tools/charts
    tools = out.get("tools")
    if not isinstance(tools, dict):
        tools = {}
    if not isinstance(tools.get("charts"), list):
        tools["charts"] = []
    out["tools"] = tools

    # confidence numeric
    try:
        out["confidence"] = float(out.get("confidence", 0.7))
    except Exception:
        out["confidence"] = 0.7

    return out


# =============================================================================
# Fallbacks
# =============================================================================

def _fallback_nonempty_ea() -> Dict[str, Any]:
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


def _fallback_from_doc(doc_text: str) -> Dict[str, Any]:
    # Minimal deterministic extraction (good enough as safety net)
    t = doc_text or ""
    t = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060]", "", t)
    t = t.replace("\u00a0", " ").replace("ﬁ", "fi")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    preview = " | ".join(lines[:4])

    money = re.findall(r"(₹\s*\d[\d,]*(?:\.\d+)?)", t)
    perc = re.findall(r"(\d{1,3}\s?%)(?!\w)", t)

    deliverables = []
    for kw in ["masterclass", "reels", "shorts", "long form", "long-form", "podcast", "vodcast"]:
        if re.search(rf"\b{re.escape(kw)}\b", t, flags=re.IGNORECASE):
            deliverables.append(kw)

    money_s = ", ".join(dict.fromkeys([m.strip() for m in money if m.strip()])[:3]) or "pricing not found"
    perc_s = ", ".join(dict.fromkeys([p.strip() for p in perc if p.strip()])[:3]) or "percent terms not found"
    dels_s = ", ".join(dict.fromkeys(deliverables)[:6]) or "deliverables unclear"

    evidence = f"Evidence: {money_s}; {perc_s}; deliverables: {dels_s}. Preview: {preview}"

    return {
        "executive_summary": (
            "Document-first plan generated via deterministic extraction because the model returned empty/invalid JSON. "
            f"({evidence})"
        ),
        "top_priorities": [
            f"Confirm commercial terms and scope ({evidence})",
            f"Convert deliverables into a 30-day production calendar (Evidence: {dels_s})",
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
            f"CMO: Draft 30-day content calendar from deliverables (Evidence: {dels_s})",
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


def _fallback_decision_review() -> Dict[str, Any]:
    # Must match NEW DR schema
    return _normalize_decision_review_dict({
        "plan_summary": "Decision Review failed to generate a valid schema.",
        "critical_gaps": ["Insufficient evidence: model output invalid or incomplete"],
        "missing_metrics": ["Insufficient evidence: missing metrics not provided"],
        "risk_flags": ["Insufficient evidence: risk flags not provided"],
        "recommendation": {
            "verdict": "CAUTION",
            "why": ["Insufficient evidence: recommendation rationale not provided"],
            "next_steps": ["Re-run Decision Review with higher num_predict or lower temperature"],
        },
        "owner_matrix": {r: ["Insufficient evidence"] for r in REQUIRED_ROLES},
        "confidence": 0.6,
        "tools": {"charts": []},
    })


# =============================================================================
# Decision Review prompt
# =============================================================================

def build_decision_review_prompt(pkt: Dict[str, Any]) -> str:
    plan_text = (pkt.get("document_text") or pkt.get("text") or "").strip()

    schema = {
        "plan_summary": "2-4 sentences",
        "critical_gaps": ["5-12 items"],
        "missing_metrics": ["3-10 items"],
        "risk_flags": ["3-10 items"],
        "recommendation": {
            "verdict": "GO | CAUTION | NO-GO",
            "why": ["3-6 bullets"],
            "next_steps": ["5 bullets (actionable)"],
        },
        "owner_matrix": {r: ["1-3 actions"] for r in REQUIRED_ROLES},
        "confidence": "number between 0.5 and 0.9",
    }

    return (
        "You are an Executive Decision Reviewer.\n"
        "You will NOT rewrite the plan.\n"
        "Your job is to evaluate feasibility and execution readiness.\n\n"
        "INPUT_PLAN (verbatim):\n```text\n"
        + plan_text[:14000]
        + "\n```\n\n"
        "Return ONLY valid JSON with this exact schema (no extra keys):\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\n"
        "RULES:\n"
        "- plan_summary must summarize the INPUT_PLAN in 2-4 sentences.\n"
        "- critical_gaps must list gaps/unknowns that block confident execution.\n"
        "- missing_metrics must list metrics required to judge success/ROI.\n"
        "- risk_flags must list execution risks and failure modes.\n"
        "- recommendation.verdict must be one of: GO, CAUTION, NO-GO.\n"
        "- recommendation.why must explain the verdict based on gaps/metrics/risks.\n"
        "- recommendation.next_steps must be concrete actions the user should take next.\n"
        "- If evidence is missing, write 'Insufficient evidence: <what>'.\n"
        "- confidence must be a number between 0.5 and 0.9.\n"
    )


# =============================================================================
# Repair prompt (generic)
# =============================================================================

def _build_repair_prompt(base_prompt: str, broken_output: str) -> str:
    return (
        base_prompt
        + "\n\nIMPORTANT:\n"
          "- Your previous output was INVALID JSON or failed schema checks.\n"
          "- Return ONLY valid JSON. No markdown. No code fences.\n"
          "- Do not include any commentary outside JSON.\n\n"
          "BROKEN OUTPUT:\n```text\n"
        + (broken_output or "")[:4000]
        + "\n```\n"
    )


# =============================================================================
# Charts (EA only)
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
# Per-brain normalization
# =============================================================================

def _normalize_per_brain(per_brain: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
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

    doc_text = (pkt.get("document_text") or pkt.get("text") or "").strip()
    doc_text_len = len(doc_text)

    meta = pkt.get("meta") or {}
    review_mode = meta.get("mode") in ("decision_review_from_plan", "decision_review")
    mode = "decision_review" if review_mode else ("doc" if doc_text_len > 0 else "fusion")

    # Prompt selection
    if review_mode:
        prompt = build_decision_review_prompt(pkt)
    else:
        per_brain_norm = _normalize_per_brain(per_brain or {})
        prompt = build_ea_doc_prompt(pkt) if doc_text_len > 0 else build_ea_prompt(pkt, per_brain_norm)

    def _parse_model_output(s: Any) -> Dict[str, Any]:
        return _try_parse_json(s) if isinstance(s, str) else {}

    def _needs_repair_mode(obj: Dict[str, Any]) -> bool:
        return _needs_repair_dr(obj) if review_mode else _needs_repair_ea(obj)

    # Pass 1
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

    if review_mode:
        parsed1 = _normalize_decision_review_dict(parsed1)
    else:
        parsed1 = _normalize_model_ea_dict(parsed1)

    raw2 = ""
    parsed2: Dict[str, Any] = {}

    parsed = parsed1
    raw = raw1

    # Pass 2 (repair)
    if _needs_repair_mode(parsed1):
        repair_prompt = _build_repair_prompt(prompt, raw1 if isinstance(raw1, str) else "")

        runner2 = OllamaRunner(
            model=model,
            host=host,
            timeout_sec=timeout_sec,
            num_predict=num_predict,
            temperature=0.0,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
        raw2 = runner2.infer(prompt=repair_prompt, system=EA_SYSTEM)
        parsed2 = _parse_model_output(raw2)

        if review_mode:
            parsed2 = _normalize_decision_review_dict(parsed2)
        else:
            parsed2 = _normalize_model_ea_dict(parsed2)

        if not _needs_repair_mode(parsed2):
            raw = raw2
            parsed = parsed2

    # Final decision
    if _needs_repair_mode(parsed):
        try:
            print("[EA_DEBUG] Fallback triggered (still invalid after repair). review_mode=", review_mode)
            if isinstance(raw1, str):
                print("[EA_DEBUG] raw1_head:", raw1[:400].replace("\n", "\\n"))
                print("[EA_DEBUG] raw1_tail:", raw1[-400:].replace("\n", "\\n"))
            if isinstance(raw2, str) and raw2:
                print("[EA_DEBUG] raw2_head:", raw2[:400].replace("\n", "\\n"))
                print("[EA_DEBUG] raw2_tail:", raw2[-400:].replace("\n", "\\n"))
        except Exception:
            pass

        out = _fallback_decision_review() if review_mode else (_fallback_from_doc(doc_text) if doc_text_len > 0 else _fallback_nonempty_ea())
    else:
        out = parsed
        if isinstance(out, dict):
            out.setdefault("tools", {"charts": []})
            if isinstance(out["tools"], dict):
                out["tools"].setdefault("charts", [])

    # Always ensure dict
    if not isinstance(out, dict):
        out = _fallback_decision_review() if review_mode else _fallback_nonempty_ea()

    out["_meta"] = {
        "engine": "ollama",
        "model": model,
        "bytes_in": len(prompt) if isinstance(prompt, str) else 0,
        "bytes_out": len(raw) if isinstance(raw, str) else 0,
        "confidence": out.get("confidence", 0.6 if review_mode else 0.8),
        "mode": mode,
        "doc_text_len": doc_text_len,
    }

    # Charts only for EA
    if not review_mode:
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
