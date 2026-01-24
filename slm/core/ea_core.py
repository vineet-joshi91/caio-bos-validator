# -*- coding: utf-8 -*-
"""
EA Core helpers:
- build_ea_prompt(pkt, per_brain): make a compact DATA + strict SCHEMA prompt
- coerce_ea_json(raw_text): parse/repair EA JSON into a consistent dict
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# -----------------------------
# Output container (typed)
# -----------------------------
@dataclass
class EAOutput:
    executive_summary: str
    top_priorities: List[str]
    key_risks: List[str]
    cross_brain_actions_7d: List[str]
    cross_brain_actions_30d: List[str]
    owner_matrix: Dict[str, List[str]]  # owner -> list of actions
    confidence: float


# -----------------------------
# Internal helpers
# -----------------------------
def _take(xs: Optional[list], n: int) -> list:
    if not isinstance(xs, list):
        return []
    return xs[:n]


def _string(x: Any) -> str:
    return x if isinstance(x, str) else ""


def _float(x: Any, default: float = 0.7) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _brain_brief(brain_name: str, brain_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Trim each brain's output to only what's useful for EA."""
    plan = brain_obj.get("plan", {}) or {}
    rec = brain_obj.get("recommendation", {}) or {}

    return {
        "brain": brain_name.upper(),
        "plan_priorities": _take(plan.get("priorities", []), 5),
        "plan_gaps": _take(plan.get("data_gaps", []), 5),
        "rec_summary": _string(rec.get("summary", ""))[:240],
        "rec_actions_7d": _take(rec.get("actions_7d", []), 6),
        "rec_actions_30d": _take(rec.get("actions_30d", []), 6),
        "kpis": _take(rec.get("kpis_to_watch", []), 8),
        "risks": _take(rec.get("risks", []), 6),
        "confidence": _float(brain_obj.get("confidence", 0.7)),
    }


def _schema_hint() -> Dict[str, Any]:
    # The exact shape we want the EA model to return
    return {
        "executive_summary": "",
        "top_priorities": [],
        "key_risks": [],
        "cross_brain_actions_7d": [],
        "cross_brain_actions_30d": [],
        "owner_matrix": {
            # example:
            # "CFO": ["Tighten cash reconciliation for Q3", "Rebuild BS mapping for FY25"],
            # "CMO": ["Normalize UTMs", "Reduce CAC by pausing underperformers"],
        },
        "confidence": 0.8
    }


# -----------------------------
# Public: build prompt for EA
# -----------------------------
def build_ea_prompt(pkt: Dict[str, Any], per_brain: Dict[str, Any]) -> str:
    """
    pkt: validator packet (bos_index, brain_indices, insights, findings, ...)
    per_brain: { "cfo": {...}, "cmo": {...}, ... }  5 brain SLM outputs
    """
    bos_index = _float(pkt.get("bos_index", 0.0), 0.0)
    brain_indices = pkt.get("brain_indices", {}) or {}
    insights_map = pkt.get("insights", {}) or {}
    findings = pkt.get("findings", []) or []

    # Compact cross-brain snapshot
    brains = ["cfo", "cmo", "coo", "chro", "cpo"]
    brief = {
        "bos_index": bos_index,
        "brain_indices": {k: _float(v, 0.0) for k, v in brain_indices.items()},
    
        # ✅ Evidence scaffolding
        "source": pkt.get("source") or {},
        "facts": pkt.get("facts") or {},
        "text_excerpt": (pkt.get("document_text") or pkt.get("text") or "")[:8000],
    
        "insights": {b: _take(insights_map.get(b, []), 6) for b in brains},
        "top_findings_sample": [
            {
                "rule_id": f.get("rule_id"),
                "severity": f.get("severity"),
                "title": f.get("title"),
            }
            for f in findings[:12]
        ],
        "per_brain": {b: _brain_brief(b, per_brain.get(b, {}) or {}) for b in brains},
    }


    schema = _schema_hint()

    prompt = (
        "Fuse these per-brain CXO results and validator context into ONE executive JSON.\n\n"
        "DATA:\n" + json.dumps(brief, ensure_ascii=False) + "\n\n"
        "SCHEMA (return EXACTLY this shape, no extra keys):\n"
        + json.dumps(schema, ensure_ascii=False) + "\n\n"
        "RULES:\n"
        "- You MUST use the document evidence in DATA.text_excerpt and/or DATA.facts.\n"
        "- You are NOT allowed to return empty fields.\n"
        "  * executive_summary must be 2–4 sentences.\n"
        "  * top_priorities must have 3–5 items.\n"
        "  * key_risks must have 2–6 items.\n"
        "  * cross_brain_actions_7d must have 5 items (CFO/CMO/COO/CHRO/CPO each at least 1).\n"
        "  * cross_brain_actions_30d must have 5 items (CFO/CMO/COO/CHRO/CPO each at least 1).\n"
        "  * owner_matrix must contain CFO/CMO/COO/CHRO/CPO with 1–3 actions each.\n"
        "- Every item must cite evidence in parentheses, e.g. \"Do X (Evidence: ₹130,000/mo)\".\n"
        "- If the document lacks evidence, DO NOT generalize. Instead:\n"
        "  * put a priority like \"Clarify missing scope/details (Evidence: missing in document)\"\n"
        "  * and put risks like \"Insufficient evidence to estimate ROI (Evidence: no baseline metrics)\".\n"
        "- Use simple bullet phrases, not long paragraphs.\n"
        "- Confidence is 0.0..1.0.\n"
        "Return ONLY the JSON."
    )
    return prompt

def build_ea_doc_prompt(pkt: Dict[str, Any]) -> str:
    """
    Document-first EA prompt.
    Uses pkt['document_text'] (or pkt['text']) and produces the EA schema output.
    """
    source = pkt.get("source") or {}
    facts = pkt.get("facts") or {}

    doc_text = (pkt.get("document_text") or pkt.get("text") or "").strip()
    if not doc_text:
        raise ValueError("No document_text extracted; cannot build EA prompt.")
    
    if len(doc_text) > 9000:
        text_excerpt = doc_text[:6000] + "\n\n--- [TRUNCATED] ---\n\n" + doc_text[-3000:]
    else:
        text_excerpt = doc_text

    
    schema = _schema_hint()

    prompt = (
        "You are an executive planning engine. Produce a concrete, evidence-based Executive Action Plan.\n"
        "You MUST use the provided document excerpt as your primary evidence.\n\n"
        "SOURCE:\n```json\n" + json.dumps(source, ensure_ascii=False, indent=2) + "\n```\n\n"
        "FACTS (may be empty):\n```json\n" + json.dumps(facts, ensure_ascii=False, indent=2) + "\n```\n\n"
        "DOCUMENT_EXCERPT (verbatim):\n```text\n" + (text_excerpt or "").strip() + "\n```\n\n"
        "SCHEMA (return EXACTLY this shape, no extra keys):\n"
        + json.dumps(schema, ensure_ascii=False, indent=2) + "\n\n"
        "RULES:\n"
        "- You are NOT allowed to return empty fields.\n"
        "  * executive_summary must be 2–4 sentences.\n"
        "  * top_priorities must have 3–5 items.\n"
        "  * key_risks must have 2–6 items.\n"
        "  * cross_brain_actions_7d must have 5 items (CFO/CMO/COO/CHRO/CPO each at least 1).\n"
        "  * cross_brain_actions_30d must have 5 items (CFO/CMO/COO/CHRO/CPO each at least 1).\n"
        "  * owner_matrix must contain CFO/CMO/COO/CHRO/CPO with 1–3 actions each.\n"
        "- Every item MUST cite evidence in parentheses from DOCUMENT_EXCERPT or FACTS.\n"
        "  Example: \"Confirm pricing model (Evidence: ₹130,000/month + 10% revenue share)\".\n"
        "- If evidence is missing, DO NOT generalize; write \"Insufficient evidence: <what>\" as a risk/action.\n"
        "Return ONLY the JSON."
    )
    return prompt


# -----------------------------
# Public: coerce model output
# -----------------------------
def coerce_ea_json(raw_text: str) -> EAOutput:
    """
    Parse the EA model's raw JSON string, fill defaults, and ensure type-stability.
    """
    try:
        obj = json.loads(raw_text)
        if not _string(obj.get("executive_summary", "")).strip() and not any(
            _take(obj.get(k), 1) for k in ["top_priorities", "key_risks", "cross_brain_actions_7d", "cross_brain_actions_30d"]
        ):
            obj["executive_summary"] = "Model returned empty output. Re-run with stronger evidence or a shorter excerpt."
            obj["key_risks"] = ["Empty model output (Evidence: model returned blank fields)"]

    except Exception:
        obj = {}

    # Defaults
    exec_summary = _string(obj.get("executive_summary", "")).strip()
    top_priorities = _take(obj.get("top_priorities"), 5)
    key_risks = _take(obj.get("key_risks"), 6)
    a7 = _take(obj.get("cross_brain_actions_7d"), 10)
    a30 = _take(obj.get("cross_brain_actions_30d"), 12)
    owner_matrix = obj.get("owner_matrix", {}) or {}
    if not isinstance(owner_matrix, dict):
        owner_matrix = {}

    # Normalize owner keys and lists
    owners = ["CFO", "CMO", "COO", "CHRO", "CPO"]
    norm_owner_matrix: Dict[str, List[str]] = {}
    for k in owners:
        v = owner_matrix.get(k) or owner_matrix.get(k.upper()) or owner_matrix.get(k.lower())
        norm_owner_matrix[k] = _take(v if isinstance(v, list) else [], 6)

    conf = _float(obj.get("confidence", 0.8), 0.8)
    
    # --- Fallback: if model left owner_matrix empty, derive from cross_brain_actions_7d ---
    if all(len(v) == 0 for v in norm_owner_matrix.values()):
        actions = a7 if isinstance(a7, list) else []
        owners_rr = ["CFO", "CMO", "COO", "CHRO", "CPO"]
        rr_i = 0
    
        for act in actions:
            if not isinstance(act, str):
                continue
    
            # If action is prefixed like "CFO: ..." map directly
            prefix = act.split(":", 1)[0].strip().upper() if ":" in act else ""
            if prefix in owners_rr and ":" in act:
                norm_owner_matrix[prefix].append(act.split(":", 1)[1].strip())
            else:
                norm_owner_matrix[owners_rr[rr_i % len(owners_rr)]].append(act.strip())
                rr_i += 1
    
        # cap to 3 actions per owner
        for k in owners_rr:
            norm_owner_matrix[k] = _take(norm_owner_matrix[k], 3)

    
    return EAOutput(
        executive_summary=exec_summary,
        top_priorities=top_priorities,
        key_risks=key_risks,
        cross_brain_actions_7d=a7,
        cross_brain_actions_30d=a30,
        owner_matrix=norm_owner_matrix,
        confidence=conf,
    )


# -----------------------------
# Optional: to dict (for JSON dump)
# -----------------------------
def ea_output_to_dict(ea: EAOutput) -> Dict[str, Any]:
    return {
        "executive_summary": ea.executive_summary,
        "top_priorities": ea.top_priorities,
        "key_risks": ea.key_risks,
        "cross_brain_actions_7d": ea.cross_brain_actions_7d,
        "cross_brain_actions_30d": ea.cross_brain_actions_30d,
        "owner_matrix": ea.owner_matrix,
        "confidence": ea.confidence,
    }
