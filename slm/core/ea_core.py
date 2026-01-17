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
        "- Be concise, board-ready, and action-focused.\n"
        "- Use simple bullet phrases, not long paragraphs.\n"
        "- Top 5 priorities max; Top 6 risks max.\n"
        "- 7-day and 30-day actions should be cross-functional and non-duplicative.\n"
        "- owner_matrix keys must be CFO/CMO/COO/CHRO/CPO, each with 1-5 clear actions.\n"
        "- Confidence is 0.0..1.0.\n"
        "- You MUST ground priorities/risks/actions in the document evidence (DATA.text_excerpt / DATA.facts).\n"
        "- Do NOT output generic strategy language. If you cannot cite evidence, put it in key_risks as \"Insufficient evidence: ...\".\n"
        "- cross_brain_actions_7d and cross_brain_actions_30d must be concrete and measurable (include numbers/terms when available).\n"
        "- owner_matrix actions must reference the document (deliverables, costs, milestones) using parentheses, e.g. \"... (Evidence: 10% rev share)\".\n"
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
