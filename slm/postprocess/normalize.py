# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 12:16:21 2025

@author: Vineet
"""

# slm/postprocess/normalize.py
from typing import Any, Dict, List

def _list(x): return x if isinstance(x, list) else ([] if x in (None, "") else [x])

def to_ui_payload(ea: Dict[str, Any]) -> Dict[str, Any]:
    # fallbacks
    exec_sum = ea.get("executive_summary") or ""
    top_priorities = ea.get("top_priorities") or []
    key_risks = _list(ea.get("key_risks") or [])
    cross7 = _list(ea.get("cross_brain_actions_7d") or [])
    cross30 = _list(ea.get("cross_brain_actions_30d") or [])
    owner_matrix = ea.get("owner_matrix") or {}
    meta = ea.get("_meta") or {}

    # coerce priorities into a simple, safe shape
    priorities: List[Dict[str, Any]] = []
    for p in top_priorities:
        priorities.append({
            "brain": (p.get("brain") or "").upper(),
            "actions_7d": _list(p.get("actions_7d")),
            "actions_30d": _list(p.get("actions_30d")),
        })

    owners = {k.upper(): _list(v) for k, v in owner_matrix.items()} if isinstance(owner_matrix, dict) else {}

    return {
        "summary": exec_sum[:2000],
        "priorities": priorities,                      # [{brain, actions_7d[], actions_30d[]}]
        "risks": key_risks,                            # []
        "actions": {"cross_7d": cross7, "cross_30d": cross30},
        "owners": owners,                              # {CFO:[], CMO:[], ...}
        "meta": {
            "engine": meta.get("engine"), "model": meta.get("model"),
            "bytes_out": meta.get("bytes_out"), "confidence": ea.get("confidence"),
        },
    }
