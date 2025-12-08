# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 11:00:48 2025

@author: Vineet
"""

# -*- coding: utf-8 -*-
from typing import List, Dict, Any

SEVERITY_WEIGHTS = {
    "block": 1.0,
    "warn": 0.6,
    "info": 0.3,
}

def severity_weight(severity: str) -> float:
    return SEVERITY_WEIGHTS.get(severity.lower(), 0.5)

def aggregate_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Each item: {"id": ..., "severity": ..., "status": ..., "score": float}
    Returns weighted aggregate and breakdown.
    """
    num = 0.0
    den = 0.0
    breakdown = []
    for r in results:
        w = severity_weight(r.get("severity", "warn"))
        num += r["score"] * w
        den += w
        breakdown.append({
            "id": r.get("id"),
            "title": r.get("title"),
            "severity": r.get("severity"),
            "status": r.get("status"),
            "score": float(r["score"]),
            "_file": r.get("_file"),
        })
    agg = (num / den) if den else 0.0
    return {"aggregate_score": float(agg), "breakdown": breakdown}
