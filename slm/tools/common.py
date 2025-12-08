# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:16:19 2025

@author: Vineet
"""

from typing import Dict, Any, List, Tuple, Optional


def safe_div(n: float, d: float) -> Optional[float]:
    try:
        if d == 0 or d is None or n is None:
            return None
        return n / d
    except Exception:
        return None


def add_need(needs: List[str], text: str) -> None:
    if text not in needs:
        needs.append(text)


def clip(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    return max(lo, min(hi, v))


def as_months(days: Optional[float]) -> Optional[float]:
    if days is None:
        return None
    return days / 30.0


def ensure_recommendation_shape(obj: Dict[str, Any]) -> None:
    """
    Ensure that obj['recommendation'] has all the standard keys
    used across brains, including longer-term action buckets.

    This lets CFO/CMO/COO/CHRO/CPO/EA all rely on the same structure
    without duplicating the key list in every *_slm.py file.
    """
    rec = obj.setdefault("recommendation", {})

    # Always present
    rec.setdefault("summary", "")

    # Action buckets – short, medium, long term
    rec.setdefault("actions_7d", [])
    rec.setdefault("actions_30d", [])
    rec.setdefault("actions_quarter", [])
    rec.setdefault("actions_half_year", [])
    rec.setdefault("actions_year", [])

    # Monitoring & risk sections
    rec.setdefault("kpis_to_watch", [])
    rec.setdefault("risks", [])

    # Narrative forecast
    rec.setdefault("forecast_note", "")
