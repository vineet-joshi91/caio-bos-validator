# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:23:39 2025

@author: Vineet
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import yaml
import os
from collections import defaultdict

# ----- Helpers -----
def _brain_from_rule_id(rule_id: str) -> str:
    # Ex: CFO-R-001 -> cfo
    return rule_id.split("-")[0].lower()

def _bucket_score_init() -> Dict[str, float]:
    # buckets are optional; we’ll create on-the-fly per brain
    return defaultdict(float)

def _severity_penalty(sev: str, weights: Dict[str, Any]) -> float:
    # default penalties if not overridden in config
    if sev == "block":
        return float(weights.get("penalty_block", 10))
    return float(weights.get("penalty_warn", 5))

def _label_from_boards(label_components: Dict[str, Any]) -> str:
    # High-level label for whole submission
    if label_components["any_block"]:
        return "Blocked (critical issues)"
    if label_components["any_warn"]:
        return "Needs attention"
    return "Authentic enough"

# ----- Main scorer -----
def compute_scores_and_insights(
    findings: List[Dict[str, Any]],
    templates_path: str,
    weights_path: str
) -> Dict[str, Any]:
    """
    Input:
      findings: engine output list of dicts: {rule_id, severity, where, message, title}
      templates_path: insights/insight_templates.yaml
      weights_path:   config/weights.yaml
    Output:
      dict with:
        - bos_index (0-100)
        - brain_indices {cfo, cmo, coo, chro, cpo}
        - bucket_scores {brain: {bucket: score}}
        - insights {brain: [strings]}
        - label (overall high-level)
    """
    # Load config
    with open(weights_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    with open(templates_path, "r", encoding="utf-8") as f:
        tpl = yaml.safe_load(f) or {}

    penalty_weights = cfg.get("penalties", {})
    bucket_weights = cfg.get("bucket_weights", {})  # optional: per-brain bucket weighting

    # Organize failures by brain & bucket
    brains = ["cfo", "cmo", "coo", "chro", "cpo"]
    failed_counts = {b: 0 for b in brains}
    penalties = {b: 0.0 for b in brains}
    bucket_fails = {b: defaultdict(int) for b in brains}
    any_block = False
    any_warn = False

    for fnd in findings:
        rid = fnd.get("rule_id", "UNKNOWN-0")
        brain = _brain_from_rule_id(rid)
        if brain not in brains:
            # treat as common; distribute softly across all brains
            brain = "common"
        sev = fnd.get("severity", "warn")
        bucket = fnd.get("bucket") or "unspecified"

        if brain == "common":
            # apply to all brains (soft penalty factor)
            factor = float(cfg.get("common_penalty_factor", 0.5))
            for b in brains:
                failed_counts[b] += 1
                p = _severity_penalty(sev, penalty_weights) * factor
                penalties[b] += p
                bucket_fails[b][bucket] += 1
        else:
            failed_counts[brain] += 1
            p = _severity_penalty(sev, penalty_weights)
            penalties[brain] += p
            bucket_fails[brain][bucket] += 1

        if sev == "block":
            any_block = True
        elif sev == "warn":
            any_warn = True

    # Convert penalties to indices
    # start=100, subtract, then clamp >=0
    def idx_from_penalty(p: float) -> float:
        return max(0.0, 100.0 - p)

    brain_indices = {b: idx_from_penalty(penalties[b]) for b in brains}
    # BOS Index is the mean of available brain indices (brains with no data = still 100)
    bos_index = sum(brain_indices.values()) / len(brains)

    # Compute simple bucket scores per brain: 100 - (fail_count * bucket_penalty)
    bucket_penalty = float(cfg.get("bucket_penalty", 3.0))
    bucket_scores = {b: {} for b in brains}
    for b in brains:
        for buck, cnt in bucket_fails[b].items():
            bucket_scores[b][buck] = max(0.0, 100.0 - (cnt * bucket_penalty))

    # Create insight text using templates
    insights = {b: [] for b in brains}
    for b in brains:
        # map overall index to level
        lvl = "high" if brain_indices[b] >= 85 else "medium" if brain_indices[b] >= 60 else "low"
        btpl = tpl.get(b.upper(), {})
        # Topline insight
        topline = btpl.get("topline", {}).get(lvl)
        if topline:
            insights[b].append(topline)

        # Per-bucket remarks if available
        for buck, score in bucket_scores[b].items():
            blvl = "high" if score >= 85 else "medium" if score >= 60 else "low"
            buck_tpl = btpl.get(buck, {}).get(blvl)
            if buck_tpl:
                insights[b].append(buck_tpl)

    label = _label_from_boards({"any_block": any_block, "any_warn": any_warn})

    return {
        "label": label,
        "bos_index": round(bos_index, 2),
        "brain_indices": {k: round(v, 2) for k, v in brain_indices.items()},
        "bucket_scores": bucket_scores,
        "insights": insights
    }
