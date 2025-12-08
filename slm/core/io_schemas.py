# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 13:29:47 2025

@author: Vineet
"""

from typing import TypedDict, List, Dict, Any, Optional

class ValidatorFinding(TypedDict, total=False):
    rule_id: str
    severity: str
    where: str
    message: str
    title: str
    bucket: str

class ValidatorPacket(TypedDict, total=False):
    label: str
    engine_rationale: str
    findings: List[Dict[str, Any]]
    bos_index: float
    brain_indices: Dict[str, float]
    bucket_scores: Dict[str, Dict[str, float]]
    insights: Dict[str, List[str]]
    meta: Dict[str, Any]        # OPTIONAL: place to stash high-level numbers
    tables: Dict[str, Any]      # OPTIONAL: normalized tables if/when available

class BrainPlan(TypedDict, total=False):
    brain: str
    assumptions: List[str]
    priorities: List[str]
    queries_to_run: List[str]
    data_gaps: List[str]

class BrainRecommendation(TypedDict, total=False):
    summary: str
    actions_7d: List[str]
    actions_30d: List[str]
    kpis_to_watch: List[str]
    risks: List[str]
    forecast_note: Optional[str]

class BrainSLMOutput(TypedDict, total=False):
    plan: Dict[str, Any]
    recommendation: Dict[str, Any]
    confidence: float
    tools: Dict[str, Any]       # <--- NEW: the snapshot we add
