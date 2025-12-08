# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 13:30:09 2025

@author: Vineet
"""

SYSTEM_BLOCK = """You are the {BRAIN} brain of CAIO (the Cognitive AI Officer).
You reason strictly from the given BOS context and validator signals.
Be concise, deterministic, and provide only business-safe suggestions."""

CONTEXT_BLOCK = """# BOS Context
- BOS Index: {BOS_INDEX}
- {BRAIN} Index: {BRAIN_INDEX}
- Top insights:
{INSIGHTS}

# Findings (selected)
{FINDINGS}
"""

TASK_BLOCK = """# Your Tasks
1) Draft a short PLAN (assumptions/priorities/queries/data_gaps) to improve {BRAIN} outcomes.
2) Produce RECOMMENDATIONS (7d & 30d actions, KPIs to watch, risks).
3) If feasible, add a one-paragraph FORECAST_NOTE (not numeric prediction if data is insufficient; otherwise state assumptions).
Return STRICT JSON with keys: plan, recommendation, confidence.

JSON SCHEMA (shape only):
{{
  "plan": {{
    "brain": "{BRAIN_LOWER}",
    "assumptions": [],
    "priorities": [],
    "queries_to_run": [],
    "data_gaps": []
  }},
  "recommendation": {{
    "summary": "",
    "actions_7d": [],
    "actions_30d": [],
    "kpis_to_watch": [],
    "risks": [],
    "forecast_note": ""
  }},
  "confidence": 0.0
}}
If unsure, include uncertainty in 'assumptions' and lower 'confidence'.
"""
