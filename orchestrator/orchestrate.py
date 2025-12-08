# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:42:20 2025

@author: Vineet
"""

# orchestrator/orchestrate.py
from concurrent.futures import ThreadPoolExecutor
from evaluator.engine import run_brain_validation
from orchestrator.cross_store import Facts
from orchestrator.cross_rules_engine import run_cross_rules
import numpy as np

def orchestrate_all(input_data, rules_root):
    brains = ["cfo", "cmo", "coo", "chro", "cpo"]
    brain_reports = {}

    with ThreadPoolExecutor(max_workers=len(brains)) as pool:
        futures = {b: pool.submit(run_brain_validation, b, f"{rules_root}/{b}", input_data) for b in brains}
        for b, f in futures.items():
            brain_reports[b] = f.result()

    # ingest results into unified store
    facts = Facts()
    for b, rep in brain_reports.items():
        facts.ingest_brain(b, rep)

    cross_report = run_cross_rules(facts, f"{rules_root}/_cross")
    brain_reports["cross"] = cross_report

    bos_index = np.mean([r["aggregate_score"] for r in brain_reports.values()])
    bos_label = _label(bos_index)

    return {"brains": brain_reports, "bos_index": bos_index, "bos_label": bos_label}


def _label(idx):
    if idx >= 0.75: return "Healthy"
    if idx >= 0.55: return "Watch"
    if idx >= 0.40: return "Critical"
    return "Severe"
