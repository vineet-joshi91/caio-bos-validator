# -*- coding: utf-8 -*-
from typing import Dict, Any

DEFAULT_BRAIN_WEIGHTS = {"cfo":0.25,"coo":0.25,"cmo":0.20,"chro":0.15,"cpo":0.15}

def combine_brains(brains_out: Dict[str, Any], brain_weights: Dict[str, float] = None) -> Dict[str, Any]:
    bw = brain_weights or DEFAULT_BRAIN_WEIGHTS
    num, den = 0.0, 0.0
    top_risks = []

    for brain, res in brains_out.items():
        w = bw.get(brain, 0.0)
        s = float(res.get("aggregate_score", 0.0))
        num += w * s
        den += w
        # collect lowest-scoring rules
        for r in res.get("breakdown", []):
            top_risks.append({
                "brain": brain,
                "rule_id": r.get("id"),
                "title": r.get("title"),
                "severity": r.get("severity"),
                "status": r.get("status"),
                "score": float(r.get("score", 0.0)),
            })

    bos_index = (num / den) if den else 0.0
    top_risks = sorted(top_risks, key=lambda x: x["score"])[:5]

    if   bos_index >= 0.85: label = "Healthy"
    elif bos_index >= 0.70: label = "Caution"
    else:                   label = "Critical"

    return {"bos_index": bos_index, "bos_label": label, "top_risks": top_risks}
