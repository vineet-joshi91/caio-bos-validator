# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:52:22 2025

@author: Vineet
"""

import argparse, json, os
from evaluator.engine import evaluate
from evaluator.scorer import compute_scores_and_insights

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to normalized payload JSON")
    ap.add_argument("--rules", default="rules", help="Rules directory (default: rules)")
    ap.add_argument("--templates", default="insights/insight_templates.yaml", help="Path to insight templates")
    ap.add_argument("--weights", default="config/weights.yaml", help="Path to weights config")
    ap.add_argument("--out", default=None, help="Optional: path to save combined JSON")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    engine_res = evaluate(payload, args.rules)
    scored = compute_scores_and_insights(engine_res.get("findings", []), args.templates, args.weights)

    combined = {
        "label": engine_res.get("label"),
        "engine_rationale": engine_res.get("rationale"),
        "findings": engine_res.get("findings", []),
        **scored
    }

    text = json.dumps(combined, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            fo.write(text)
