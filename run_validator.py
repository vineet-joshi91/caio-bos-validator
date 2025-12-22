# -*- coding: utf-8 -*-
import os, sys, glob
import argparse, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# --- Make local modules importable no matter how this script is launched ---
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Try to import the cross engine, but don't mask real errors.
_CROSS_AVAILABLE = False
_CROSS_IMPORT_ERR = None
try:
    import cross_rules_engine  # local module in project root
    from cross_rules_engine import evaluate_cross_rules  # function
    _CROSS_AVAILABLE = True
except Exception as e:
    evaluate_cross_rules = None
    _CROSS_IMPORT_ERR = f"{type(e).__name__}: {e}"

# Try to import the reality engine
_REALITY_AVAILABLE = False
_REALITY_IMPORT_ERR = None
try:
    import reality_rules_engine  # local module in project root
    from reality_rules_engine import evaluate_reality
    _REALITY_AVAILABLE = True
except Exception as e:
    evaluate_reality = None
    _REALITY_IMPORT_ERR = f"{type(e).__name__}: {e}"

from evaluator import load_brain_rules, run_brain_validation
from evaluator.bos_index import combine_brains
from evaluator.ingest import load_any

BRAINS = ["cfo", "coo", "cmo", "chro", "cpo"]

def run_single_brain(brain: str, df: pd.DataFrame, rules_dir: str) -> dict:
    rules = load_brain_rules(rules_dir, brain)
    return run_brain_validation(brain, rules, df)

def _resolve_cross_dir(rules_root: str):
    """
    Prefer rules/cross, fallback to rules/_cross.
    Returns (cross_dir_path, count_of_yaml_files).
    """
    cand = [os.path.join(rules_root, "cross"),
            os.path.join(rules_root, "_cross")]
    for d in cand:
        if os.path.isdir(d):
            paths = sorted(glob.glob(os.path.join(d, "*.yaml")))
            return d, len(paths)
    return cand[0], 0

def _resolve_reality_dir(rules_root: str):
    """
    Prefer rules/reality, fallback to rules/_reality.
    Returns (reality_dir_path, count_of_yaml_files).
    """
    cand = [os.path.join(rules_root, "reality"),
            os.path.join(rules_root, "_reality")]
    for d in cand:
        if os.path.isdir(d):
            paths = sorted(glob.glob(os.path.join(d, "**", "*.yaml"), recursive=True))
            paths += sorted(glob.glob(os.path.join(d, "**", "*.yml"), recursive=True))
            return d, len(paths)
    return cand[0], 0

def main():
    ap = argparse.ArgumentParser(description="CAIO BOS Validator")
    ap.add_argument("--brain", choices=BRAINS, help="Run a single brain")
    ap.add_argument("--all-brains", action="store_true", help="Run all brains on one input")
    ap.add_argument("--input", required=True, help="Input file (.xlsx/.csv/.json/.docx/.pdf/.txt)")
    ap.add_argument("--rules", default="rules", help="Rules root directory (default: rules/)")
    ap.add_argument("--output", default=None, help="Optional JSON output path")
    ap.add_argument("--no-cross", action="store_true", help="Skip cross-brain evaluation")
    ap.add_argument("--no-reality", action="store_true", help="Skip reality signals evaluation")
    args = ap.parse_args()

    # 1) Load inputs (works for every supported format the ingestor handles)
    brain_inputs = load_any(args.input)  # dict: { "cfo": df, ... }

    # 2) Per-brain evaluation
    if args.all_brains:
        results = {}
        with ThreadPoolExecutor(max_workers=max(1, len(BRAINS))) as ex:
            futs = {
                ex.submit(run_single_brain, b, brain_inputs.get(b), args.rules): b
                for b in BRAINS if b in brain_inputs
            }
            for fut in as_completed(futs):
                b = futs[fut]
                results[b] = fut.result()
        bos = combine_brains(results)
        payload = {"brains": results, **bos}
    else:
        if not args.brain:
            ap.error("Provide --brain or --all-brains.")
        df = brain_inputs[args.brain]
        payload = run_single_brain(args.brain, df, args.rules)

    # 3) Cross-brain rules (if available)
    cross_dir, cross_count = _resolve_cross_dir(args.rules)
    cross_out = None

    if not args.no_cross:
        if _CROSS_AVAILABLE and callable(evaluate_cross_rules):
            try:
                cross_findings = evaluate_cross_rules(brain_inputs, cross_dir)
                cross_out = {
                    "meta": {
                        "engine": "native",
                        "rules_path": os.path.normpath(cross_dir),
                        "rules_count": cross_count,
                        "status": "ok",
                        "error": None,
                        "engine_file": getattr(cross_rules_engine, "__file__", None),
                    },
                    "findings": cross_findings or [],
                }
            except Exception as e:
                cross_out = {
                    "meta": {
                        "engine": "native",
                        "rules_path": os.path.normpath(cross_dir),
                        "rules_count": cross_count,
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "engine_file": getattr(cross_rules_engine, "__file__", None),
                    },
                    "findings": [],
                }
        else:
            cross_out = {
                "meta": {
                    "engine": "missing",
                    "rules_path": os.path.normpath(cross_dir),
                    "rules_count": cross_count,
                    "status": "engine_import_failed",
                    "error": _CROSS_IMPORT_ERR,
                    "engine_file": None,
                },
                "findings": [],
            }

    if cross_out is not None:
        payload["cross"] = cross_out

    # 3b) Reality rules (cached/curated external grounding)
    reality_dir, reality_count = _resolve_reality_dir(args.rules)
    reality_out = None

    if not args.no_reality:
        if _REALITY_AVAILABLE and callable(evaluate_reality):
            try:
                reality_out = evaluate_reality(reality_dir, payload)
                # patch in discovered counts
                if isinstance(reality_out, dict):
                    reality_out.setdefault("meta", {})
                    reality_out["meta"]["signals_count_discovered"] = reality_count
                    reality_out["meta"]["engine_file"] = getattr(reality_rules_engine, "__file__", None)
            except Exception as e:
                reality_out = {
                    "meta": {
                        "engine": "yaml_reality_v0",
                        "rules_path": os.path.normpath(reality_dir),
                        "signals_count_discovered": reality_count,
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "engine_file": getattr(reality_rules_engine, "__file__", None),
                    },
                    "signals": [],
                    "feasibility": {"by_domain": {}, "by_brain": {}},
                }
        else:
            reality_out = {
                "meta": {
                    "engine": "missing",
                    "rules_path": os.path.normpath(reality_dir),
                    "signals_count_discovered": reality_count,
                    "status": "engine_import_failed",
                    "error": _REALITY_IMPORT_ERR,
                    "engine_file": None,
                },
                "signals": [],
                "feasibility": {"by_domain": {}, "by_brain": {}},
            }

    # Always attach a reality block (even if skipped) so frontend can rely on it
    if reality_out is None:
        reality_out = {
            "meta": {
                "engine": "skipped",
                "rules_path": os.path.normpath(reality_dir),
                "signals_count_discovered": reality_count,
                "status": "skipped",
                "error": None,
                "engine_file": None,
            },
            "signals": [],
            "feasibility": {"by_domain": {}, "by_brain": {}},
        }

    payload["reality"] = reality_out

    # 4) Output
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()
