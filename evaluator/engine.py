# evaluator/engine.py
from __future__ import annotations
import os, glob, json, yaml
from typing import Dict, List, Any, Tuple, Set
import pandas as pd
import numpy as np

# New unified imports (internal only, no APIs)
from evaluator.formula_registry import FORMULA_MAP, run_check
from evaluator.scoring import aggregate_scores, severity_weight
from evaluator.intent_resolver import resolve_intents

TOL = 1e-6

def _run_single_rule(rule: Dict[str, Any], df_resolved: pd.DataFrame) -> Dict[str, Any]:
    statuses = {"pass": 2, "warn": 1, "fail": 0}
    worst = 2
    min_score = 1.0
    details_out = []
    for chk in rule.get("evidence", {}).get("checks", []):
        result = run_check(chk, df_resolved)
        worst = min(worst, statuses.get(result["status"], 1))
        min_score = min(min_score, float(result["score"]))
        details_out.append(result.get("details", {}))
    rev_status = {v: k for k, v in statuses.items()}[worst]
    return {
        "id": rule.get("id"),
        "title": rule.get("title"),
        "severity": rule.get("severity", "warn"),
        "status": rev_status,
        "score": float(min_score),
        "_file": rule.get("_filepath"),
        "details": details_out,
    }

def run_brain_validation(brain: str, rules: List[Dict[str, Any]], df_input: pd.DataFrame) -> Dict[str, Any]:
    """Single-brain validator used by the all-brains runner."""
    df_resolved = resolve_intents(df_input, brain=brain)
    per_rule = [_run_single_rule(r, df_resolved) for r in rules]
    agg = aggregate_scores(per_rule)
    agg["brain"] = brain
    return agg

# ---------------------------
# YAML loading
# ---------------------------
def _load_yaml_files(rules_dir: str) -> List[dict]:
    files = sorted(glob.glob(os.path.join(rules_dir, "**/*.yaml"), recursive=True))
    rules = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            rule = yaml.safe_load(fh)
            rule["_filepath"] = f
            rules.append(rule)
    return rules

# ---------------------------
# Legacy table-style helpers
# ---------------------------
def _required_tables_present(payload: Dict[str, Any], tables: List[str]) -> Tuple[bool, List[str]]:
    missing = [t for t in tables if t not in payload or not isinstance(payload[t], list)]
    return (len(missing)==0, missing)

def _columns_present(table_rows: List[dict], columns: List[str]) -> Tuple[bool, List[str]]:
    if not table_rows:
        return False, columns
    present = set().union(*[set(r.keys()) for r in table_rows])
    missing = [c for c in columns if c not in present]
    return (len(missing)==0, missing)

def _equation_ok(table_rows: List[dict], lhs: str, rhs_terms: List[str], group_by: str|None=None) -> Tuple[bool, List[str]]:
    failures = []
    for row in table_rows:
        try:
            lhs_val = float(row.get(lhs, 0.0))
            rhs_val = sum(float(row.get(t, 0.0)) for t in rhs_terms)
            if abs(lhs_val - rhs_val) > TOL:
                key = f"{group_by}={row.get(group_by)}" if group_by else ""
                failures.append(key or "row_mismatch")
        except Exception:
            failures.append("parse_error")
    return (len(failures)==0, failures)

def _range_ok(table_rows: List[dict], columns: List[str], mn=None, mx=None) -> Tuple[bool, List[str]]:
    fails = []
    for i, row in enumerate(table_rows):
        for c in columns:
            v = row.get(c, None)
            try:
                if v is None: 
                    continue
                v = float(v)
                if (mn is not None and v < mn) or (mx is not None and v > mx):
                    fails.append(f"row{i}:{c}={v}")
            except Exception:
                fails.append(f"row{i}:{c}=?")
    return (len(fails)==0, fails)

def _ratio_bounds(table_rows: List[dict], num: str, den: str, mn=None, mx=None, require_den_pos=False) -> Tuple[bool, List[str]]:
    fails = []
    for i, row in enumerate(table_rows):
        try:
            n = float(row.get(num, 0.0))
            d = float(row.get(den, 0.0))
            if require_den_pos and d <= 0:
                fails.append(f"row{i}:den={d}")
                continue
            if d == 0:
                fails.append(f"row{i}:den=0")
                continue
            r = n / d
            if (mn is not None and r < mn) or (mx is not None and r > mx):
                fails.append(f"row{i}:ratio={r}")
        except Exception:
            fails.append(f"row{i}:calc_err")
    return (len(fails)==0, fails)

def _periods(table_rows: List[dict], col: str="period") -> Set[str]:
    return set([str(r.get(col)) for r in table_rows if r.get(col) is not None])

def _period_align(payload: Dict[str, Any], tables: List[str]) -> Tuple[bool, List[str]]:
    sets = []
    for t in tables:
        rows = payload.get(t, [])
        sets.append((t, _periods(rows)))
    base_name, base = sets[0]
    fails = []
    for name, s in sets[1:]:
        if s != base:
            fails.append(f"{name} != {base_name}")
    return (len(fails)==0, fails)

def _monotonic_time(table_rows: List[dict], col: str="period") -> Tuple[bool, List[str]]:
    periods = [str(r.get(col)) for r in table_rows if r.get(col) is not None]
    if periods != sorted(periods):
        return False, ["non_monotonic_or_gaps"]
    return True, []

# ---------------------------
# Adapters to new engine
# ---------------------------
def _to_df(table_rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(table_rows or [])

def _dispatch_new_check(df: pd.DataFrame, brain: str, chk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a modern Formula-Map check against a DataFrame.
    We also run the intent resolver so `_like` columns can be auto-filled.
    """
    # Some checks may reference columns that aren't exact; resolve best-effort:
    df_resolved = resolve_intents(df, brain=brain)
    res = run_check(chk, df_resolved)  # {"status","score","details"}
    return res

# ---------------------------
# Evaluate (hybrid support)
# ---------------------------
LEGACY_TYPES = {
    "required_columns",
    "equation",
    "range_check",
    "period_align",
    "ratio_bounds",
    "monotonic_time",
}

def evaluate(payload: Dict[str, Any], rules_dir: str, brain: str|None=None) -> Dict[str, Any]:
    """
    payload: dict of tables -> list-of-rows (legacy)
    rules_dir: directory containing YAMLs
    brain: optional hint for intent resolution on new checks ("cfo","cmo","coo","chro","cpo")
    """
    rules = _load_yaml_files(rules_dir)
    findings: List[Dict[str, Any]] = []
    per_rule_scored: List[Dict[str, Any]] = []

    any_block = False
    any_warn = False

    for rule in rules:
        rid = rule.get("id")
        sev = rule.get("severity","warn")
        title = rule.get("title","")
        ev = rule.get("evidence",{})

        # Legacy: requires_tables
        req_tables = ev.get("requires_tables", [])
        if req_tables:
            ok_tables, missing = _required_tables_present(payload, req_tables)
            if not ok_tables:
                findings.append({
                    "rule_id": rid, "severity": sev, "where": ",".join(missing),
                    "message": f"Missing required tables: {missing}", "title": title
                })
                if sev=="block": any_block = True
                else: any_warn = True
                # still record a scored item
                per_rule_scored.append({"id": rid, "title": title, "severity": sev, "status": "fail", "score": 0.0, "_file": rule.get("_filepath")})
                continue

        # Run checks (legacy OR new)
        checks = ev.get("checks", [])
        statuses_rank = {"pass": 2, "warn": 1, "fail": 0}
        worst = 2
        min_score = 1.0
        detail_blobs = []

        for chk in checks:
            ctype = chk.get("type")

            # --- legacy block ---
            if ctype in LEGACY_TYPES:
                status_ok = True
                details = []
                if ctype == "required_columns":
                    table = chk["table"]
                    cols = chk["columns"]
                    status_ok, details = _columns_present(payload.get(table, []), cols)

                elif ctype == "equation":
                    table = chk["table"]
                    expr = chk["expression"]  # "assets = liabilities + equity"
                    lhs, rhs = expr.split("=")
                    lhs = lhs.strip()
                    rhs_terms = [p.strip() for p in rhs.split("+")]
                    status_ok, details = _equation_ok(payload.get(table, []), lhs, rhs_terms, chk.get("group_by"))

                elif ctype == "range_check":
                    table = chk["table"]
                    cols = chk["columns"]
                    status_ok, details = _range_ok(payload.get(table, []), cols, chk.get("min"), chk.get("max"))

                elif ctype == "period_align":
                    tables = chk.get("tables", [])
                    status_ok, details = _period_align(payload, tables)

                elif ctype == "ratio_bounds":
                    table = chk["table"]
                    status_ok, details = _ratio_bounds(payload.get(table, []),
                                                      chk.get("numerator"),
                                                      chk.get("denominator"),
                                                      chk.get("min"),
                                                      chk.get("max"),
                                                      chk.get("require_denominator_positive", False))
                elif ctype == "monotonic_time":
                    table = chk["table"]
                    status_ok, details = _monotonic_time(payload.get(table, []), chk.get("column","period"))
                else:
                    status_ok, details = False, [f"unknown_check:{ctype}"]

                status = "pass" if status_ok else ("warn" if sev!="block" else "fail")
                score = 1.0 if status_ok else (0.6 if status=="warn" else 0.0)
                worst = min(worst, statuses_rank[status])
                min_score = min(min_score, score)
                detail_blobs.append({"type": ctype, "details": details})

            # --- new engine path (dispatch to FORMULA_MAP) ---
            elif ctype in FORMULA_MAP:
                # If the YAML references a specific table, use it, else try to merge all rows
                if "table" in chk and isinstance(chk["table"], str):
                    df = _to_df(payload.get(chk["table"], []))
                else:
                    # fallback: concatenate all tables with a table name column
                    frames = []
                    for tname, rows in payload.items():
                        if isinstance(rows, list):
                            df_t = _to_df(rows)
                            df_t["_table"] = tname
                            frames.append(df_t)
                    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

                res = _dispatch_new_check(df, brain or "", chk)
                status = res["status"]
                score  = float(res["score"])
                worst = min(worst, statuses_rank.get(status, 1))
                min_score = min(min_score, score)
                detail_blobs.append({"type": ctype, "details": res.get("details", {})})

            else:
                # Unknown type
                status = "fail" if sev=="block" else "warn"
                worst = min(worst, statuses_rank[status])
                min_score = min(min_score, 0.0 if status=="fail" else 0.6)
                detail_blobs.append({"type": ctype, "details": {"error": "unknown_check_type"}})

        # finalize this rule
        rev_status = {v: k for k,v in statuses_rank.items()}[worst]
        if rev_status != "pass":
            any_warn = any_warn or (sev!="block")
            any_block = any_block or (sev=="block")
            findings.append({
                "rule_id": rid, "severity": sev, "where": "", "message": rule.get("description",""), "title": title
            })

        per_rule_scored.append({
            "id": rid, "title": title, "severity": sev,
            "status": rev_status, "score": float(min_score),
            "_file": rule.get("_filepath"),
            "details": detail_blobs
        })

    # Old-style label + rationale
    if any_block:
        label = "Blocked (critical issues)"
        rationale = "One or more blocking rules failed."
    elif any_warn:
        label = "Needs attention"
        rationale = "Non-blocking issues detected."
    else:
        label = "Authentic enough"
        rationale = "All active rules passed."

    # New-style aggregate score
    agg = aggregate_scores(per_rule_scored)

    return {
        "label": label,
        "rationale": rationale,
        "aggregate_score": agg["aggregate_score"],
        "findings": findings,
        "breakdown": agg["breakdown"],
    }
