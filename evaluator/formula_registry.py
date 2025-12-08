# -*- coding: utf-8 -*-
"""
CAIO BOS – Formula Dictionary & Registry
----------------------------------------
Single entry-point for all YAML rule `type:` implementations used across CFO/CHRO/COO/CMO/CPO.

Dependencies: pandas, numpy
Return schema for every check:
  { "status": "pass|warn|fail", "score": float, "details": {...} }

Severity mapping (block/warn/info) is handled by the orchestrator using the rule's YAML severity.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

Result = Dict[str, Any]  # {"status": str, "score": float, "details": dict}

# ---------- helpers ----------

def _res(status: str, score: float, **details) -> Result:
    return {"status": status, "score": float(score), "details": details or {}}

def _bounded_score(ok: bool, *, warn: bool=False,
                   pass_score: float=1.0, warn_score: float=0.6, fail_score: float=0.0) -> Result:
    if ok:
        return _res("pass", pass_score)
    if warn:
        return _res("warn", warn_score)
    return _res("fail", fail_score)

def _safe_div(a, b, eps: float=1e-9):
    return a / np.where(np.abs(b) < eps, eps, b)

def _require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    return len(missing) == 0, missing

def _soft_missing(cols):
    return _res("warn", 0.6, note="missing_columns", missing=cols)

# robust grouping helper (no KeyError when group_by is missing/None)
def _group(df, by=None):
    """
    Yields (key, frame). If `by` is None or not a column in df, returns one group ('all', df).
    """
    if by is None or by not in df.columns:
        yield "all", df
        return
    # keep NaNs as their own group for audits
    for k, g in df.groupby(by, dropna=False):
        yield k, g

# ---------- A) Ratio / Equation / Value ----------

def ratio_bounds_intents(df, numerator, denominator, low=None, high=None, group_by=None) -> dict:
    """
    Check that (numerator / denominator) stays within [low, high] (if provided).
    Works with intent columns and supports optional grouping.
    Returns a soft 'warn' when columns are missing instead of raising.
    """
    ok, miss = _require_columns(df, [numerator, denominator])
    if not ok:
        return _soft_missing(miss)

    def _eval(g):
        num = pd.to_numeric(g[numerator], errors="coerce")
        den = np.maximum(pd.to_numeric(g[denominator], errors="coerce"), 1e-9)
        r = num / den
        cond = True
        if low  is not None: cond &= (r >= float(low)).all()
        if high is not None: cond &= (r <= float(high)).all()
        details = {}
        if len(r):
            details = {"min": float(np.nanmin(r)), "max": float(np.nanmax(r))}
        out = _bounded_score(cond)
        out["details"] = details
        return out

    # evaluate groups and aggregate details
    agg_status = "pass"
    by_details = {}
    order = {"fail": 0, "warn": 1, "pass": 2}
    for k, g in _group(df, group_by):
        res = _eval(g)
        by_details[str(k)] = res["details"]
        if order[res["status"]] < order[agg_status]:
            agg_status = res["status"]

    # overall score simple mapping (pass=1, warn=0.6, fail=0)
    score = {"pass": 1.0, "warn": 0.6, "fail": 0.0}[agg_status]
    return {"status": agg_status, "score": score, "details": {"by_group": by_details}}

def ratio_bounds_intents_grouped(df, numerator, denominator, group_by, defaults) -> Result:
    lows, highs = defaults["low"], defaults["high"]
    det, ok_all = {}, True
    for ch, gch in df.groupby(group_by):
        r = _safe_div(gch[numerator].astype(float), gch[denominator].astype(float))
        ok = (r >= lows) & (r <= highs)
        ok_all &= ok.all()
        det[str(ch)] = {"min": float(np.nanmin(r)), "max": float(np.nanmax(r))}
    out = _bounded_score(ok_all); out["details"] = {"by": group_by, "details": det}; return out

def equation_intents(
    df,
    expression: str | None = None,
    left: str | None = None,
    right: str | None = None,
    lhs: str | None = None,
    rhs: str | None = None,
    left_sum: list[str] | None = None,
    right_sum: list[str] | None = None,
    group_by: str | None = None,
    # tolerance variants:
    tolerance: float | None = None,
    tolerance_abs: float | None = None,
    tol_abs: float | None = None,
    tolerance_mode: str | None = None,
    **_ignore,   # swallow any extra YAML keys safely
) -> Result:
    """
    YAML-friendly wrapper.

    Accepted shapes:
      - expression: "lhs = rhs"
      - left/right (or lhs/rhs) strings
      - left_sum/right_sum lists -> builds "a+b+... = x+y+..."

    Tolerance:
      - relative mode (default): uses 'equation_intents_tolerance' -> |lhs-rhs| <= tol * |lhs|
      - absolute mode: uses 'equation_intents_absolute'          -> |lhs-rhs| <= tol
      - value can be provided via 'tolerance' OR 'tolerance_abs' OR 'tol_abs'
    """
    # normalize synonyms for sides
    if left is None and lhs is not None:
        left = lhs
    if right is None and rhs is not None:
        right = rhs

    # build expression if needed
    if expression is None:
        if left_sum:
            left_expr = " + ".join(map(str, left_sum))
        else:
            left_expr = left
        if right_sum:
            right_expr = " + ".join(map(str, right_sum))
        else:
            right_expr = right
        if not left_expr or not right_expr:
            return _res("warn", 0.6, note="equation_missing_params",
                        hint="Provide `expression` or left/right (or left_sum/right_sum).")
        expression = f"{left_expr} = {right_expr}"

    # pick tolerance value
    tol = tolerance
    if tol is None:
        tol = tolerance_abs
    if tol is None:
        tol = tol_abs
    if tol is None:
        tol = 1e-6

    mode = (tolerance_mode or "relative").strip().lower()
    if mode in ("absolute", "abs"):
        return equation_intents_absolute(df, expression=expression, abs_tol=tol, group_by=group_by)
    else:
        # relative by default
        return equation_intents_tolerance(df, expression=expression, tolerance_abs=tol, group_by=group_by)

def equation_intents_tolerance(df, expression, tolerance_abs, group_by=None, **_ignore) -> Result:
    lhs, rhs = [s.strip() for s in expression.split("=")]

    def _required_cols(expr: str, cols) -> list[str]:
        # find tokens that look like column names and end with _like
        tokens = set(re.findall(r"[A-Za-z_]\w*", expr))
        return [t for t in tokens if t.endswith("_like") and t not in cols]

    det, ok_all = {}, True
    for k, g in _group(df, group_by):
        missing = _required_cols(lhs + " + " + rhs, g.columns)
        if missing:
            # soft warn for this group; keep the run alive
            ok_all = False
            det[str(k)] = {"missing_columns": missing}
            continue

        try:
            l = g.eval(lhs).astype(float)
            r = g.eval(rhs).astype(float)
        except Exception as e:
            ok_all = False
            det[str(k)] = {"eval_error": str(e)}
            continue

        err = (l - r).abs()
        tol = float(tolerance_abs) * (l.abs() + 1e-9)  # relative tolerance
        ok = (err <= tol).fillna(True).all()
        ok_all &= bool(ok)
        det[str(k)] = {
            "max_err": float(np.nanmax(err)) if len(err) else 0.0,
            "max_rel_err": float(np.nanmax((err / (l.abs() + 1e-9)).replace([np.inf,-np.inf], np.nan))) if len(err) else 0.0,
        }

    out = _bounded_score(ok_all)
    out["details"] = {"by_group": det}
    return out


def equation_tolerance_optional(df, expression, tolerance_abs, group_by=None) -> Result:
    try:
        return equation_intents_tolerance(df, expression, tolerance_abs, group_by)
    except Exception as e:
        return _res("warn", 0.6, note="optional_equation_failed", error=str(e))

def equation_intents_absolute(df, expression, abs_tol=1e-6, group_by=None) -> Result:
    """|lhs - rhs| <= abs_tol (absolute tolerance)."""
    # parse
    if "=" not in expression:
        return _res("warn", 0.6, note="equation_parse_error", expression=expression)
    lhs, rhs = [s.strip() for s in expression.split("=", 1)]

    def _eval(g):
        try:
            l = g.eval(lhs).astype(float)
            r = g.eval(rhs).astype(float)
        except Exception as e:
            return _res("warn", 0.6, note="equation_eval_error", expression=expression, error=str(e))
        err = (l - r).abs()
        ok = (err <= float(abs_tol)).fillna(True).all()
        det = {}
        if len(err):
            det = {"max_abs_err": float(np.nanmax(err))}
        out = _bounded_score(ok); out["details"] = det; return out

    # tolerant grouping
    status_order = {"fail":0,"warn":1,"pass":2}
    worst = 2; by = {}
    for k, g in _group(df, group_by):
        r = _eval(g); by[str(k)] = r["details"]; worst = min(worst, status_order[r["status"]])
    status = {v:k for k,v in status_order.items()}[worst]
    score = {"pass":1.0,"warn":0.6,"fail":0.0}[status]
    return {"status": status, "score": score, "details": {"by_group": by}}

def value_bounds(df, column, low, high, group_by=None) -> Result:
    det, ok_all = {}, True
    for k, g in _group(df, group_by):
        v = g[column].astype(float)
        ok = (v >= low) & (v <= high)
        ok_all &= ok.all()
        det[str(k)] = {"min": float(v.min()), "max": float(v.max())}
    out = _bounded_score(ok_all); out["details"] = {"details": det}; return out

def value_in_range(df, value, low_ref, high_ref) -> Result:
    v, lo, hi = df[value].astype(float), df[low_ref].astype(float), df[high_ref].astype(float)
    ok = (v >= lo) & (v <= hi)
    return _bounded_score(ok.all(), warn=~ok.all()), {"violations": int((~ok).sum())}

def derived_metric(df, name, expression, group_by=None) -> Result:
    """
    Creates/overwrites df[name] using a safe evaluator.
    Supports plain pandas .eval() expressions and the common pattern:
      "... / max(<col>, <eps>)"
    Returns pass unconditionally (it’s a preparatory step for later checks).
    """
    expr = expression.strip()

    # handle "... / max(col, eps)" robustly (since pandas.eval can't do Python max())
    import re
    m = re.search(r"^(?P<num>.+)/\s*max\(\s*(?P<col>[^,]+)\s*,\s*(?P<eps>[^)]+)\s*\)\s*$", expr)
    try:
        if m:
            num = df.eval(m.group("num"))
            col = df.eval(m.group("col").strip())
            eps = float(eval(m.group("eps").strip()))  # allow 1e-9 etc.
            den = np.maximum(col.astype(float), eps)
            df[name] = (num.astype(float) / den).astype(float)
        else:
            # best-effort generic eval
            df[name] = df.eval(expr).astype(float)
        return _res("pass", 1.0, created=name)
    except Exception as e:
        # don't crash the rule chain; mark as warn so later checks can decide
        return _res("warn", 0.6, note="derived_metric_eval_failed", expression=expression, error=str(e))

def variance_threshold(df, columns: List[str]=None, min_variance=None, max_var=None, column: str=None, min_var=None) -> Result:
    # accept single 'column' and alias 'min_var'
    if columns is None and column is not None:
        columns = [column]
    if columns is None:
        return _res("warn", 0.6, note="variance_no_columns")

    if min_variance is None and min_var is not None:
        min_variance = min_var

    det, ok_all = {}, True
    for c in columns:
        if c not in df:
            det[c] = {"variance": None, "missing": True}
            ok_all = False
            continue
        v = float(df[c].astype(float).var(ddof=0))
        cond = True
        if min_variance is not None: cond &= (v >= float(min_variance))
        if max_var is not None:      cond &= (v <= float(max_var))
        det[c] = {"variance": v}
        ok_all &= cond
    out = _bounded_score(ok_all); out["details"] = {"details": det}; return out

# ---------- B) Time / Period ----------

def monotonic_time_intents(df, column) -> Result:
    s = pd.to_datetime(df[column], errors="coerce")
    ok = s.is_monotonic_increasing
    return _bounded_score(ok)

def fiscal_year_close_present(df, period_column) -> Result:
    s = pd.to_datetime(df[period_column], errors="coerce").dropna()
    months = set(s.dt.month.tolist())
    ok = any(m in (3, 12) for m in months)
    out = _bounded_score(ok); out["details"] = {"months_present": sorted(months)}; return out

def period_gap_check(df, column, max_gap_months) -> Result:
    s = pd.to_datetime(df[column], errors="coerce").dropna().sort_values()
    gaps = s.diff().dt.days.dropna() / 30.0
    ok = gaps.le(max_gap_months).all() if len(gaps) else True
    out = _bounded_score(ok); out["details"] = {"max_gap_months": float(gaps.max() if len(gaps) else 0)}; return out

def period_alignment_multi(df, columns: List[str]) -> Result:
    sets = [set(pd.to_datetime(df[c], errors="coerce").dropna().dt.date) for c in columns]
    inter = set.intersection(*sets) if sets else set()
    ok = all(len(s) == len(inter) for s in sets) if sets else True
    out = _bounded_score(ok); out["details"] = {"common_periods": len(inter), "columns": columns}; return out

# ---------- C) Variance / Rolling ----------

def deviation_from_rolling_mean(df, column, window=3, max_dev_pct=0.2) -> Result:
    # Guard missing column → soft warn, keep pipeline alive
    if column not in df.columns:
        return _res("warn", 0.6, note="missing_column", missing=[column])

    x = df[column].astype(float)
    roll = x.rolling(int(window), min_periods=1).mean()
    dev = (x - roll).abs() / (roll.abs() + 1e-9)

    ok = (dev <= float(max_dev_pct)).all()
    out = _bounded_score(ok)
    out["details"] = {
        "window": int(window),
        "max_deviation_pct": float(dev.max() if len(dev) else 0.0),
    }
    return out


def variance_threshold(df, columns: List[str], min_variance=None, max_var=None) -> Result:
    det, ok_all = {}, True
    for c in columns:
        v = float(df[c].astype(float).var(ddof=0)) if c in df else 0.0
        cond = True
        if min_variance is not None: cond &= (v >= min_variance)
        if max_var is not None:      cond &= (v <= max_var)
        ok_all &= cond
        det[c] = {"variance": v}
    out = _bounded_score(ok_all); out["details"] = {"details": det}; return out

def rolling_mean_range(df, column, low_factor, high_factor, window=3) -> Result:
    x = df[column].astype(float)
    m = x.rolling(window, min_periods=window).mean()
    lo, hi = m * low_factor, m * high_factor
    mask = m.notna()
    ok = ((x[mask] >= lo[mask]) & (x[mask] <= hi[mask])).all() if mask.any() else True
    return _bounded_score(ok)

# ---------- D) Trends / Correlations / Lags ----------

def pct_change_range(df, column, min_abs_pct) -> Result:
    x = df[column].astype(float)
    ch = x.pct_change(fill_method=None).abs()
    ok = (ch.dropna() >= min_abs_pct).any()
    return _bounded_score(ok)

def trend_correlation_intents(df, left, right, min_corr=None, max_corr=None) -> Result:
    xl, xr = df[left].astype(float), df[right].astype(float)
    if xl.count() < 2 or xr.count() < 2:
        return _res("warn", 0.6, note="insufficient_points")
    corr = float(np.corrcoef(xl.fillna(0), xr.fillna(0))[0, 1])
    ok = True
    if min_corr is not None: ok &= corr >= min_corr
    if max_corr is not None: ok &= corr <= max_corr
    out = _bounded_score(ok); out["details"] = {"corr": corr}; return out

def lead_lag_correlation(df, left, right, max_lag_periods, min_corr) -> Result:
    xl = df[left].astype(float).reset_index(drop=True)
    xr = df[right].astype(float).reset_index(drop=True)
    best = -2.0
    for lag in range(max_lag_periods + 1):
        a = xl[lag:].reset_index(drop=True)
        b = xr[:len(a)].reset_index(drop=True)
        if len(a) >= 2:
            c = float(np.corrcoef(a.fillna(0), b.fillna(0))[0, 1])
            best = max(best, c)
    ok = best >= min_corr
    out = _bounded_score(ok); out["details"] = {"best_corr": best}; return out

def conditional_trend_flag_intents(df, left, right) -> Result:
    def _cond(x, token):
        d = x.diff().dropna()
        if token.startswith("increasing_"):
            n = int(token.split("_")[-1])
            return (d > 0).rolling(n).sum().max() >= n
        if token.startswith("decreasing_"):
            n = int(token.split("_")[-1])
            return (d < 0).rolling(n).sum().max() >= n
        return False
    ok = not (_cond(df[left["column"]].astype(float), left["condition"])
              and _cond(df[right["column"]].astype(float), right["condition"]))
    return _bounded_score(ok)

def correlation_threshold(df, x, y, min_corr=None, max_corr=None) -> Result:
    return trend_correlation_intents(df, x, y, min_corr, max_corr)

# ---------- E) Composition / Reconciliation / Mix ----------

def sum_reconciliation_intents(df, total, parts, tolerance_abs=0.01, group_by=None) -> Result:
    """
    Check that TOTAL ≈ sum(PARTS) within tolerance_abs (relative to TOTAL).
    Returns soft 'warn' if required columns are missing.
    """
    cols_needed = [total] + list(parts or [])
    present, miss = _require_columns(df, cols_needed)
    if not present:
        return _soft_missing(miss)

    def _eval(g):
        t = pd.to_numeric(g[total], errors="coerce").astype(float)
        s = pd.to_numeric(g[parts], errors="coerce").sum(axis=1).astype(float)
        err = (t - s).abs()
        bound = tolerance_abs * (t.abs() + 1e-9)
        ok = (err <= bound).fillna(True).all()
        det = {}
        if len(err):
            det = {
                "max_abs_err": float(np.nanmax(err)),
                "max_rel_err": float(np.nanmax((err / (t.abs() + 1e-9)).replace([np.inf, -np.inf], np.nan))),
            }
        out = _bounded_score(ok); out["details"] = det; return out

    # Aggregate by groups (tolerant)
    status_order = {"fail": 0, "warn": 1, "pass": 2}
    worst = 2; by = {}
    for k, g in _group(df, group_by):
        r = _eval(g)
        by[str(k)] = r["details"]
        worst = min(worst, status_order[r["status"]])
    status = {v:k for k,v in status_order.items()}[worst]
    score = {"pass":1.0,"warn":0.6,"fail":0.0}[status]
    return {"status": status, "score": score, "details": {"by_group": by}}


def mix_change_bounds(df, part, total, key, period, max_change_pct_of_baseline) -> Result:
    shares = df.assign(share=_safe_div(df[part].astype(float), df[total].astype(float)))
    baseline = shares.sort_values(period).groupby(key)["share"].first()
    shares = shares.join(baseline, on=key, rsuffix="_base")
    dev = (shares["share"] - shares["share_base"]).abs() / (shares["share_base"].abs() + 1e-9)
    ok = (dev <= max_change_pct_of_baseline).all()
    out = _bounded_score(ok); out["details"] = {"max_dev": float(dev.max())}; return out

def department_mix_change_bounds(df, dept_headcount, total_headcount, department, period, max_change_pct_of_baseline):
    return mix_change_bounds(df, dept_headcount, total_headcount, department, period, max_change_pct_of_baseline)

def ratio_consistency(df, numerator, denominator, tolerance_abs) -> Result:
    # soft-guard: if columns are missing, return a warn instead of crashing
    missing = [c for c in [numerator, denominator] if c not in df.columns]
    if missing:
        return _res("warn", 0.6, note="missing_columns", missing=missing)

    r = _safe_div(df[numerator].astype(float), df[denominator].astype(float))
    med = r.median()
    dev = np.abs(r - med) / (np.abs(med) + 1e-9)
    ok = (dev <= tolerance_abs).all()
    out = _bounded_score(ok)
    out["details"] = {"median_ratio": float(med), "max_dev": float(dev.max())}
    return out



# ---------- F) Duplicates / Presence / Policy / PII ----------

def presence_rate(df, flag, weight, min_rate) -> Result:
    w = df[weight].astype(float).fillna(0)
    f = df[flag].fillna(0).astype(int)
    rate = float((w * f).sum() / (w.sum() + 1e-9))
    ok = rate >= min_rate
    out = _bounded_score(ok); out["details"] = {"rate": rate}; return out

def duplicate_values(df, column) -> Result:
    dup = df[column].duplicated(keep=False)
    ok = (~dup).all()
    out = _bounded_score(ok, warn=not ok); out["details"] = {"duplicates": int(dup.sum())}; return out

def duplicate_values_multi(df, columns: List[str]) -> Result:
    dup = df.duplicated(subset=columns, keep=False)
    ok = (~dup).all()
    out = _bounded_score(ok, warn=not ok); out["details"] = {"duplicates": int(dup.sum())}; return out

def policy_presence(df, docs_required: List[str]) -> Result:
    have = set(df["policy_category_like"].astype(str).str.lower().unique()) if "policy_category_like" in df else set()
    missing = [d for d in docs_required if d not in have]
    ok = len(missing) == 0
    out = _bounded_score(ok, warn=not ok); out["details"] = {"missing": missing}; return out

def policy_age_max_days(df, max_days) -> Result:
    col = "policy_last_modified_days"
    if col not in df:
        return _res("warn", 0.6, note="missing_age_field")
    ok = (df[col].astype(float) <= max_days).all()
    out = _bounded_score(ok); out["details"] = {"max_age_days": int(df[col].max())}; return out

def pii_scan(df, tables="*") -> Result:
    email_like = df.apply(lambda c: c.astype(str).str.contains(r"[^@\s]+@[^@\s]+\.[^@\s]+", regex=True, na=False).any()
                          if c.dtype == object else False).any()
    phone_like = df.apply(lambda c: c.astype(str).str.contains(r"\+?\d[\d\-\s]{6,}", regex=True, na=False).any()
                          if c.dtype == object else False).any()
    ok = not (email_like or phone_like)
    out = _bounded_score(ok); out["details"] = {"email_like": bool(email_like), "phone_like": bool(phone_like)}; return out

# ---------- G) Identity / Patterns / Outliers ----------

def identical_rows_across_periods(df, column=None, min_consecutive=2) -> Result:
    s = df[column] if column else df.select_dtypes(include=[np.number]).sum(axis=1)
    run = (s.diff() == 0).astype(int)
    streak = int((run.groupby((run != run.shift()).cumsum()).cumcount() + 1).max() or 1)
    ok = streak < min_consecutive
    out = _bounded_score(ok); out["details"] = {"max_identical_streak": streak}; return out

def outlier_sigma_intents(df, column, sigma=3.0) -> Result:
    x = df[column].astype(float)
    mu, sd = x.mean(), x.std(ddof=0)
    z = np.abs((x - mu) / (sd + 1e-9))
    ok = (z <= sigma).all()
    out = _bounded_score(ok); out["details"] = {"max_z": float(z.max())}; return out

def non_negative(df, table_intent=None, columns: List[str]=None) -> Result:
    cols = columns or [c for c in df.columns if c.endswith("_like")]
    mask = pd.concat([(df[c] >= 0) for c in cols if c in df], axis=1)
    ok = mask.all(axis=1).all() if not mask.empty else True
    out = _bounded_score(ok); out["details"] = {"neg_rows": int((~mask.all(axis=1)).sum()) if not mask.empty else 0}; return out

def min_value(df, table_intent, column, min, group_by=None) -> Result:
    det, ok_all = {}, True
    for k, g in _group(df, group_by):
        series = g[column].astype(float)
        ok = (series >= min).all()
        ok_all &= ok
        det[str(k)] = {"min_seen": float(series.min())}
    out = _bounded_score(ok_all); out["details"] = {"details": det}; return out

# ---------- H) HR / Hiring ----------

def headcount_flow_consistency(df, headcount, hires, exits, transfers, group_by=None, tolerance_abs=0.05) -> Result:
    det, ok_all = {}, True
    for k, g in _group(df, group_by):
        hc = g[headcount].astype(float)
        delta = hc.diff().fillna(0)
        rhs = g[hires].astype(float) - g[exits].astype(float) + g[transfers].astype(float)
        err = np.abs(delta - rhs)
        tol = tolerance_abs * (np.abs(hc) + 1e-9)
        ok = (err <= tol)
        ok_all &= ok.all()
        det[str(k)] = {"max_err": float(err.max())}
    out = _bounded_score(ok_all); out["details"] = {"details": det}; return out

def attrition_rate_bounds(df, exits, headcount, period, annualize=True, low=0.0, high=0.30) -> Result:
    hc = df[headcount].astype(float)
    ex = df[exits].astype(float)
    rate = _safe_div(ex, (hc.shift(1).fillna(hc)))
    if annualize: rate *= 12
    ok = ((rate >= low) & (rate <= high)).all()
    out = _bounded_score(ok); out["details"] = {"min": float(rate.min()), "max": float(rate.max())}; return out

def band_variance_bound(df, value, band, max_std_over_mean, trim_pct=0.05) -> Result:
    det, ok_all = {}, True
    for b, g in df.groupby(band):
        v = g[value].astype(float).sort_values()
        n = len(v); t = int(n * trim_pct)
        v = v.iloc[t:n-t] if n > 2*t else v
        mean, std = v.mean(), v.std(ddof=0)
        ratio = float(std / (np.abs(mean) + 1e-9)) if mean != 0 else 0.0
        ok_all &= ratio <= max_std_over_mean
        det[str(b)] = {"std_over_mean": ratio}
    out = _bounded_score(ok_all); out["details"] = {"details": det}; return out

def median_gap_bound(df, value, group, max_gap_pct) -> Result:
    meds = df.groupby(group)[value].median().sort_values()
    gap = float((meds.max() - meds.min()) / (np.abs(meds.median()) + 1e-9)) if len(meds) else 0.0
    ok = gap <= max_gap_pct
    out = _bounded_score(ok); out["details"] = {"median_gap_pct": gap}; return out

def median_gap_bound_grouped(df, value, group, condition_groups, max_gap_pct) -> Result:
    subs = {name: df.query(spec["filter"]) for name, spec in condition_groups.items()}
    diffs = []
    grades = sorted(set().union(*(s[group].unique() for s in subs.values() if not s.empty)))
    for gname in grades:
        med = {name: s.loc[s[group] == gname, value].median() for name, s in subs.items()}
        if all(k in med and not np.isnan(med[k]) for k in ["fresh_hires", "tenured"]):
            diff = (med["fresh_hires"] - med["tenured"]) / (np.abs(med["tenured"]) + 1e-9)
            diffs.append(diff)
    gap = float(np.nanmax(np.abs(diffs)) if diffs else 0.0)
    ok = np.abs(gap) <= max_gap_pct
    out = _bounded_score(ok); out["details"] = {"max_gap_pct": gap}; return out

def promotion_rate_trend(df, tenure, promoted, period, min_trend_slope=0.0) -> Result:
    tb = pd.qcut(df[tenure].rank(method="first"), q=min(5, max(2, df.shape[0])), labels=False, duplicates="drop")
    grp = pd.DataFrame({"tenure_bin": tb, "promoted": df[promoted].astype(int)})
    rates = grp.groupby("tenure_bin")["promoted"].mean()
    x = np.arange(len(rates)); y = rates.values
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
    ok = slope >= min_trend_slope
    out = _bounded_score(ok); out["details"] = {"slope": slope}; return out

def training_hours_bounds(df, training_hours, headcount, period, low, high) -> Result:
    avg = _safe_div(df[training_hours].astype(float), df[headcount].astype(float))
    ok = ((avg >= low) & (avg <= high)).all()
    out = _bounded_score(ok); out["details"] = {"min_avg": float(avg.min()), "max_avg": float(avg.max())}; return out

def onboarding_completion_rate(df, numerator, denominator, min_rate) -> Result:
    rate = float(df[numerator].sum() / (df[denominator].sum() + 1e-9))
    ok = rate >= min_rate
    out = _bounded_score(ok); out["details"] = {"rate": rate}; return out

def document_metadata_check(df, required_fields: List[str]) -> Result:
    missing = [f for f in required_fields if f not in df.columns]
    ok = len(missing) == 0
    out = _bounded_score(ok, warn=not ok); out["details"] = {"missing_fields": missing}; return out

def band_alignment_check(df, experience, band, tolerance_bands=1) -> Result:
    exp = df[experience].astype(float)
    pred_band = pd.cut(exp, bins=[-1,2,5,8,12,99], labels=[1,2,3,4,5]).astype(int)
    gap = (pred_band - df[band].astype(int)).abs()
    ok = (gap <= tolerance_bands).all()
    out = _bounded_score(ok); out["details"] = {"max_band_gap": int(gap.max() if len(gap) else 0)}; return out

# ---------- I) Marketing / Data Quality Heuristics ----------

def mapping_consistency(df, left_key: List[str], right_key: str, max_conflict_rate=0.05) -> Result:
    grp = df[left_key + [right_key]].drop_duplicates()
    conflict = grp.groupby(left_key)[right_key].nunique() > 1
    rate = float(conflict.mean()) if len(conflict) else 0.0
    ok = rate <= max_conflict_rate
    out = _bounded_score(ok); out["details"] = {"conflict_rate": rate}; return out

def heuristic_flag(df, conditions: List[Dict[str, str]]) -> Result:
    flagged = False
    for cond in conditions:
        exprs = cond["exprs"]
        m = pd.Series(True, index=df.index)
        for e in exprs:
            m &= df.eval(e)
        if m.any():
            flagged = True
            break
    ok = not flagged
    out = _bounded_score(ok); out["details"] = {"flagged": flagged}; return out

# ---------- J) Generated / Semantic (delegated) ----------

def resume_business_match(resume_input, business_snapshot, max_suggestions=3, output_fields=None) -> Result:
    # Delegate to CPO generator/microservice; this stub just returns a pass with empty suggestions.
    return _res("pass", 1.0, suggestions=[], note="delegate_to_cpo_generator")

def skill_overlap_ratio(df=None, resume=None, jd=None, min_overlap=0.7) -> Result:
    return _res("warn", 0.6, note="compute_overlap_upstream")

def semantic_similarity_overlap(df, columns: List[str], threshold=0.85) -> Result:
    return _res("warn", 0.6, note="delegate_to_similarity_service")

# ---------- Registry & runner ----------

FORMULA_MAP = {
    # Ratios / equations / values
    "ratio_bounds_intents": ratio_bounds_intents,
    "ratio_bounds_intents_grouped": ratio_bounds_intents_grouped,
    "equation_intents": equation_intents,
    "equation_intents_tolerance": equation_intents_tolerance,
    "equation_tolerance_optional": equation_tolerance_optional,
    "equation_intents_absolute": equation_intents_absolute,
    "value_bounds": value_bounds,
    "value_in_range": value_in_range,
    "derived_metric": derived_metric,

    # Period / gaps
    "monotonic_time_intents": monotonic_time_intents,
    "fiscal_year_close_present": fiscal_year_close_present,
    "period_gap_check": period_gap_check,
    "period_alignment_multi": period_alignment_multi,

    # Variance / rolling
    "deviation_from_rolling_mean": deviation_from_rolling_mean,
    "variance_threshold": variance_threshold,
    "rolling_mean_range": rolling_mean_range,
    "variance_bounds": variance_threshold,  # alias

    # Trends / correlations / lags
    "pct_change_range": pct_change_range,
    "trend_correlation_intents": trend_correlation_intents,
    "lead_lag_correlation": lead_lag_correlation,
    "conditional_trend_flag_intents": conditional_trend_flag_intents,
    "correlation_threshold": correlation_threshold,

    # Composition / reconciliation / mix
    "sum_reconciliation_intents": sum_reconciliation_intents,
    "mix_change_bounds": mix_change_bounds,
    "department_mix_change_bounds": department_mix_change_bounds,
    "ratio_consistency": ratio_consistency,

    # Duplicates / presence / policy / pii
    "presence_rate": presence_rate,
    "duplicate_values": duplicate_values,
    "duplicate_values_multi": duplicate_values_multi,
    "policy_presence": policy_presence,
    "policy_age_max_days": policy_age_max_days,
    "pii_scan": pii_scan,

    # Identity / patterns / outliers
    "identical_rows_across_periods": identical_rows_across_periods,
    "outlier_sigma_intents": outlier_sigma_intents,
    "non_negative": non_negative,
    "min_value": min_value,

    # HR / Hiring
    "headcount_flow_consistency": headcount_flow_consistency,
    "attrition_rate_bounds": attrition_rate_bounds,
    "band_variance_bound": band_variance_bound,
    "median_gap_bound": median_gap_bound,
    "median_gap_bound_grouped": median_gap_bound_grouped,
    "promotion_rate_trend": promotion_rate_trend,
    "training_hours_bounds": training_hours_bounds,
    "onboarding_completion_rate": onboarding_completion_rate,
    "document_metadata_check": document_metadata_check,
    "band_alignment_check": band_alignment_check,

    # Marketing heuristics
    "mapping_consistency": mapping_consistency,
    "heuristic_flag": heuristic_flag,

    # Generated / semantic
    "resume_business_match": resume_business_match,
    "skill_overlap_ratio": skill_overlap_ratio,
    "semantic_similarity_overlap": semantic_similarity_overlap,
    
    #Additional Business Terms
    "revenue": ["Revenue","Sales","Booked Revenue","Turnover"],
    "orders": ["Orders","Total Orders","Completed Orders","Shipments"],
    "marketing_spend": ["Marketing Spend","Ad Spend","Total Spend","Spend"],
    "gross_margin_pct": ["Gross Margin %","GM%","GrossMarginPct"],
    "operating_cashflow": ["Operating Cash Flow","Operating Cashflow","CFO_Operating"],
    "headcount": ["Headcount","Employees","Total Headcount"],
    "attrition_rate": ["Attrition Rate","Attrition%","Exits Rate"],
    "leads": ["Leads","Total Leads","Unique Leads"],
    "sql": ["SQL","Qualified Leads","Sales Qualified Leads"],
}

def run_check(check_spec, df):
    """
    Dispatch a rule safely:
    - normalize common kw aliases (tolerance/tol_abs -> tolerance_abs)
    - drop unknown kwargs for the target function (unless it accepts **kwargs)
    - never crash the pipeline; return soft 'warn' on exceptions
    """
    ctype = check_spec.get("type")
    fn = FORMULA_MAP.get(ctype)
    if fn is None:
        return _res("warn", 0.6, note="unknown_check_type", type=ctype)

    # copy & normalize kwargs
    kwargs = {k: v for k, v in check_spec.items() if k != "type"}

    # normalize tolerance names
    if "tolerance_abs" not in kwargs:
        if "tolerance" in kwargs:
            kwargs["tolerance_abs"] = kwargs.pop("tolerance")
        elif "tol_abs" in kwargs:
            kwargs["tolerance_abs"] = kwargs.pop("tol_abs")

    # If the target function doesn't accept a kw, drop it.
    try:
        import inspect
        sig = inspect.signature(fn)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_kwargs:
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        # if inspection fails, proceed (fn may accept **kwargs)
        pass

    try:
        return fn(df=df, **kwargs)
    except Exception as e:
        return _res("warn", 0.4, note="check_threw_exception", type=ctype, error=str(e))

