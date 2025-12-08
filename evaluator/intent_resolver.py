# -*- coding: utf-8 -*-
"""
Intent resolver
---------------
Converts arbitrary column names in incoming DataFrames into stable *intent* columns
(e.g., `revenue_like`, `cogs_like`, `period_like`, etc.) that our rules reference.

Features:
- Case/space-insensitive alias matching (robust to "Total Revenue", "total_revenue", etc.)
- Per-brain alias maps (CFO/CMO/COO/CHRO/CPO)
- Safe numeric casting helper (only when needed by checks later)
- Light synthesis of common derived intents when inputs exist
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

# -------------------------
# Helpers
# -------------------------
def _canon(name: str) -> str:
    """Lowercase + remove spaces/underscores for tolerant matching."""
    return (name or "").strip().lower().replace(" ", "").replace("_", "")

def _build_lookup(df: pd.DataFrame) -> Dict[str, str]:
    """Map canonical -> original column name for fast reverse lookup."""
    return {_canon(c): c for c in df.columns}

def _apply_aliases(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    """
    If any candidate column exists, copy it into the target intent column.
    Matching is case/space-insensitive. Does NOT overwrite if intent already present.
    """
    out = df.copy()
    look = _build_lookup(out)
    for intent, candidates in aliases.items():
        if intent in out.columns:
            continue
        for cand in candidates:
            ckey = _canon(cand)
            if ckey in look:
                out[intent] = out[look[ckey]]
                break
    return out

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Best-effort numeric coercion for listed columns (if present).
    Non-convertible values become NaN; column left untouched if absent.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _ensure_period(df: pd.DataFrame, period_col: str = "period_like") -> pd.DataFrame:
    """
    Normalize the period column for stable grouping/rolling:
    - accept real datetimes, strings like '2024-03', '202403', '20240315', '2024-Q1', '2024-W05'
    - accept Excel serial dates (numeric) and coerce
    - output canonical strings (YYYY-MM or YYYY-MM-DD)
    """
    out = df.copy()
    if period_col not in out.columns:
        return out

    s = out[period_col]

    # If numeric like Excel serial or yyyymm/yyyymmdd
    if pd.api.types.is_numeric_dtype(s):
        try:
            dt = pd.to_datetime(s, origin="1899-12-30", unit="D", errors="coerce")  # Excel base
            # Fallback: treat as yyyymm or yyyymmdd if Excel parse failed a lot
            if dt.isna().mean() > 0.6:
                s_str = s.astype("Int64").astype(str)
                # yyyymmdd → date
                mask_ymd = s_str.str.len().eq(8) & s_str.str.match(r"^\d{8}$")
                dt = pd.to_datetime(s_str.where(mask_ymd, pd.NA), format="%Y%m%d", errors="coerce")
                # yyyymm → month start
                mask_ym = s_str.str.len().eq(6) & s_str.str.match(r"^\d{6}$")
                dt = dt.fillna(pd.to_datetime(s_str.where(mask_ym, pd.NA) + "01", format="%Y%m%d", errors="coerce"))
        except Exception:
            dt = pd.to_datetime(s, errors="coerce")
    else:
        # Strings / mixed
        s_str = s.astype(str).str.strip()

        # Recognize YYYY-Qn and YYYY-Www
        q_mask = s_str.str.match(r"^\d{4}-?[Qq][1-4]$")
        w_mask = s_str.str.match(r"^\d{4}-?[Ww]\d{1,2}$")

        dt = pd.to_datetime(s_str, errors="coerce")
        # Try YYYYMMDD
        dt = dt.fillna(pd.to_datetime(s_str.where(s_str.str.match(r"^\d{8}$"), pd.NA), format="%Y%m%d", errors="coerce"))
        # Try YYYYMM -> first day of month
        dt = dt.fillna(pd.to_datetime(s_str.where(s_str.str.match(r"^\d{6}$"), pd.NA) + "01",
                                      format="%Y%m%d", errors="coerce"))
        # Quarter → first day of quarter
        if q_mask.any():
            qdt = pd.PeriodIndex(s_str.where(q_mask).str.replace("-", "", regex=False).str.upper(),
                                 freq="Q").start_time
            dt = dt.fillna(qdt)
        # ISO week → Monday of that ISO week
        if w_mask.any():
            w = s_str.where(w_mask).str.upper().str.replace("-", "", regex=False)
            # YYYYWww → parse with ISO week
            tmp = pd.to_datetime(w.str[:4] + "-W" + w.str[-2:] + "-1", format="%G-W%V-%u", errors="coerce")
            dt = dt.fillna(tmp)

    # Final formatting
    try:
        if hasattr(dt, "dt"):
            # If any day precision → YYYY-MM-DD else YYYY-MM
            if dt.dt.day.nunique(dropna=True) > 1:
                out[period_col] = dt.dt.strftime("%Y-%m-%d")
            else:
                out[period_col] = dt.dt.strftime("%Y-%m")
        else:
            out[period_col] = s.astype(str)
    except Exception:
        out[period_col] = s.astype(str)

    return out


# -------------------------
# Generic aliases (brain-agnostic)
# -------------------------
GENERIC_ALIASES: Dict[str, List[str]] = {
    "period_like":        ["period", "date", "month", "reporting_period","posting_date", "txn_date", "transaction_date",
        "fiscal_period", "fiscal_month", "yearmonth", "ym"],
    "channel_like":       ["channel", "utm_channel"],
    "source_like":        ["source", "utm_source"],
    "medium_like":        ["medium", "utm_medium"],
    "campaign_like":      ["campaign", "utm_campaign"],

    # finance-ish generic
    "revenue_like":       ["revenue", "sales", "booked_revenue", "total_revenue"],
    "cogs_like":          ["cogs", "costofgoodssold", "cost_of_goods_sold"],

    # people generic
    "headcount_like":     ["headcount", "employees_total", "employee_count"],
}

# -------------------------
# Per-brain alias maps
# -------------------------
CFO_ALIASES = {
    "booked_revenue_like": ["revenue_like", "revenue", "sales"],
    "ltv_like":            ["ltv", "lifetimevalue"],
    "cac_like":            ["cac", "customeraqc", "acquisitioncost"],
    # common tables often vary; these help intent mapping
    "cash_in_like":        ["cashin", "cash_in"],
    "cash_out_like":       ["cashout", "cash_out"],
}

CMO_ALIASES = {
    "impressions_like":        ["impressions"],
    "clicks_like":             ["clicks"],
    "leads_like":              ["leads", "conversions"],
    "spend_like":              ["spend", "cost", "adspend", "marketing_spend"],
    "total_spend_like":        ["total_spend", "total_cost", "overall_spend"],
    "attributed_revenue_like": ["attributed_revenue", "conv_value", "purchase_value"],
    "sessions_like":           ["sessions", "visits"],
    "utm_present_flag_like":   ["utm_present", "has_utm"],
}

COO_ALIASES = {
    "output_units_like": [
        "output_units", "units_out", "produced_units", "completed_units",
        "throughput_units", "good_units", "units_produced"
    ],
    "input_units_like": [
        "input_units", "units_in", "raw_units", "started_units",
        "units_started", "materials_units"
    ],
    "capacity_used_like":      ["capacity_used", "utilization", "utilisation"],
    "capacity_available_like": ["capacity_available", "available_capacity"],
    "downtime_hours_like":     ["downtime_hours", "downtime", "mttr_hours"],
    "available_hours_like":    ["available_hours", "planned_hours"],
    "output_per_employee":     ["output_per_employee", "rev_per_employee", "revenue_per_employee"
    ],
    "orders_completed_like": [
        "orders_completed", "completed_orders", "orders_done",
        "orders_fulfilled", "shipments_completed", "closed_orders"
    ],
    "orders_started_like": [
        "orders_started", "orders_created", "new_orders", "orders_opened"
    ],
}

CHRO_ALIASES = {
    "headcount_total_like": ["headcount_like", "headcount", "employees_total"],
    "new_hires_like":       ["new_hires", "joins", "hires"],
    "exits_like":           ["exits", "separations", "attrition_count"],
    "job_openings_like":    ["job_openings", "open_roles", "requisitions_open"],
}

CPO_ALIASES = {
    "salary_like":                        ["salary", "ctc", "compensation"],
    "hire_type_like":                     ["hire_type", "employment_type"],
    "tenure_like":                        ["tenure_months", "months_in_company", "tenure"],
    "grade_like":                         ["grade", "band", "level"],
    "join_date_like":                     ["join_date", "doj", "date_of_joining"],
    "requisition_date_like":              ["requisition_date", "req_date", "opened_on"],
    "total_revenue_like":                 ["total_revenue", "revenue"],
    "headcount_total_like":               ["headcount_like", "headcount", "employees_total"],
    # output per employee (productivity/revenue per head)
    "output_per_employee": [
        "output_per_employee", "rev_per_employee", "revenue_per_employee",
        "productivity_per_head", "output_per_head"
    ],
    "expected_output_per_employee_like": [
        "expected_output_per_employee", "monthly_output_expected", "target_output_per_employee"
    ],
}

# Columns we often want numeric if present (kept light/on-demand)
NUMERIC_HINTS = list({
    # CFO/Finance
    "revenue_like", "booked_revenue_like", "cogs_like", "ltv_like", "cac_like",
    "cash_in_like", "cash_out_like",
    # CMO/Marketing
    "impressions_like", "clicks_like", "leads_like", "spend_like", "attributed_revenue_like", "sessions_like",
    # COO/Ops
    "output_units_like", "input_units_like", "capacity_used_like", "capacity_available_like","orders_completed_like", "orders_started_like",
    "downtime_hours_like", "available_hours_like", "output_per_employee",
    # CHRO/People Ops
    "headcount_like", "headcount_total_like", "new_hires_like", "exits_like", "job_openings_like",
    # CPO/Talent
    "salary_like", "tenure_like", "total_revenue_like",
})

# -------------------------
# Resolver (public)
# -------------------------
def resolve_intents(df: pd.DataFrame, brain: str) -> pd.DataFrame:
    """
    Returns a DataFrame with best-effort intent columns present.
    - Applies generic and brain-specific aliases
    - Synthesizes required intents when possible (period_like, output_per_employee)
    - Normalizes period formatting
    - Coerces numeric types for known numeric intents when present
    """
    if df is None or df.empty:
        return df

    resolved = _apply_aliases(df, GENERIC_ALIASES)

    b = (brain or "").strip().lower()
    if b == "cfo":
        resolved = _apply_aliases(resolved, CFO_ALIASES)
    elif b == "cmo":
        resolved = _apply_aliases(resolved, CMO_ALIASES)
    elif b == "coo":
        resolved = _apply_aliases(resolved, COO_ALIASES)
    elif b == "chro":
        resolved = _apply_aliases(resolved, CHRO_ALIASES)
    elif b == "cpo":
        resolved = _apply_aliases(resolved, CPO_ALIASES)

    # --- Synthesize period_like if missing (before normalization) ---
    if "period_like" not in resolved.columns:
        try:
            # Prefer a unique first column as a stable "period"
            if len(resolved.columns) > 0:
                first = resolved.columns[0]
                if resolved[first].nunique(dropna=False) == len(resolved):
                    resolved["period_like"] = resolved[first].astype(str)
                else:
                    resolved["period_like"] = pd.RangeIndex(1, len(resolved) + 1).astype(str)
            else:
                resolved["period_like"] = pd.RangeIndex(1, len(resolved) + 1).astype(str)
        except Exception:
            resolved["period_like"] = pd.RangeIndex(1, len(resolved) + 1).astype(str)

    # --- Light synthesis: output_per_employee if ingredients exist ---
    if "output_per_employee" not in resolved.columns:
        if "total_revenue_like" in resolved.columns and "headcount_total_like" in resolved.columns:
            try:
                den = np.maximum(pd.to_numeric(resolved["headcount_total_like"], errors="coerce"), 1e-9)
                num = pd.to_numeric(resolved["total_revenue_like"], errors="coerce")
                resolved["output_per_employee"] = (num / den)
            except Exception:
                pass  # checks will soft-warn if still missing

    # Normalize period for grouping/sorting
    resolved = _ensure_period(resolved, "period_like")

    # Coerce commonly numeric intents where present
    resolved = _ensure_numeric(resolved, NUMERIC_HINTS)

    return resolved
