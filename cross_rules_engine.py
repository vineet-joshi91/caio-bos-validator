# -*- coding: utf-8 -*-
"""
CAIO-BOS native cross-brain rules engine — FULL 25 RULE SET.

Outputs for every rule are one of: pass | warn | fail | na | error.
No 'todo'.

Heuristics are intentionally light but practical. If your sheet/column
names differ, expand evaluator.formula_registry.FORMULA_MAP aliases.
"""

from __future__ import annotations
import os, glob, math, traceback
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import yaml

# ---------------------------------------------
# Optional alias map from your repo (used if present)
# ---------------------------------------------
FORMULA_MAP: Dict[str, List[str]] = {}
try:
    from evaluator.formula_registry import FORMULA_MAP as _MAP  # your file path
    if isinstance(_MAP, dict):
        FORMULA_MAP = _MAP
except Exception:
    FORMULA_MAP = {}

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def _lc(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")

def _all_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {_lc(c): c for c in df.columns}

def _find_one(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    """Resolve a single column name using FORMULA_MAP and fuzzy matching."""
    if df is None or df.empty:
        return None
    lc_map = _all_columns(df)

    # candidate alias pool
    pool: List[str] = []
    for k in keys:
        pool.append(k)
        for a in FORMULA_MAP.get(k, []):
            pool.append(a)
    pool = list({_lc(x) for x in pool})

    # exact
    for a in pool:
        if a in lc_map:
            return lc_map[a]
    # contains
    for a in pool:
        for k, orig in lc_map.items():
            if a in k or k in a:
                return orig
    return None

def _find_any(df: pd.DataFrame, key_groups: List[List[str]]) -> Optional[str]:
    """Try a series of key lists until one resolves."""
    for group in key_groups:
        col = _find_one(df, group)
        if col:
            return col
    return None

def _period_col(df: pd.DataFrame) -> Optional[str]:
    return _find_one(df, ["period", "date", "month", "year_month", "fiscal_period"])

def _pct_change(s: pd.Series) -> pd.Series:
    with pd.option_context("mode.use_inf_as_na", True):
        return pd.to_numeric(s, errors="coerce").astype(float).pct_change().replace([pd.NA, pd.NaT], 0).fillna(0)

def _growth_bool(s: pd.Series, up_thresh=0.1, down_thresh=-0.05) -> Tuple[pd.Series, pd.Series]:
    p = _pct_change(s)
    return (p > up_thresh), (p < down_thresh)

def _score(status: str) -> float:
    return {"pass": 1.0, "warn": 0.6, "fail": 0.0, "na": 0.0, "error": 0.0}.get(status, 0.0)

def _df_or_none(dfs: Dict[str, pd.DataFrame], key: str) -> Optional[pd.DataFrame]:
    df = dfs.get(key)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()
    return None

def _sort_by_period(df: pd.DataFrame) -> pd.DataFrame:
    p = _period_col(df)
    if not p:
        return df
    try:
        return df.assign(__dt=pd.to_datetime(df[p], errors="coerce")).sort_values("__dt").drop(columns="__dt")
    except Exception:
        return df

def _na(msg: str) -> Tuple[str, float, str]:
    return ("na", _score("na"), msg)

# ---------------------------------------------
# Rule dataclass + YAML loader
# ---------------------------------------------
@dataclass
class Rule:
    rule_id: str
    title: str
    severity: str
    file_path: str

def _load_rules(path: str) -> List[Rule]:
    files = sorted(glob.glob(os.path.join(path, "*.yaml")))
    rules: List[Rule] = []
    for p in files:
        with open(p, "r", encoding="utf-8") as fh:
            y = yaml.safe_load(fh) or {}
        rules.append(
            Rule(
                rule_id=str(y.get("rule_id") or y.get("id") or os.path.basename(p)),
                title=str(y.get("title", "")),
                severity=str(y.get("severity", "warn")),
                file_path=os.path.normpath(p),
            )
        )
    return rules

# ---------------------------------------------
# Evaluators (ALL 25)
# ---------------------------------------------
def _eval_101(dfs):
    cmo, coo, cfo = _df_or_none(dfs, "cmo"), _df_or_none(dfs, "coo"), _df_or_none(dfs, "cfo")
    if not (cmo is not None and coo is not None and cfo is not None):
        return _na("Requires CMO, COO, CFO")
    spend = _find_one(cmo, ["marketing_spend","ad_spend","total_spend","spend"])
    orders = _find_one(coo, ["orders","total_orders","completed_orders","shipments"])
    revenue = _find_one(cfo, ["revenue","sales","booked_revenue","turnover"])
    if not all([spend, orders, revenue]): return _na("Columns not found (spend/orders/revenue)")
    s_up, _ = _growth_bool(cmo[spend])
    _, o_down = _growth_bool(coo[orders])
    _, r_down = _growth_bool(cfo[revenue])
    bad = (s_up & o_down & r_down).sum()
    n = max(len(cmo), len(coo), len(cfo))
    if bad >= max(2, math.ceil(0.3*n)): return ("fail", _score("fail"), f"{bad} adverse funnel periods")
    if bad > 0: return ("warn", _score("warn"), f"{bad} adverse funnel periods")
    return ("pass", _score("pass"), "No adverse funnel pattern")

def _eval_102(dfs):
    cmo, cfo = _df_or_none(dfs,"cmo"), _df_or_none(dfs,"cfo")
    if cmo is None or cfo is None: return _na("Requires CMO, CFO")
    attr = _find_one(cmo, ["attributed_revenue","platform_revenue","mkt_attributed_revenue"])
    rev  = _find_one(cfo, ["revenue","sales","booked_revenue","turnover"])
    if not (attr and rev): return _na("Columns not found (attributed_revenue/revenue)")
    a = pd.to_numeric(cmo[attr], errors="coerce").sum(skipna=True)
    r = pd.to_numeric(cfo[rev], errors="coerce").sum(skipna=True)
    if pd.isna(a) or pd.isna(r): return _na("Insufficient numeric data")
    if a > r*1.02: return ("fail", _score("fail"), f"Attributed {a:,.0f} > Revenue {r:,.0f}")
    if a > r*0.98: return ("warn", _score("warn"), f"Attributed ≈ Revenue ({a:,.0f} vs {r:,.0f})")
    return ("pass", _score("pass"), "Attributed ≤ Revenue")

def _eval_103(dfs):
    # Marketing payback months ~ Spend / (AttributedRev * margin or CFO gross profit * share)
    cmo, cfo = _df_or_none(dfs,"cmo"), _df_or_none(dfs,"cfo")
    if cmo is None or cfo is None: return _na("Requires CMO, CFO")
    spend = _find_one(cmo, ["marketing_spend","ad_spend","total_spend","spend"])
    attr  = _find_one(cmo, ["attributed_revenue","platform_revenue","mkt_attributed_revenue"])
    gpct  = _find_one(cfo, ["gross_margin_pct","gross_margin_percent","gm_pct"])
    if not (spend and attr): return _na("Columns not found (spend/attributed_revenue)")
    S = pd.to_numeric(cmo[spend], errors="coerce").fillna(0)
    A = pd.to_numeric(cmo[attr], errors="coerce").fillna(0)
    if gpct and gpct in cfo.columns:
        GM = pd.to_numeric(cfo[gpct], errors="coerce").fillna(0.4)  # default 40%
        g = (A * GM.iloc[:len(A)].reindex_like(A, fill_value=GM.mean())).replace(0, pd.NA)
    else:
        g = A.replace(0, pd.NA)
    payback = (S / g).replace([pd.NA, pd.NaT, pd.Series([float("inf")]*len(S))], pd.NA).fillna(float("inf"))
    too_low = (payback < 0.2).sum()
    too_high = (payback > 24).sum()
    if too_low + too_high >= max(2, math.ceil(0.3*len(payback))): 
        return ("fail", _score("fail"), f"Unrealistic payback in {too_low+too_high} periods")
    if too_low + too_high > 0:
        return ("warn", _score("warn"), f"{too_low} very-low and {too_high} very-high payback periods")
    return ("pass", _score("pass"), "Payback looks realistic")

def _eval_104(dfs):
    coo, cfo = _df_or_none(dfs,"coo"), _df_or_none(dfs,"cfo")
    if coo is None or cfo is None: return _na("Requires COO, CFO")
    orders = _find_one(coo, ["orders","total_orders","completed_orders","shipments"])
    returns = _find_one(coo, ["returns","refunds","returned_orders"])
    revenue = _find_one(cfo, ["revenue","sales","booked_revenue","turnover"])
    if not (orders and returns and revenue): return _na("Columns not found (orders/returns/revenue)")
    o_up,_ = _growth_bool(coo[orders])
    r_up,_ = _growth_bool(coo[returns])
    _, rev_down = _growth_bool(cfo[revenue])
    bad = (o_up & r_up & rev_down).sum()
    if bad >= 2: return ("warn", _score("warn"), f"{bad} periods: returns dampened revenue despite order growth")
    return ("pass", _score("pass"), "No strong returns-dampening pattern")

def _eval_105(dfs):
    chro, cpo = _df_or_none(dfs,"chro"), _df_or_none(dfs,"cpo")
    if chro is None or cpo is None: return _na("Requires CHRO, CPO")
    head = _find_one(chro, ["headcount","total_headcount","employees"])
    hires = _find_one(cpo, ["new_hires","hiring","offers_accepted","joined"])
    exits = _find_one(chro, ["attrition","exits","separations","leavers"])
    if not (head and hires and exits): return _na("Columns not found (headcount/hires/exits)")
    H = pd.to_numeric(chro[head], errors="coerce").fillna(method="ffill").fillna(0)
    J = pd.to_numeric(cpo[hires], errors="coerce").reindex(range(len(H)), fill_value=0)
    X = pd.to_numeric(chro[exits], errors="coerce").reindex(range(len(H)), fill_value=0)
    dH = H.diff().fillna(0)
    diff = (J - X) - dH
    off = (diff.abs() > max(1, 0.2 * (H.mean() if H.mean() else 1))).sum()
    if off >= 2: return ("warn", _score("warn"), f"{off} periods where hires-exits ≠ Δheadcount")
    return ("pass", _score("pass"), "Hires - exits aligns with net headcount change")

def _eval_106(dfs):
    cfo, cpo = _df_or_none(dfs,"cfo"), _df_or_none(dfs,"cpo")
    if cfo is None or cpo is None: return _na("Requires CFO, CPO")
    runway = _find_one(cfo, ["runway_months","months_of_runway","cash_runway_months"])
    payroll = _find_one(cfo, ["payroll_cost","total_payroll","sga_payroll"])
    hires = _find_one(cpo, ["new_hires","hiring","offers_accepted","joined"])
    if not (runway and payroll and hires): return _na("Columns not found (runway/payroll/hires)")
    R = pd.to_numeric(cfo[runway], errors="coerce").fillna(0)
    P = pd.to_numeric(cfo[payroll], errors="coerce").fillna(0)
    H = pd.to_numeric(cpo[hires], errors="coerce").fillna(0)
    _, r_down = _growth_bool(R, up_thresh=0.05, down_thresh=-0.02)
    p_up,_ = _growth_bool(P)
    h_up,_ = _growth_bool(H)
    bad = (r_down & p_up & h_up).sum()
    if bad >= 2: return ("fail", _score("fail"), "Runway falling while payroll & hiring rise")
    if bad > 0: return ("warn", _score("warn"), "Runway pressure with payroll/hiring up")
    return ("pass", _score("pass"), "Runway vs payroll/hiring looks OK")

def _eval_107(dfs):
    cmo, cpo, coo = _df_or_none(dfs,"cmo"), _df_or_none(dfs,"cpo"), _df_or_none(dfs,"coo")
    if not (cmo is not None and cpo is not None and coo is not None): return _na("Requires CMO, CPO, COO")
    spend = _find_one(cmo, ["marketing_spend","ad_spend","total_spend","spend"])
    hc = _find_one(cpo, ["headcount","total_headcount","employees"])
    orders = _find_one(coo, ["orders","total_orders","completed_orders","shipments"])
    if not (spend and hc and orders): return _na("Columns not found (spend/headcount/orders)")
    s_up,_ = _growth_bool(cmo[spend])
    h_up,_ = _growth_bool(cpo[hc])
    _, o_down = _growth_bool(coo[orders])
    bad = (s_up & h_up & o_down).sum()
    if bad >= 2: return ("warn", _score("warn"), f"{bad} periods show efficiency paradox")
    return ("pass", _score("pass"), "No persistent efficiency paradox")

def _eval_108(dfs):
    cfo = _df_or_none(dfs,"cfo")
    if cfo is None: return _na("Requires CFO")
    op  = _find_one(cfo, ["operating_cashflow","operating_cash_flow","cashflow_operating"])
    inv = _find_one(cfo, ["investing_cashflow","investing_cash_flow","cashflow_investing"])
    fin = _find_one(cfo, ["financing_cashflow","financing_cash_flow","cashflow_financing"])
    net = _find_one(cfo, ["net_change_in_cash","net_change_cash","net_cash_change"])
    if not all([op, inv, fin, net]): return _na("Required columns not found (op, inv, fin, net)")
    co = pd.to_numeric(cfo[op], errors="coerce").fillna(0)
    ci = pd.to_numeric(cfo[inv], errors="coerce").fillna(0)
    cf = pd.to_numeric(cfo[fin], errors="coerce").fillna(0)
    cn = pd.to_numeric(cfo[net], errors="coerce").fillna(0)
    delta = (co + ci + cf) - cn
    tol = (cn.abs()*0.05).clip(lower=1.0)
    bad = (delta.abs() > tol).sum()
    if bad >= max(1, math.ceil(0.2*len(cn))): return ("fail", _score("fail"), f"{bad} periods violate cashflow identity")
    if bad > 0: return ("warn", _score("warn"), f"{bad} borderline periods")
    return ("pass", _score("pass"), "Cashflow identity holds")

def _eval_109(dfs):
    cmo, cfo = _df_or_none(dfs,"cmo"), _df_or_none(dfs,"cfo")
    if cmo is None or cfo is None: return _na("Requires CMO, CFO")
    spend = _find_one(cmo, ["marketing_spend","ad_spend","total_spend","spend"])
    gmpct = _find_one(cfo, ["gross_margin_pct","gross_margin_percent","gm_pct"])
    if not (spend and gmpct): return _na("Columns not found (spend/gross_margin_pct)")
    s_up,_ = _growth_bool(cmo[spend], up_thresh=0.05)
    _, gm_down = _growth_bool(cfo[gmpct], down_thresh=-0.01)
    bad = (s_up & gm_down).sum()
    if bad >= 2: return ("warn", _score("warn"), f"{bad} periods of margin compression with rising spend")
    return ("pass", _score("pass"), "No persistent margin compression from spend")

def _eval_110(dfs):
    chro, coo = _df_or_none(dfs,"chro"), _df_or_none(dfs,"coo")
    if chro is None or coo is None: return _na("Requires CHRO, COO")
    attr = _find_one(chro, ["attrition_rate","attrition","exits_rate"])
    backlog = _find_one(coo, ["backlog","order_backlog","pending_orders","open_orders"])
    if not (attr and backlog): return _na("Columns not found (attrition/backlog)")
    a_up,_ = _growth_bool(chro[attr], up_thresh=0.02)
    b_up,_ = _growth_bool(coo[backlog], up_thresh=0.05)
    bad = (a_up & b_up).sum()
    if bad >= 2: return ("warn", _score("warn"), f"{bad} periods where attrition likely driving backlog")
    return ("pass", _score("pass"), "Attrition vs backlog not strongly linked")

def _eval_111(dfs):
    chro, coo = _df_or_none(dfs,"chro"), _df_or_none(dfs,"coo")
    if chro is None or coo is None: return _na("Requires CHRO, COO")
    training = _find_one(chro, ["training_hours","training_hours_per_emp","lnd_hours"])
    defects  = _find_one(coo, ["defect_rate","defects","quality_defect_rate"])
    if not (training and defects): return _na("Columns not found (training/defects)")
    t_up,_ = _growth_bool(chro[training], up_thresh=0.05)
    d_up,_ = _growth_bool(coo[defects], up_thresh=0.01)
    # if training up but defects not falling
    not_helping = (t_up & d_up).sum()
    if not_helping >= 2: return ("warn", _score("warn"), f"{not_helping} periods: training up but defects up")
    return ("pass", _score("pass"), "Training effect on defects acceptable")

def _eval_112(dfs):
    coo, cfo = _df_or_none(dfs,"coo"), _df_or_none(dfs,"cfo")
    if coo is None or cfo is None: return _na("Requires COO, CFO")
    inv = _find_one(coo, ["inventory","inventory_value","stock_value","inventory_level"])
    op  = _find_one(cfo, ["operating_cashflow","operating_cash_flow","cashflow_operating"])
    if not (inv and op): return _na("Columns not found (inventory/operating_cashflow)")
    inv_up,_ = _growth_bool(coo[inv], up_thresh=0.05)
    _, op_down = _growth_bool(cfo[op], down_thresh=-0.05)
    bad = (inv_up & op_down).sum()
    if bad >= 2: return ("warn", _score("warn"), f"{bad} periods show inventory rising while op CF falls")
    return ("pass", _score("pass"), "No sustained inventory drag")

def _eval_113(dfs):
    chro, cfo = _df_or_none(dfs,"chro"), _df_or_none(dfs,"cfo")
    if chro is None or cfo is None: return _na("Requires CHRO, CFO")
    head = _find_one(chro, ["headcount","total_headcount","employees"])
    payroll = _find_one(cfo, ["payroll_cost","total_payroll","sga_payroll"])
    revenue = _find_one(cfo, ["revenue","sales","booked_revenue","turnover"])
    if not (head and payroll and revenue): return _na("Columns not found (headcount/payroll/revenue)")
    H = pd.to_numeric(chro[head], errors="coerce").replace(0, pd.NA).fillna(method="ffill")
    Rev = pd.to_numeric(cfo[revenue], errors="coerce").fillna(0)
    RPE = Rev / H.replace(0, pd.NA)
    rp_down = (_pct_change(RPE) < -0.15).sum()
    pay_up  = (_pct_change(pd.to_numeric(cfo[payroll], errors="coerce")) > 0.10).sum()
    if rp_down >= 2 and pay_up >= 2: return ("warn", _score("warn"), "Revenue per employee falling while payroll grows")
    return ("pass", _score("pass"), "Headcount, payroll, revenue broadly aligned")

def _eval_114(dfs):
    chro, cmo = _df_or_none(dfs,"chro"), _df_or_none(dfs,"cmo")
    if chro is None or cmo is None: return _na("Requires CHRO, CMO")
    attr = _find_one(chro, ["attrition_rate","attrition","exits_rate"])
    rec  = _find_one(cmo, ["recruitment_spend","talent_spend","hr_marketing_spend","hiring_spend"])
    if not rec: 
        # allow recruitment spend in CPO too
        cpo = _df_or_none(dfs,"cpo")
        rec = _find_one(cpo, ["recruitment_spend","talent_spend","hiring_spend"]) if cpo is not None else None
    if not (attr and rec): return _na("Columns not found (attrition/recruitment_spend)")
    a_up,_ = _growth_bool(chro[attr], up_thresh=0.02)
    r_up,_ = _growth_bool((cmo if rec in (cmo.columns if cmo is not None else []) else cpo)[rec], up_thresh=0.10)
    bad = (a_up & r_up).sum()
    if bad >= 2: return ("warn", _score("warn"), f"{bad} periods: attrition & recruitment spend rising together")
    return ("pass", _score("pass"), "Attrition vs recruitment spend acceptable")

def _eval_115(dfs):
    cmo = _df_or_none(dfs,"cmo")
    if cmo is None: return _na("Requires CMO")
    paid  = _find_one(cmo, ["paid_traffic","paid_sessions","ads_clicks"])
    organic = _find_one(cmo, ["organic_traffic","organic_sessions","seo_sessions"])
    if not (paid and organic): return _na("Columns not found (paid/organic)")
    ratio = (pd.to_numeric(cmo[paid], errors="coerce").replace(0, pd.NA) /
             pd.to_numeric(cmo[organic], errors="coerce").replace(0, pd.NA))
    if ratio.isna().all(): return _na("Insufficient numeric data")
    high = (ratio > 5).sum()
    volatile = (ratio.pct_change().abs() > 0.5).sum()
    if high >= 2 or volatile >= 3: return ("warn", _score("warn"), "Paid vs organic looks imbalanced/volatile")
    return ("pass", _score("pass"), "Paid vs organic balance reasonable")

def _eval_116(dfs):
    cmo, coo = _df_or_none(dfs,"cmo"), _df_or_none(dfs,"coo")
    if cmo is None or coo is None: return _na("Requires CMO, COO")
    leads = _find_one(cmo, ["leads","total_leads","unique_leads"])
    sql   = _find_one(cmo, ["sql","qualified_leads","sales_qualified_leads"])
    orders = _find_one(coo, ["orders","total_orders","completed_orders","shipments"])
    if not (leads and sql and orders): return _na("Columns not found (leads/sql/orders)")
    L = pd.to_numeric(cmo[leads], errors="coerce").fillna(0)
    S = pd.to_numeric(cmo[sql], errors="coerce").fillna(0)
    O = pd.to_numeric(coo[orders], errors="coerce").fillna(0)
    viol = ((L + 1e-6) < (S - 1e-6)) | ((S + 1e-6) < (O - 1e-6))
    v = int(viol.sum()); n = len(viol)
    if v >= max(2, math.ceil(0.25*n)): return ("fail", _score("fail"), f"Funnel inconsistency in {v}/{n} periods")
    if v > 0: return ("warn", _score("warn"), f"Minor funnel inconsistency in {v}/{n} periods")
    return ("pass", _score("pass"), "Lead→SQL→Order consistent")

def _eval_117(dfs):
    cfo = _df_or_none(dfs,"cfo")
    if cfo is None: return _na("Requires CFO")
    forecast = _find_one(cfo, ["revenue_forecast","forecast_revenue","proj_revenue"])
    actual   = _find_one(cfo, ["revenue","sales","booked_revenue","turnover"])
    if not (forecast and actual): return _na("Columns not found (forecast/actual revenue)")
    F = pd.to_numeric(cfo[forecast], errors="coerce").fillna(0)
    A = pd.to_numeric(cfo[actual], errors="coerce").fillna(0)
    err = (F - A).abs()
    tol = (A.abs()*0.15).clip(lower=1.0)  # 15% tolerance
    bad = (err > tol).sum()
    if bad >= max(2, math.ceil(0.3*len(A))): return ("fail", _score("fail"), f"{bad} periods outside tolerance")
    if bad > 0: return ("warn", _score("warn"), f"{bad} periods borderline forecast error")
    return ("pass", _score("pass"), "Forecast vs actual within tolerance")

def _eval_118(dfs):
    # Price proxy = revenue / orders (if both exist). Expect *negative* correlation with orders.
    cfo, coo = _df_or_none(dfs,"cfo"), _df_or_none(dfs,"coo")
    if cfo is None or coo is None: return _na("Requires CFO, COO")
    revenue = _find_one(cfo, ["revenue","sales","booked_revenue","turnover"])
    orders = _find_one(coo, ["orders","total_orders","completed_orders","shipments"])
    if not (revenue and orders): return _na("Columns not found (revenue/orders)")
    Rev = pd.to_numeric(cfo[revenue], errors="coerce").fillna(0)
    Ord = pd.to_numeric(coo[orders], errors="coerce").replace(0, pd.NA).fillna(method="ffill")
    price = (Rev / Ord).replace([pd.NA, pd.NaT, float("inf")], pd.NA)
    # If correlation between price proxy and orders is strongly positive, elasticity looks odd
    if price.isna().all() or Ord.isna().all(): return _na("Insufficient data to estimate elasticity")
    corr = price.corr(Ord)
    if pd.isna(corr): return _na("Insufficient overlap to compute correlation")
    if corr > 0.4: return ("warn", _score("warn"), f"Positive price–volume correlation (corr≈{corr:.2f})")
    return ("pass", _score("pass"), f"Elasticity plausible (corr≈{corr:.2f})")

def _eval_119(dfs):
    coo = _df_or_none(dfs,"coo")
    if coo is None: return _na("Requires COO")
    backlog = _find_one(coo, ["backlog","order_backlog","pending_orders","open_orders"])
    complaints = _find_one(coo, ["complaints","customer_complaints","tickets"])
    if not (backlog and complaints): return _na("Columns not found (backlog/complaints)")
    B = pd.to_numeric(coo[backlog], errors="coerce").fillna(0)
    C = pd.to_numeric(coo[complaints], errors="coerce").fillna(0)
    corr = B.corr(C)
    if pd.isna(corr): return _na("Insufficient overlap to compute correlation")
    if corr < 0.0: return ("warn", _score("warn"), f"Backlog↑ with complaints↓ (corr≈{corr:.2f}) – data check?")
    return ("pass", _score("pass"), f"Complaints move with backlog (corr≈{corr:.2f})")

def _eval_120(dfs):
    coo = _df_or_none(dfs,"coo")
    if coo is None: return _na("Requires COO")
    maint = _find_one(coo, ["maintenance_spend","maintenance_cost","maint_spend"])
    defects = _find_one(coo, ["defect_rate","defects","quality_defect_rate"])
    if not (maint and defects): return _na("Columns not found (maintenance_spend/defects)")
    m_up,_ = _growth_bool(coo[maint], up_thresh=0.05)
    d_up,_ = _growth_bool(coo[defects], up_thresh=0.01)
    bad = (m_up & d_up).sum()
    if bad >= 2: return ("warn", _score("warn"), f"{bad} periods: maintenance spend up but defects up")
    return ("pass", _score("pass"), "Maintenance spend aligns with defect trend")

def _eval_121(dfs):
    chro, cfo, cpo = _df_or_none(dfs,"chro"), _df_or_none(dfs,"cfo"), _df_or_none(dfs,"cpo")
    if chro is None or cfo is None or cpo is None: return _na("Requires CHRO, CFO, CPO")
    head = _find_one(chro, ["headcount","total_headcount","employees"])
    hires = _find_one(cpo, ["new_hires","hiring","offers_accepted","joined"])
    revenue = _find_one(cfo, ["revenue","sales","booked_revenue","turnover"])
    if not (head and hires and revenue): return _na("Columns not found (headcount/hires/revenue)")
    H = pd.to_numeric(chro[head], errors="coerce").replace(0, pd.NA).fillna(method="ffill")
    Rev = pd.to_numeric(cfo[revenue], errors="coerce").fillna(0)
    RPE = Rev / H.replace(0, pd.NA)
    rpe_down = (_pct_change(RPE) < -0.15).sum()
    h_up,_ = _growth_bool(pd.to_numeric(cpo[hires], errors="coerce"), up_thresh=0.1)
    if rpe_down >= 2 and h_up >= 2: return ("warn", _score("warn"), "Revenue/employee falling while hiring velocity increases")
    return ("pass", _score("pass"), "Revenue/employee vs hiring velocity acceptable")

def _eval_122(dfs):
    cmo, cfo = _df_or_none(dfs,"cmo"), _df_or_none(dfs,"cfo")
    if cmo is None or cfo is None: return _na("Requires CMO, CFO")
    ltv = _find_one(cmo, ["ltv","customer_ltv","avg_ltv"])
    cac = _find_one(cmo, ["cac","customer_acquisition_cost"])
    if not (ltv and cac): return _na("Columns not found (ltv/cac)")
    L = pd.to_numeric(cmo[ltv], errors="coerce").replace([0, pd.NA], pd.NA)
    C = pd.to_numeric(cmo[cac], errors="coerce").replace([0, pd.NA], pd.NA)
    ratio = (L / C).dropna()
    if ratio.empty: return _na("Insufficient LTV or CAC data")
    low = (ratio < 1).sum()
    extreme = (ratio > 10).sum()
    if low >= 2: return ("fail", _score("fail"), f"{low} periods LTV:CAC < 1")
    if extreme >= 2: return ("warn", _score("warn"), f"{extreme} periods LTV:CAC unusually high")
    return ("pass", _score("pass"), "LTV:CAC within reasonable bounds")

def _eval_123(dfs):
    cmo = _df_or_none(dfs,"cmo")
    if cmo is None: return _na("Requires CMO")
    spend = _find_one(cmo, ["marketing_spend","ad_spend","total_spend","spend"])
    leads = _find_one(cmo, ["leads","total_leads","unique_leads"])
    conv  = _find_one(cmo, ["lead_to_sql_rate","lead_quality_score","conversion_rate"])
    if not (spend and leads and conv): return _na("Columns not found (spend/leads/conversion)")
    s_up,_ = _growth_bool(cmo[spend])
    _, q_down = _growth_bool(cmo[conv], down_thresh=-0.05)
    bad = (s_up & q_down).sum()
    if bad >= 2: return ("warn", _score("warn"), f"{bad} periods: spend↑ while lead quality/conv↓")
    return ("pass", _score("pass"), "Spend vs lead quality stable")

def _eval_124(dfs):
    coo = _df_or_none(dfs,"coo")
    if coo is None: return _na("Requires COO")
    overtime = _find_one(coo, ["overtime_hours","overtime_cost","ot_hours"])
    sla_breach = _find_one(coo, ["sla_breaches","breaches","sla_missed"])
    if not (overtime and sla_breach): return _na("Columns not found (overtime/sla_breaches)")
    ot_up,_ = _growth_bool(coo[overtime], up_thresh=0.05)
    _, breaches_down = _growth_bool(coo[sla_breach], down_thresh=-0.05)
    # If overtime rises but breaches don't fall, it’s a miss
    miss = (ot_up & (~breaches_down)).sum()
    if miss >= 2: return ("warn", _score("warn"), f"{miss} periods: overtime↑ without SLA improvement")
    return ("pass", _score("pass"), "Overtime seems to help SLA outcomes")

def _eval_125(dfs):
    cfo, cmo, cpo = _df_or_none(dfs,"cfo"), _df_or_none(dfs,"cmo"), _df_or_none(dfs,"cpo")
    if cfo is None or cmo is None or cpo is None: return _na("Requires CFO, CMO, CPO")
    runway = _find_one(cfo, ["runway_months","months_of_runway","cash_runway_months"])
    spend  = _find_one(cmo, ["marketing_spend","ad_spend","total_spend","spend"])
    hires  = _find_one(cpo, ["new_hires","hiring","offers_accepted","joined"])
    if not (runway and spend and hires): return _na("Columns not found (runway/spend/hiring)")
    R = pd.to_numeric(cfo[runway], errors="coerce").fillna(0)
    S = pd.to_numeric(cmo[spend], errors="coerce").fillna(0)
    H = pd.to_numeric(cpo[hires], errors="coerce").fillna(0)
    low_runway = (R < 6).sum()
    s_up,_ = _growth_bool(S)
    h_up,_ = _growth_bool(H)
    if low_runway >= 2 and s_up.sum() >= 2 and h_up.sum() >= 2:
        return ("fail", _score("fail"), "Low runway while spend & hiring rise across multiple periods")
    if (R < 6).sum() >= 1 and (s_up.sum() >= 1 or h_up.sum() >= 1):
        return ("warn", _score("warn"), "Runway tight with rising spend/hiring")
    return ("pass", _score("pass"), "Runway vs spend & hiring looks reasonable")

# Map rule_id -> evaluator
EVALS: Dict[str, Callable[[Dict[str, pd.DataFrame]], Tuple[str, float, str]]] = {
    "CROSS-R-101": _eval_101,
    "CROSS-R-102": _eval_102,
    "CROSS-R-103": _eval_103,
    "CROSS-R-104": _eval_104,
    "CROSS-R-105": _eval_105,
    "CROSS-R-106": _eval_106,
    "CROSS-R-107": _eval_107,
    "CROSS-R-108": _eval_108,
    "CROSS-R-109": _eval_109,
    "CROSS-R-110": _eval_110,
    "CROSS-R-111": _eval_111,
    "CROSS-R-112": _eval_112,
    "CROSS-R-113": _eval_113,
    "CROSS-R-114": _eval_114,
    "CROSS-R-115": _eval_115,
    "CROSS-R-116": _eval_116,
    "CROSS-R-117": _eval_117,
    "CROSS-R-118": _eval_118,
    "CROSS-R-119": _eval_119,
    "CROSS-R-120": _eval_120,
    "CROSS-R-121": _eval_121,
    "CROSS-R-122": _eval_122,
    "CROSS-R-123": _eval_123,
    "CROSS-R-124": _eval_124,
    "CROSS-R-125": _eval_125,
}

# ---------------------------------------------
# Public API
# ---------------------------------------------
def evaluate_cross_rules(brain_inputs: Dict[str, pd.DataFrame], rules_dir: str) -> Dict[str, Any]:
    rules_dir = os.path.normpath(rules_dir)
    rules = _load_rules(rules_dir)
    findings: List[Dict[str, Any]] = []
    for r in rules:
        try:
            fn = EVALS.get(r.rule_id)
            if fn is None:
                status, score, detail = _na("No native evaluator for this rule_id")
            else:
                status, score, detail = fn(brain_inputs)
            findings.append({
                "rule_id": r.rule_id,
                "title": r.title,
                "severity": r.severity,
                "status": status,
                "score": score,
                "_file": r.file_path,
                "detail": detail,
            })
        except Exception as e:
            findings.append({
                "rule_id": r.rule_id,
                "title": r.title,
                "severity": r.severity,
                "status": "error",
                "score": 0.0,
                "_file": r.file_path,
                "detail": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=2),
            })
    meta = {
        "engine": "native",
        "rules_path": rules_dir,
        "rules_count": len(rules),
        "status": "ok",
        "error": None,
    }
    return {"meta": meta, "findings": findings}
