# -*- coding: utf-8 -*-
"""
CPO tools module (Chief People Officer).

Hybrid / auto-detect CPO layer that:
- detects external talent context
- computes core external people metrics
- computes context-specific metrics
- returns {"context": ..., "metrics": {...}, "needs": [...]}
"""

from typing import Dict, Any, List, Optional
from .common import safe_div, add_need, clip


# ----------------------------------------------------------
# 1. AUTO-CONTEXT DETECTION
# ----------------------------------------------------------
def _detect_context(meta: Dict[str, Any]) -> str:
    """
    Detect CPO context based on available fields.

    Possible contexts:
      - agency_hiring
      - contract_workforce
      - bpo_outsourcing
      - talent_bench
      - generic
    """
    explicit = meta.get("cpo_context")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()

    keys = {str(k).lower() for k in meta.keys()}

    agency_signals = [
        "agency_open_roles",
        "agency_submissions",
        "agency_shortlists",
        "agency_offers_made",
        "agency_joins",
        "agency_time_to_fill_days",
    ]
    if any(k in keys for k in agency_signals):
        return "agency_hiring"

    contract_signals = [
        "contractors_count",
        "avg_contract_tenure_months",
        "conversion_to_fte_count",
        "contractor_billable_utilisation_pct",
    ]
    if any(k in keys for k in contract_signals):
        return "contract_workforce"

    bpo_signals = [
        "bpo_fte_equivalent",
        "bpo_sla_compliance_pct",
        "bpo_volume_handled_pct",
        "bpo_cost_per_unit",
    ]
    if any(k in keys for k in bpo_signals):
        return "bpo_outsourcing"

    bench_signals = [
        "bench_headcount",
        "bench_cost_per_month",
        "average_bench_days",
        "bench_to_billable_conversion_pct",
    ]
    if any(k in keys for k in bench_signals):
        return "talent_bench"

    return "generic"


# ----------------------------------------------------------
# 2. CORE METRICS (EXTERNAL TALENT)
# ----------------------------------------------------------
def _core_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Core external people metrics that matter in almost every CPO context.
    """
    m: Dict[str, Any] = {}

    external_headcount_total = (
        meta.get("external_headcount_total")
        or meta.get("contractor_count")
        or meta.get("outsourced_headcount")
    )
    external_billable_headcount = meta.get("external_billable_headcount")
    external_bench_headcount = meta.get("external_bench_headcount")

    talent_vendor_count = (
        meta.get("talent_vendor_count")
        or meta.get("staffing_partner_count")
        or meta.get("agency_count")
    )

    external_talent_cost_total = (
        meta.get("external_talent_cost_total")
        or meta.get("contractor_cost_total")
        or meta.get("outsourcing_cost_total")
    )

    external_attrition_rate_pct = meta.get("external_attrition_rate_pct")

    external_billable_utilisation_pct = meta.get("external_billable_utilisation_pct")
    bench_cost_per_month = meta.get("bench_cost_per_month")

    if external_headcount_total is None:
        add_need(needs, "external_headcount_total / contractor_count / outsourced_headcount")
    if external_talent_cost_total is None:
        add_need(needs, "external_talent_cost_total / contractor_cost_total / outsourcing_cost_total")
    if external_billable_utilisation_pct is None:
        add_need(needs, "external_billable_utilisation_pct for utilisation tracking")

    # Cost per FTE equivalent (if we can approximate)
    external_cost_per_fte_equivalent: Optional[float] = None
    if external_talent_cost_total is not None and external_headcount_total not in (None, 0):
        external_cost_per_fte_equivalent = safe_div(
            external_talent_cost_total, external_headcount_total
        )

    m.update(
        {
            "external_headcount_total": external_headcount_total,
            "external_billable_headcount": external_billable_headcount,
            "external_bench_headcount": external_bench_headcount,
            "talent_vendor_count": talent_vendor_count,
            "external_talent_cost_total": external_talent_cost_total,
            "external_cost_per_fte_equivalent": None
            if external_cost_per_fte_equivalent is None
            else round(external_cost_per_fte_equivalent, 2),
            "external_attrition_rate_pct": None
            if external_attrition_rate_pct is None
            else round(clip(float(external_attrition_rate_pct), 0.0, 100.0), 2),
            "external_billable_utilisation_pct": None
            if external_billable_utilisation_pct is None
            else round(
                clip(float(external_billable_utilisation_pct), 0.0, 150.0), 2
            ),
            "bench_cost_per_month": bench_cost_per_month,
        }
    )

    return m


# ----------------------------------------------------------
# 3. CONTEXT-SPECIFIC METRICS
# ----------------------------------------------------------
def _agency_hiring_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Metrics for agency-based hiring (external vendors filling roles).
    """
    m: Dict[str, Any] = {}

    agency_open_roles = meta.get("agency_open_roles")
    agency_submissions = meta.get("agency_submissions")
    agency_shortlists = meta.get("agency_shortlists")
    agency_interviews = meta.get("agency_interviews")
    agency_offers_made = meta.get("agency_offers_made")
    agency_joins = meta.get("agency_joins")

    agency_time_to_fill_days = meta.get("agency_time_to_fill_days")
    agency_cost_per_hire = meta.get("agency_cost_per_hire")
    agency_fees_total = meta.get("agency_fees_total")

    if agency_open_roles is None:
        add_need(needs, "agency_open_roles for external hiring load")
    if agency_submissions is None:
        add_need(needs, "agency_submissions for vendor funnel")
    if agency_joins is None:
        add_need(needs, "agency_joins for placement success")
    if agency_time_to_fill_days is None:
        add_need(needs, "agency_time_to_fill_days for speed metrics")

    # Funnel conversion rates
    submissions_to_shortlist_pct: Optional[float] = None
    if agency_submissions not in (None, 0) and agency_shortlists is not None:
        r = safe_div(agency_shortlists, agency_submissions)
        if r is not None:
            submissions_to_shortlist_pct = clip(r * 100.0, 0.0, 100.0)

    shortlist_to_interview_pct: Optional[float] = None
    if agency_shortlists not in (None, 0) and agency_interviews is not None:
        r = safe_div(agency_interviews, agency_shortlists)
        if r is not None:
            shortlist_to_interview_pct = clip(r * 100.0, 0.0, 100.0)

    interview_to_offer_pct: Optional[float] = None
    if agency_interviews not in (None, 0) and agency_offers_made is not None:
        r = safe_div(agency_offers_made, agency_interviews)
        if r is not None:
            interview_to_offer_pct = clip(r * 100.0, 0.0, 100.0)

    offer_to_join_pct: Optional[float] = None
    if agency_offers_made not in (None, 0) and agency_joins is not None:
        r = safe_div(agency_joins, agency_offers_made)
        if r is not None:
            offer_to_join_pct = clip(r * 100.0, 0.0, 100.0)

    agency_success_rate_pct: Optional[float] = None
    if agency_open_roles not in (None, 0) and agency_joins is not None:
        r = safe_div(agency_joins, agency_open_roles)
        if r is not None:
            agency_success_rate_pct = clip(r * 100.0, 0.0, 200.0)

    m.update(
        {
            "agency_open_roles": agency_open_roles,
            "agency_submissions": agency_submissions,
            "agency_shortlists": agency_shortlists,
            "agency_interviews": agency_interviews,
            "agency_offers_made": agency_offers_made,
            "agency_joins": agency_joins,
            "agency_time_to_fill_days": agency_time_to_fill_days,
            "agency_cost_per_hire": agency_cost_per_hire,
            "agency_fees_total": agency_fees_total,
            "submissions_to_shortlist_pct": None
            if submissions_to_shortlist_pct is None
            else round(submissions_to_shortlist_pct, 2),
            "shortlist_to_interview_pct": None
            if shortlist_to_interview_pct is None
            else round(shortlist_to_interview_pct, 2),
            "interview_to_offer_pct": None
            if interview_to_offer_pct is None
            else round(interview_to_offer_pct, 2),
            "offer_to_join_pct": None
            if offer_to_join_pct is None
            else round(offer_to_join_pct, 2),
            "agency_success_rate_pct": None
            if agency_success_rate_pct is None
            else round(agency_success_rate_pct, 2),
        }
    )

    return m


def _contract_workforce_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Metrics for contractors / freelancers / consultants.
    """
    m: Dict[str, Any] = {}

    contractors_count = meta.get("contractors_count")
    avg_contract_tenure_months = meta.get("avg_contract_tenure_months")
    extension_rate_pct = meta.get("extension_rate_pct")
    conversion_to_fte_count = meta.get("conversion_to_fte_count")
    conversion_to_fte_rate_pct = meta.get("conversion_to_fte_rate_pct")
    contractor_billable_utilisation_pct = meta.get(
        "contractor_billable_utilisation_pct"
    )

    if contractors_count is None:
        add_need(needs, "contractors_count for external workforce size")
    if avg_contract_tenure_months is None:
        add_need(needs, "avg_contract_tenure_months for contract dynamics")

    m.update(
        {
            "contractors_count": contractors_count,
            "avg_contract_tenure_months": avg_contract_tenure_months,
            "extension_rate_pct": None
            if extension_rate_pct is None
            else round(clip(float(extension_rate_pct), 0.0, 100.0), 2),
            "conversion_to_fte_count": conversion_to_fte_count,
            "conversion_to_fte_rate_pct": None
            if conversion_to_fte_rate_pct is None
            else round(clip(float(conversion_to_fte_rate_pct), 0.0, 100.0), 2),
            "contractor_billable_utilisation_pct": None
            if contractor_billable_utilisation_pct is None
            else round(
                clip(float(contractor_billable_utilisation_pct), 0.0, 150.0), 2
            ),
        }
    )

    return m


def _bpo_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Metrics for BPO / outsourced teams doing entire processes.
    """
    m: Dict[str, Any] = {}

    bpo_fte_equivalent = meta.get("bpo_fte_equivalent")
    bpo_sla_compliance_pct = meta.get("bpo_sla_compliance_pct")
    bpo_volume_handled_pct = meta.get("bpo_volume_handled_pct")
    bpo_cost_per_unit = meta.get("bpo_cost_per_unit")
    bpo_vs_internal_cost_ratio = meta.get("bpo_vs_internal_cost_ratio")

    if bpo_fte_equivalent is None:
        add_need(needs, "bpo_fte_equivalent for external capacity")
    if bpo_sla_compliance_pct is None:
        add_need(needs, "bpo_sla_compliance_pct for performance tracking")

    m.update(
        {
            "bpo_fte_equivalent": bpo_fte_equivalent,
            "bpo_sla_compliance_pct": None
            if bpo_sla_compliance_pct is None
            else round(clip(float(bpo_sla_compliance_pct), 0.0, 100.0), 2),
            "bpo_volume_handled_pct": None
            if bpo_volume_handled_pct is None
            else round(clip(float(bpo_volume_handled_pct), 0.0, 100.0), 2),
            "bpo_cost_per_unit": bpo_cost_per_unit,
            "bpo_vs_internal_cost_ratio": bpo_vs_internal_cost_ratio,
        }
    )

    return m


def _bench_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Metrics for external talent bench.
    """
    m: Dict[str, Any] = {}

    bench_headcount = meta.get("bench_headcount")
    bench_cost_per_month = meta.get("bench_cost_per_month")
    average_bench_days = meta.get("average_bench_days")
    bench_to_billable_conversion_pct = meta.get("bench_to_billable_conversion_pct")
    redeployment_rate_pct = meta.get("redeployment_rate_pct")

    if bench_headcount is None:
        add_need(needs, "bench_headcount for bench sizing")
    if bench_cost_per_month is None:
        add_need(needs, "bench_cost_per_month for idle cost impact")

    m.update(
        {
            "bench_headcount": bench_headcount,
            "bench_cost_per_month": bench_cost_per_month,
            "average_bench_days": average_bench_days,
            "bench_to_billable_conversion_pct": None
            if bench_to_billable_conversion_pct is None
            else round(
                clip(float(bench_to_billable_conversion_pct), 0.0, 100.0), 2
            ),
            "redeployment_rate_pct": None
            if redeployment_rate_pct is None
            else round(clip(float(redeployment_rate_pct), 0.0, 100.0), 2),
        }
    )

    return m


# ----------------------------------------------------------
# 4. MAIN ENTRYPOINT
# ----------------------------------------------------------
def run(pkt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for CPO tools (Chief People Officer).

    Returns:
        {
            "context": "<agency_hiring|contract_workforce|bpo_outsourcing|talent_bench|generic>",
            "metrics": {...},
            "needs": [...]
        }
    """
    needs: List[str] = []
    metrics: Dict[str, Any] = {}

    meta = pkt.get("meta", {}) or {}

    context = _detect_context(meta)

    # Core external people metrics
    metrics.update(_core_metrics(meta, needs))

    # Context-specific metrics
    if context == "agency_hiring":
        metrics.update(_agency_hiring_metrics(meta, needs))
    elif context == "contract_workforce":
        metrics.update(_contract_workforce_metrics(meta, needs))
    elif context == "bpo_outsourcing":
        metrics.update(_bpo_metrics(meta, needs))
    elif context == "talent_bench":
        metrics.update(_bench_metrics(meta, needs))
    else:
        # generic: only core metrics + needs
        pass

    return {
        "context": context,
        "metrics": metrics,
        "needs": needs,
    }
