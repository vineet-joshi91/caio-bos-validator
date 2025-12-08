# -*- coding: utf-8 -*-
"""
CHRO tools module.

Hybrid / auto-detect CHRO layer that:
- detects HR context (recruiting, performance, retention, hr_ops, generic)
- computes core HR metrics
- computes context-specific metrics
- returns {"context": ..., "metrics": {...}, "needs": [...]}
"""

from typing import Dict, Any, List, Optional
from .common import safe_div, add_need, clip


def _detect_context(meta: Dict[str, Any]) -> str:
    """
    Detect CHRO context based on available fields.

    Priority:
    - explicit meta["chro_context"]
    - recruiting signals
    - performance signals
    - retention/engagement signals
    - hr_ops signals
    - fallback: "generic"
    """
    explicit = meta.get("chro_context")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()

    keys = {str(k).lower() for k in meta.keys()}

    recruiting_signals = [
        "open_roles",
        "time_to_fill_days_avg",
        "time_to_hire_days_avg",
        "applications_received",
        "offers_made",
        "offers_accepted",
    ]
    if any(k in keys for k in recruiting_signals):
        return "recruiting"

    performance_signals = [
        "high_performer_pct",
        "low_performer_pct",
        "performance_review_completion_pct",
        "promotion_rate_pct",
    ]
    if any(k in keys for k in performance_signals):
        return "performance"

    retention_signals = [
        "engagement_score",
        "avg_tenure_years",
        "regrettable_attrition_rate_pct",
        "internal_mobility_rate_pct",
        "flight_risk_headcount",
    ]
    if any(k in keys for k in retention_signals):
        return "retention"

    hr_ops_signals = [
        "hr_tickets_open",
        "hr_tickets_closed",
        "hr_tat_avg_days",
        "payroll_error_rate_pct",
        "policy_non_compliance_incidents",
    ]
    if any(k in keys for k in hr_ops_signals):
        return "hr_ops"

    return "generic"


def _core_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Core CHRO metrics that matter in almost every context.
    """
    m: Dict[str, Any] = {}

    headcount = (
        meta.get("headcount_total")
        or meta.get("employee_count")
        or meta.get("headcount")
    )

    voluntary_attrition_pct = meta.get("voluntary_attrition_rate_pct")
    involuntary_attrition_pct = meta.get("involuntary_attrition_rate_pct")
    overall_attrition_pct = meta.get("overall_attrition_rate_pct")

    # If overall attrition not provided, try deriving from leavers/headcount
    if overall_attrition_pct is None:
        leavers_12m = meta.get("leavers_12m")
        if leavers_12m is not None and headcount not in (None, 0):
            r = safe_div(leavers_12m, headcount)
            if r is not None:
                overall_attrition_pct = r * 100.0

    avg_tenure_years = meta.get("avg_tenure_years")
    engagement_score = meta.get("engagement_score")  # 0–100, or 1–5 scaled later
    absenteeism_rate_pct = meta.get("absenteeism_rate_pct")

    # Needs
    if headcount is None:
        add_need(needs, "headcount_total / employee_count for HR scale metrics")
    if overall_attrition_pct is None:
        add_need(needs, "overall_attrition_rate_pct or leavers_12m for attrition")
    if engagement_score is None:
        add_need(needs, "engagement_score from surveys / inputs")
    if absenteeism_rate_pct is None:
        add_need(needs, "absenteeism_rate_pct for attendance stability")

    # Metrics (clipped where relevant)
    m.update(
        {
            "headcount_total": headcount,
            "overall_attrition_rate_pct": None
            if overall_attrition_pct is None
            else round(clip(float(overall_attrition_pct), 0.0, 100.0), 2),
            "voluntary_attrition_rate_pct": None
            if voluntary_attrition_pct is None
            else round(clip(float(voluntary_attrition_pct), 0.0, 100.0), 2),
            "involuntary_attrition_rate_pct": None
            if involuntary_attrition_pct is None
            else round(clip(float(involuntary_attrition_pct), 0.0, 100.0), 2),
            "avg_tenure_years": avg_tenure_years,
            "engagement_score": None
            if engagement_score is None
            else round(clip(float(engagement_score), 0.0, 100.0), 2),
            "absenteeism_rate_pct": None
            if absenteeism_rate_pct is None
            else round(clip(float(absenteeism_rate_pct), 0.0, 100.0), 2),
        }
    )

    return m


def _recruiting_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Recruiting / talent acquisition metrics.
    """
    m: Dict[str, Any] = {}

    open_roles = meta.get("open_roles")
    applications_received = meta.get("applications_received")
    candidates_screened = meta.get("candidates_screened")
    interviews_scheduled = meta.get("interviews_scheduled")
    offers_made = meta.get("offers_made")
    offers_accepted = meta.get("offers_accepted")

    time_to_fill_days_avg = meta.get("time_to_fill_days_avg")
    time_to_hire_days_avg = meta.get("time_to_hire_days_avg")

    if open_roles is None:
        add_need(needs, "open_roles for recruiting load")
    if time_to_fill_days_avg is None:
        add_need(needs, "time_to_fill_days_avg for requisition speed")
    if applications_received is None:
        add_need(needs, "applications_received for funnel analysis")
    if offers_made is None or offers_accepted is None:
        add_need(needs, "offers_made, offers_accepted for offer acceptance metrics")

    # Funnel conversion rates
    # Applications -> Screened
    app_to_screen_rate: Optional[float] = None
    if applications_received not in (None, 0) and candidates_screened is not None:
        r = safe_div(candidates_screened, applications_received)
        if r is not None:
            app_to_screen_rate = clip(r * 100.0, 0.0, 100.0)

    # Screened -> Interviews
    screen_to_interview_rate: Optional[float] = None
    if candidates_screened not in (None, 0) and interviews_scheduled is not None:
        r = safe_div(interviews_scheduled, candidates_screened)
        if r is not None:
            screen_to_interview_rate = clip(r * 100.0, 0.0, 100.0)

    # Interview -> Offer
    interview_to_offer_rate: Optional[float] = None
    if interviews_scheduled not in (None, 0) and offers_made is not None:
        r = safe_div(offers_made, interviews_scheduled)
        if r is not None:
            interview_to_offer_rate = clip(r * 100.0, 0.0, 100.0)

    # Offer -> Join
    offer_acceptance_rate_pct: Optional[float] = None
    if offers_made not in (None, 0) and offers_accepted is not None:
        r = safe_div(offers_accepted, offers_made)
        if r is not None:
            offer_acceptance_rate_pct = clip(r * 100.0, 0.0, 100.0)

    m.update(
        {
            "open_roles": open_roles,
            "applications_received": applications_received,
            "candidates_screened": candidates_screened,
            "interviews_scheduled": interviews_scheduled,
            "offers_made": offers_made,
            "offers_accepted": offers_accepted,
            "time_to_fill_days_avg": time_to_fill_days_avg,
            "time_to_hire_days_avg": time_to_hire_days_avg,
            "app_to_screen_rate_pct": None
            if app_to_screen_rate is None
            else round(app_to_screen_rate, 2),
            "screen_to_interview_rate_pct": None
            if screen_to_interview_rate is None
            else round(screen_to_interview_rate, 2),
            "interview_to_offer_rate_pct": None
            if interview_to_offer_rate is None
            else round(interview_to_offer_rate, 2),
            "offer_acceptance_rate_pct": None
            if offer_acceptance_rate_pct is None
            else round(offer_acceptance_rate_pct, 2),
        }
    )

    return m


def _performance_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Performance management metrics.
    """
    m: Dict[str, Any] = {}

    high_performer_pct = meta.get("high_performer_pct")
    low_performer_pct = meta.get("low_performer_pct")
    performance_review_completion_pct = meta.get("performance_review_completion_pct")
    promotion_rate_pct = meta.get("promotion_rate_pct")
    performance_issues_count = meta.get("performance_issues_count")

    if performance_review_completion_pct is None:
        add_need(
            needs,
            "performance_review_completion_pct for performance cycle health",
        )

    m.update(
        {
            "high_performer_pct": None
            if high_performer_pct is None
            else round(clip(float(high_performer_pct), 0.0, 100.0), 2),
            "low_performer_pct": None
            if low_performer_pct is None
            else round(clip(float(low_performer_pct), 0.0, 100.0), 2),
            "performance_review_completion_pct": None
            if performance_review_completion_pct is None
            else round(
                clip(float(performance_review_completion_pct), 0.0, 100.0), 2
            ),
            "promotion_rate_pct": None
            if promotion_rate_pct is None
            else round(clip(float(promotion_rate_pct), 0.0, 100.0), 2),
            "performance_issues_count": performance_issues_count,
        }
    )

    return m


def _retention_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    Retention & engagement oriented metrics.
    """
    m: Dict[str, Any] = {}

    regrettable_attrition_rate_pct = meta.get("regrettable_attrition_rate_pct")
    internal_mobility_rate_pct = meta.get("internal_mobility_rate_pct")
    flight_risk_headcount = meta.get("flight_risk_headcount")

    if regrettable_attrition_rate_pct is None:
        add_need(needs, "regrettable_attrition_rate_pct for key talent loss")
    if internal_mobility_rate_pct is None:
        add_need(needs, "internal_mobility_rate_pct for career pathing")
    if flight_risk_headcount is None:
        add_need(needs, "flight_risk_headcount for risk planning")

    m.update(
        {
            "regrettable_attrition_rate_pct": None
            if regrettable_attrition_rate_pct is None
            else round(
                clip(float(regrettable_attrition_rate_pct), 0.0, 100.0), 2
            ),
            "internal_mobility_rate_pct": None
            if internal_mobility_rate_pct is None
            else round(clip(float(internal_mobility_rate_pct), 0.0, 100.0), 2),
            "flight_risk_headcount": flight_risk_headcount,
        }
    )

    return m


def _hr_ops_metrics(meta: Dict[str, Any], needs: List[str]) -> Dict[str, Any]:
    """
    HR operations / HRSS metrics.
    """
    m: Dict[str, Any] = {}

    hr_tickets_open = meta.get("hr_tickets_open")
    hr_tickets_closed = meta.get("hr_tickets_closed")
    hr_tat_avg_days = meta.get("hr_tat_avg_days")
    payroll_error_rate_pct = meta.get("payroll_error_rate_pct")
    policy_non_compliance_incidents = meta.get("policy_non_compliance_incidents")

    if hr_tat_avg_days is None:
        add_need(needs, "hr_tat_avg_days for HR turnaround analysis")

    m.update(
        {
            "hr_tickets_open": hr_tickets_open,
            "hr_tickets_closed": hr_tickets_closed,
            "hr_tat_avg_days": hr_tat_avg_days,
            "payroll_error_rate_pct": None
            if payroll_error_rate_pct is None
            else round(clip(float(payroll_error_rate_pct), 0.0, 100.0), 2),
            "policy_non_compliance_incidents": policy_non_compliance_incidents,
        }
    )

    return m


def run(pkt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for CHRO tools.

    Returns:
        {
            "context": "<recruiting|performance|retention|hr_ops|generic>",
            "metrics": {...},
            "needs": [...]
        }
    """
    needs: List[str] = []
    metrics: Dict[str, Any] = {}

    meta = pkt.get("meta", {}) or {}

    context = _detect_context(meta)

    # Core metrics (common across contexts)
    metrics.update(_core_metrics(meta, needs))

    # Context-specific metrics
    if context == "recruiting":
        metrics.update(_recruiting_metrics(meta, needs))
    elif context == "performance":
        metrics.update(_performance_metrics(meta, needs))
    elif context == "retention":
        metrics.update(_retention_metrics(meta, needs))
    elif context == "hr_ops":
        metrics.update(_hr_ops_metrics(meta, needs))
    else:
        # generic: nothing extra, but we still return core metrics + needs
        pass

    return {
        "context": context,
        "metrics": metrics,
        "needs": needs,
    }
