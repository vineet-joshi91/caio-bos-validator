# -*- coding: utf-8 -*-
"""
Reality Interface v0: YAML-driven external reality signals + feasibility flags.
No internet. No prediction. Pure curated constraints + checks.
"""

from __future__ import annotations
import os, glob, traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import yaml

# -----------------------------
# Data model
# -----------------------------
@dataclass
class RealitySignal:
    signal_id: str
    domain: str
    title: str
    severity: str
    confidence: str
    horizon: str
    valid_until: str
    file_path: str
    statement: str
    tags: List[str]

def _norm_domain(d: str) -> str:
    d = (d or "").strip().lower()
    # normalize common aliases
    alias = {
        "ops": "operations",
        "people": "hr",
        "workforce": "talent"
    }
    return alias.get(d, d)

def _load_signals(reality_root: str) -> List[RealitySignal]:
    # load all yaml files under rules/reality/**.yaml
    patterns = [
        os.path.join(reality_root, "**", "*.yaml"),
        os.path.join(reality_root, "**", "*.yml"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))

    signals: List[RealitySignal] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}

        sid = str(y.get("id") or y.get("signal_id") or os.path.basename(fp))
        domain = _norm_domain(str(y.get("domain") or "unknown"))
        title = str(y.get("title") or "")
        statement = str(y.get("statement") or "")
        severity = str(y.get("severity") or "medium").lower()
        confidence = str(y.get("confidence") or "medium").lower()
        horizon = str(y.get("horizon") or "6_12_months")
        valid_until = str(y.get("valid_until") or "")
        tags = y.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]

        signals.append(
            RealitySignal(
                signal_id=sid,
                domain=domain,
                title=title,
                statement=statement,
                severity=severity,
                confidence=confidence,
                horizon=horizon,
                valid_until=valid_until,
                tags=[str(t) for t in tags],
                file_path=os.path.normpath(fp),
            )
        )

    return signals

def _severity_score(sev: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get((sev or "").lower(), 2)

def _compute_feasibility_from_findings(
    brain_payload: Dict[str, Any],
    signals: List[RealitySignal]
) -> Dict[str, Any]:
    """
    v0 feasibility logic:
    - We do NOT predict.
    - We mark feasibility risk using:
      (a) internal BOS risk signals (warn/fail counts if available)
      (b) existence of high/critical reality signals in that domain
    """

    # --- Map domains to brains (your BOS brains)
    domain_to_brain = {
        "finance": "cfo",
        "marketing": "cmo",
        "operations": "coo",
        "hr": "chro",
        "talent": "cpo",  # talent/hiring/people ops often sits in CPO rules in your stack
    }

    # counts from internal findings if present
    def _count_statuses(brain_result: Dict[str, Any]) -> Tuple[int, int]:
        # Your run_brain_validation returns different shapes across versions.
        # We'll defensively scan for "findings".
        findings = brain_result.get("findings") or brain_result.get("results") or []
        warn = 0
        fail = 0
        if isinstance(findings, list):
            for it in findings:
                st = str(it.get("status") or "").lower()
                if st == "warn": warn += 1
                if st == "fail": fail += 1
        return warn, fail

    # group signals per domain
    sig_by_domain: Dict[str, List[RealitySignal]] = {}
    for s in signals:
        sig_by_domain.setdefault(s.domain, []).append(s)

    by_domain: Dict[str, Any] = {}
    by_brain: Dict[str, Any] = {}

    brains = brain_payload.get("brains") or {}
    for domain, brain in domain_to_brain.items():
        brain_result = brains.get(brain) or {}
        warn_ct, fail_ct = _count_statuses(brain_result)

        dom_sigs = sig_by_domain.get(domain, [])
        max_sev = max([_severity_score(s.severity) for s in dom_sigs], default=0)

        # heuristic status
        # - if internal fails exist OR critical signals exist -> infeasible risk
        # - if internal warns exist OR high signals exist -> feasibility risk
        if fail_ct >= 1 or max_sev >= 4:
            status = "risk_high"
        elif warn_ct >= 2 or max_sev >= 3:
            status = "risk_medium"
        elif warn_ct == 1 or max_sev == 2:
            status = "risk_low"
        else:
            status = "ok"

        by_domain[domain] = {
            "status": status,
            "internal": {"warn": warn_ct, "fail": fail_ct},
            "reality": {
                "signals_total": len(dom_sigs),
                "max_severity": max([s.severity for s in dom_sigs], default=None),
                "top_signals": [
                    {"id": s.signal_id, "title": s.title, "severity": s.severity, "_file": s.file_path}
                    for s in sorted(dom_sigs, key=lambda x: _severity_score(x.severity), reverse=True)[:3]
                ],
            },
            "message": (
                "No major feasibility flags."
                if status == "ok" else
                "Feasibility risk flagged based on internal findings and/or reality constraints."
            ),
        }

        by_brain[brain] = by_domain[domain]

    return {"by_domain": by_domain, "by_brain": by_brain}

def evaluate_reality(reality_dir: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public API similar to cross_rules_engine.evaluate_cross_rules.
    Returns meta + signals + feasibility.
    """
    reality_dir = os.path.normpath(reality_dir)

    try:
        signals = _load_signals(reality_dir)
        feasibility = _compute_feasibility_from_findings(payload, signals)

        return {
            "meta": {
                "engine": "yaml_reality_v0",
                "rules_path": reality_dir,
                "signals_count": len(signals),
                "status": "ok",
                "error": None,
            },
            "signals": [
                {
                    "id": s.signal_id,
                    "domain": s.domain,
                    "title": s.title,
                    "statement": s.statement,
                    "severity": s.severity,
                    "confidence": s.confidence,
                    "horizon": s.horizon,
                    "valid_until": s.valid_until,
                    "tags": s.tags,
                    "_file": s.file_path,
                } for s in signals
            ],
            "feasibility": feasibility,
        }

    except Exception as e:
        return {
            "meta": {
                "engine": "yaml_reality_v0",
                "rules_path": reality_dir,
                "signals_count": 0,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            },
            "signals": [],
            "feasibility": {"by_domain": {}, "by_brain": {}},
            "trace": traceback.format_exc(limit=2),
        }
