# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:29:32 2025

@author: Vineet
"""

# orchestrator/cross_store.py
from collections import defaultdict
import pandas as pd

class Facts:
    def __init__(self):
        self.metrics = defaultdict(dict)  # metrics[brain][metric_name] = Series/DataFrame
        self.signals = defaultdict(dict)  # signals[brain][rule_id] = {"score":..., "status":...}

    def ingest_brain(self, brain_name, report_json):
        # Load all numeric detail metrics
        for rule in report_json.get("breakdown", []):
            details = rule.get("details", {})
            if "by_group" in details:
                # flatten period metrics
                self.metrics[brain_name][rule["id"]] = pd.Series({
                    k: v.get("value", 0) if isinstance(v, dict) else v
                    for k, v in details["by_group"].items()
                })
            # store top-level signal
            self.signals[brain_name][rule["id"]] = {
                "score": rule["score"],
                "status": rule["status"],
                "severity": rule["severity"]
            }

    def get_metric(self, brain, key):
        return self.metrics.get(brain, {}).get(key)

    def get_signal(self, brain, rule_id):
        return self.signals.get(brain, {}).get(rule_id)

    def to_frame(self):
        """Flatten everything into a dataframe for analysis or SLMs"""
        rows = []
        for b, metrics in self.metrics.items():
            for k, v in metrics.items():
                if isinstance(v, pd.Series):
                    for period, val in v.items():
                        rows.append({"brain": b, "metric": k, "period": period, "value": val})
        return pd.DataFrame(rows)

    def to_wide(self, period_col: str = "period") -> "pd.DataFrame":
        """
        Returns a wide DataFrame with columns like:
        cfo.booked_revenue, cmo.total_spend, coo.orders_completed, ...
        indexed by normalized 'period'.
        """
        import pandas as pd
        frames = []
        for brain, metrics in self.metrics.items():
            for name, series in metrics.items():
                if not hasattr(series, "items"):  # must be a Series of period->value
                    continue
                df = pd.DataFrame({period_col: list(series.index), f"{brain}.{name}": list(series.values)})
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=[period_col]).set_index(period_col)
        wide = frames[0]
        for df in frames[1:]:
            wide = wide.merge(df, on=period_col, how="outer")
        # normalize period key
        wide[period_col] = pd.to_datetime(wide[period_col], errors="coerce").dt.strftime("%Y-%m").fillna(wide[period_col])
        return wide.set_index(period_col).sort_index()
