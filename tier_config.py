# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 19:56:48 2025

@author: Vineet
"""

# tier_config.py
# -*- coding: utf-8 -*-
"""
Central configuration for CAIO usage tiers.

Used by bos_credits.py → charge_bos_run()

Field meanings:
- credits_per_analysis: how many credits to deduct per BOS/EA run.
- daily_doc_cap: max documents per day (None = unlimited).
"""

TIER_CONFIG = {
    # --------------------------------------------------------
    # Free Demo Tier (default if no plan_tier is provided)
    # --------------------------------------------------------
    "demo": {
        "credits_per_analysis": 10,    # EA/BOS analysis cost
        "daily_doc_cap": 3,            # only 3 documents/day allowed
    },

    # --------------------------------------------------------
    # Pro Tier (paid)
    # --------------------------------------------------------
    "pro": {
        "credits_per_analysis": 10,    # Same BOS cost, higher freedom
        "daily_doc_cap": None,         # No daily limit
    },

    # --------------------------------------------------------
    # Premium Tier (paid)
    # --------------------------------------------------------
    "premium": {
        "credits_per_analysis": 5,     # Cheaper per-run cost
        "daily_doc_cap": None,
    },

    # --------------------------------------------------------
    # Enterprise Tier (custom)
    # --------------------------------------------------------
    "enterprise": {
        "credits_per_analysis": 1,     # Essentially unlimited use
        "daily_doc_cap": None,
    },
}


def get_tier_config(tier_name: str) -> dict:
    """
    Safely return tier configuration.
    Falls back to DEMO tier if unknown or None.
    """
    if not tier_name:
        return TIER_CONFIG["demo"]

    tier_key = tier_name.lower().strip()
    return TIER_CONFIG.get(tier_key, TIER_CONFIG["demo"])
