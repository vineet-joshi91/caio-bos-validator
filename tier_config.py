# tier_config.py
# -*- coding: utf-8 -*-
"""
Central configuration for CAIO usage tiers.
Used by bos_credits.py → charge_bos_run()

CAIO has TWO tiers:
1. STANDARD (free tier with credit refills)
   - Can see: Execution Plan only
   - Decision Review: hidden behind paywall
   - Limited free credits, can buy more

2. PREMIUM (paid subscription)
   - Can see: Both Execution Plan AND Decision Review
   - Lower per-analysis cost
   - Unlimited daily documents
"""

TIER_CONFIG = {
    # --------------------------------------------------------
    # STANDARD Tier (default free tier)
    # --------------------------------------------------------
    "standard": {
        "credits_per_analysis": 10,    # Cost per EA run
        "daily_doc_cap": 10,           # 10 documents/day
        "can_see_decision_review": False,  # Hidden feature
        "display_name": "Standard",
    },
    
    # --------------------------------------------------------
    # PREMIUM Tier (paid subscription)
    # --------------------------------------------------------
    "premium": {
        "credits_per_analysis": 5,     # Cheaper per-run (50% off)
        "daily_doc_cap": None,         # Unlimited documents
        "can_see_decision_review": True,   # Full access
        "display_name": "Premium",
    },
}

# Aliases for backward compatibility
TIER_CONFIG["free"] = TIER_CONFIG["standard"]  # Legacy name
TIER_CONFIG["demo"] = TIER_CONFIG["standard"]  # Legacy name
TIER_CONFIG["pro"] = TIER_CONFIG["premium"]    # Legacy name

def get_tier_config(tier_name: str, is_admin: bool = False) -> dict:
    """
    Safely return tier configuration.
    Admins get premium features regardless of tier.
    Falls back to STANDARD tier if unknown or None.
    """
    # Admins always get premium config
    if is_admin:
        return TIER_CONFIG["premium"]
    
    if not tier_name:
        return TIER_CONFIG["standard"]
    
    tier_key = tier_name.lower().strip()
    return TIER_CONFIG.get(tier_key, TIER_CONFIG["standard"])

def can_access_decision_review(tier_name: str, is_admin: bool = False) -> bool:
    """
    Check if user's tier allows Decision Review feature.
    Admins always have access.
    Returns: True if premium or admin, False otherwise
    """
    if is_admin:
        return True
    
    config = get_tier_config(tier_name)
    return config.get("can_see_decision_review", False)