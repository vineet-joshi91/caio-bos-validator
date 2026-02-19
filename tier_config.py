# tier_config.py
# -*- coding: utf-8 -*-
"""
Central configuration for CAIO usage tiers.

CAIO has TWO tiers:
1. STANDARD (free tier with credit top-ups)
   - Can see: Execution Plan only
   - Decision Review: 50 credits per use (expensive to encourage Premium)
   - Limited free credits, can buy more

2. PREMIUM (₹4,999/month paid subscription)
   - Unlimited EA and DR (no credit deductions)
   - Priority processing
"""

TIER_CONFIG = {
    # --------------------------------------------------------
    # STANDARD Tier (default free tier)
    # --------------------------------------------------------
    "standard": {
        "credits_per_ea": 10,          # Cost per EA run
        "credits_per_dr": 50,          # Cost per DR run (expensive by design)
        "daily_doc_cap": 10,           # 10 documents/day
        "can_see_decision_review": True,  # Can access but expensive
        "display_name": "Standard",
    },
    
    # --------------------------------------------------------
    # PREMIUM Tier (paid subscription - unlimited)
    # --------------------------------------------------------
    "premium": {
        "credits_per_ea": 0,           # Unlimited - no deduction
        "credits_per_dr": 0,           # Unlimited - no deduction
        "daily_doc_cap": None,         # Unlimited documents
        "can_see_decision_review": True,
        "display_name": "Premium",
    },
}

# Aliases for backward compatibility
TIER_CONFIG["free"] = TIER_CONFIG["standard"]
TIER_CONFIG["demo"] = TIER_CONFIG["standard"]
TIER_CONFIG["pro"] = TIER_CONFIG["premium"]

def get_tier_config(tier_name: str, is_admin: bool = False) -> dict:
    """
    Safely return tier configuration.
    Admins get premium features regardless of tier.
    """
    if is_admin:
        return TIER_CONFIG["premium"]
    
    if not tier_name:
        return TIER_CONFIG["standard"]
    
    tier_key = tier_name.lower().strip()
    return TIER_CONFIG.get(tier_key, TIER_CONFIG["standard"])

def can_access_decision_review(tier_name: str, is_admin: bool = False) -> bool:
    """
    Check if user's tier allows Decision Review feature.
    Standard users CAN access but pay 50 credits per use.
    """
    if is_admin:
        return True
    
    config = get_tier_config(tier_name)
    return config.get("can_see_decision_review", False)

def get_dr_cost(tier_name: str, is_admin: bool = False) -> int:
    """
    Get the credit cost for Decision Review.
    Returns 0 for premium/admin (unlimited).
    """
    if is_admin:
        return 0
    
    config = get_tier_config(tier_name, is_admin)
    return config.get("credits_per_dr", 50)