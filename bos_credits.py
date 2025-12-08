# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:28:53 2025

@author: Vineet
"""

# bos_credits.py
# -*- coding: utf-8 -*-
"""
Central gate for BOS usage.

Every BOS / EA analysis endpoint should call `charge_bos_run()`
BEFORE invoking the SLM engine.

Responsibilities:
- Read plan_tier
- Look up tier config (credits per analysis, daily cap)
- Check wallet balance
- Enforce Demo daily doc cap
- Deduct credits + increment usage_daily + log credit_transactions
"""

from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from wallet import (
    consume_credits_and_record_usage,
    InsufficientCreditsError,
    DailyLimitReachedError,
)
from tier_config import TIER_CONFIG


def charge_bos_run(
    db: Session,
    *,
    user_id: int,
    plan_tier: Optional[str],
    brain: str,
    doc_increment: int = 1,
) -> int:
    """
    Apply the BOS usage gate:

    - Resolve tier config
    - Enforce daily doc cap (for demo)
    - Deduct credits
    - Update usage_daily
    - Insert credit_transactions row

    Returns:
        credits_used (int) if successful.

    Raises:
        HTTPException 402 if insufficient credits
        HTTPException 429 if daily limit reached
    """
    tier_key = (plan_tier or "demo").lower().strip()
    cfg = TIER_CONFIG.get(tier_key, TIER_CONFIG["demo"])

    daily_cap = cfg.get("daily_doc_cap")        # int or None
    credits_required = cfg.get("credits_per_analysis", 10)

    try:
        consume_credits_and_record_usage(
            db=db,
            user_id=user_id,
            credits_required=credits_required,
            doc_increment=doc_increment,
            reason=f"{brain}_run",
            gateway="system",
            metadata={"endpoint": "bos", "brain": brain, "tier": tier_key},
            daily_doc_cap=daily_cap,
        )
        # DO NOT commit here – caller (endpoint) is responsible for db.commit()
    except InsufficientCreditsError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except DailyLimitReachedError as e:
        raise HTTPException(status_code=429, detail=str(e))

    return credits_required
