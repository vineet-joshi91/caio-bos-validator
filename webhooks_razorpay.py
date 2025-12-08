# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 20:06:34 2025

@author: Vineet
"""

# webhooks_razorpay.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import razorpay
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from db import get_db
from wallet import CreditPack, PaymentRecord, apply_credit_topup

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

if not RAZORPAY_WEBHOOK_SECRET:
    # We won't crash the app, but will reject incoming webhooks.
    # This is intentional: better to fail safe than grant credits blindly.
    pass


def _verify_signature(body: bytes, signature: str) -> None:
    """
    Verify Razorpay webhook signature.
    Raises HTTPException(400) if invalid or secret missing.
    """
    if not RAZORPAY_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Razorpay webhook secret not configured on server.",
        )

    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature header")

    try:
        razorpay.Utility.verify_webhook_signature(
            body.decode("utf-8"),
            signature,
            RAZORPAY_WEBHOOK_SECRET,
        )
    except razorpay.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")


def _find_payment_record(
    db: Session,
    order_id: str,
) -> Optional[PaymentRecord]:
    return (
        db.query(PaymentRecord)
        .filter(
            PaymentRecord.gateway == "razorpay",
            PaymentRecord.gateway_order_id == order_id,
        )
        .first()
    )


@router.post("/razorpay")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(default=None, alias="X-Razorpay-Signature"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Handle Razorpay credit-pack payment events.

    Expected flow:
    - Frontend starts purchase via /wallet/create-credit-order (creates order + PaymentRecord).
    - Razorpay sends webhook when payment succeeds/fails.
    - We:
        - Verify signature.
        - Look up PaymentRecord by order_id.
        - If paid: apply_credit_topup() and mark status='completed'.
        - If failed: mark status='failed'.
    """
    body = await request.body()

    # 1) Verify signature
    _verify_signature(body, x_razorpay_signature)

    # 2) Parse payload
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in webhook")

    event = payload.get("event") or ""
    event = event.lower().strip()

    # We care mostly about successful payments
    if event not in {"payment.captured", "order.paid", "payment.failed", "order.payment_failed"}:
        # For now, just ignore other events
        return {"status": "ignored", "event": event}

    # Extract order_id, payment_id, amount, currency
    order_id = None
    payment_id = None
    amount = None
    currency = None

    if event.startswith("payment."):
        payment = (payload.get("payload") or {}).get("payment", {}).get("entity") or {}
        order_id = payment.get("order_id")
        payment_id = payment.get("id")
        amount = payment.get("amount")
        currency = payment.get("currency")
    elif event.startswith("order."):
        order = (payload.get("payload") or {}).get("order", {}).get("entity") or {}
        order_id = order.get("id")
        amount = order.get("amount")
        currency = order.get("currency")

    if not order_id:
        # Without order_id we can't map to PaymentRecord; just acknowledge.
        return {"status": "no_order_id", "event": event}

    # 3) Find PaymentRecord
    record = _find_payment_record(db, order_id)
    if not record:
        # Could be some unrelated order; don't crash webhook.
        return {"status": "no_matching_record", "event": event, "order_id": order_id}

    # Idempotency: if already completed, do nothing
    if record.status == "completed":
        return {"status": "already_completed", "event": event, "order_id": order_id}

    # 4) Handle failure events
    if event in {"payment.failed", "order.payment_failed"}:
        record.status = "failed"
        # Optionally store failure detail in metadata
        extra = record.metadata or {}
        extra["failure_event"] = event
        extra["raw_payload"] = payload
        record.metadata = extra
        db.commit()
        return {"status": "marked_failed", "event": event, "order_id": order_id}

    # 5) At this point, it's a success event (payment.captured or order.paid)
    # Sanity check amount/currency
    if amount is not None and currency:
        # amount from Razorpay is in minor units (like our DB)
        if amount != record.amount_minor_units or currency.upper() != record.currency.upper():
            # Mismatch -> mark suspicious and don't credit
            record.status = "mismatch"
            extra = record.metadata or {}
            extra["raw_payload"] = payload
            record.metadata = extra
            db.commit()
            return {
                "status": "amount_currency_mismatch",
                "event": event,
                "order_id": order_id,
            }

    # 6) Look up the pack to know how many credits to add
    pack = (
        db.query(CreditPack)
        .filter(CreditPack.pack_id == record.pack_id)
        .first()
    )
    if not pack:
        record.status = "pack_not_found"
        extra = record.metadata or {}
        extra["raw_payload"] = payload
        record.metadata = extra
        db.commit()
        return {"status": "pack_not_found", "event": event, "order_id": order_id}

    # 7) Apply credit top-up
    credits = int(pack.credits)
    try:
        apply_credit_topup(
            db=db,
            user_id=record.user_id,
            credits=credits,
            gateway="razorpay",
            payment_id=payment_id or "",
            reason=f"topup_{pack.pack_id}",
            metadata={
                "order_id": order_id,
                "payment_id": payment_id,
                "event": event,
            },
        )
        record.status = "completed"
        record.gateway_payment_id = payment_id
        extra = record.metadata or {}
        extra["raw_payload"] = payload
        record.metadata = extra
        db.commit()
    except Exception as e:
        db.rollback()
        # Mark as error so we can manually inspect later
        record.status = "error"
        extra = record.metadata or {}
        extra["error"] = str(e)
        extra["raw_payload"] = payload
        record.metadata = extra
        db.commit()
        # Respond 200 anyway so Razorpay doesn't keep retrying endlessly
        return {"status": "error_while_crediting", "event": event, "order_id": order_id}

    return {
        "status": "credited",
        "event": event,
        "order_id": order_id,
        "user_id": record.user_id,
        "credits_added": credits,
    }
