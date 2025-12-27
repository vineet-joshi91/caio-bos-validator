# -*- coding: utf-8 -*-
"""
webhooks_razorpay.py

Razorpay webhook handler for CAIO_BOS credit topups.

Key design choices (important):
- We DO NOT use `record.metadata` anywhere.
  PaymentRecord has Python attribute `tx_metadata` mapped to DB column "metadata".
  Using `metadata` as a Python attribute causes collisions/confusion in SQLAlchemy ecosystems.
- Signature verification is done via HMAC SHA256(body, webhook_secret) and constant-time compare.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from db import get_db
from wallet import CreditPack, PaymentRecord, apply_credit_topup

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _get_webhook_secret() -> str:
    """
    Support both env var names (in case you changed it earlier).
    Prefer RAZORPAY_WEBHOOK_SECRET.
    """
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET") or os.getenv("RAZORPAY_WEBHOOK_SECRET_KEY")
    if not secret:
        raise ValueError("RAZORPAY_WEBHOOK_SECRET is not set")
    return secret


def _verify_signature(body: bytes, signature: str) -> None:
    """
    Verify Razorpay webhook signature.

    Razorpay sends:
      X-Razorpay-Signature = HMAC_SHA256(body, webhook_secret)
    """
    if not signature:
        # Guardrail: FastAPI will pass None if header missing
        raise ValueError("Missing signature header")

    secret = _get_webhook_secret()
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    # Constant-time compare to avoid timing attacks
    if not hmac.compare_digest(expected, signature):
        raise ValueError("Invalid webhook signature")


def _find_payment_record(db: Session, order_id: str) -> Optional[PaymentRecord]:
    return (
        db.query(PaymentRecord)
        .filter(
            PaymentRecord.gateway == "razorpay",
            PaymentRecord.gateway_order_id == order_id,
        )
        .first()
    )


def _safe_set_metadata(record: PaymentRecord, patch: Dict[str, Any]) -> None:
    """
    Safe helper to update PaymentRecord metadata without ever touching `record.metadata`.
    We always use `record.tx_metadata` (Python attribute) mapped to DB column "metadata".
    """
    current = record.tx_metadata or {}
    # Ensure it's a dict (sometimes can be a JSON string if inserted wrongly earlier)
    if isinstance(current, str):
        try:
            current = json.loads(current)
        except Exception:
            current = {"_raw": current}
    if not isinstance(current, dict):
        current = {"_raw": current}

    current.update(patch)
    record.tx_metadata = current


@router.post("/razorpay")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: Optional[str] = Header(default=None, alias="X-Razorpay-Signature"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Handle Razorpay credit-pack payment events.

    Expected flow:
    1) Frontend calls /wallet/create-credit-order
       - creates Razorpay order
       - inserts PaymentRecord with status='initiated'
    2) Razorpay sends webhook when payment succeeds/fails
    3) We:
       - verify signature
       - parse payload
       - match PaymentRecord by order_id
       - on success: apply_credit_topup() and mark PaymentRecord status='completed'
       - on failure: mark status='failed'
    """
    body = await request.body()

    # 1) Verify signature (fail safe)
    if not x_razorpay_signature:
        raise HTTPException(status_code=400, detail="Missing signature header")
    try:
        _verify_signature(body, x_razorpay_signature)
    except ValueError as e:
        # Signature problems => reject (do not silently accept)
        raise HTTPException(status_code=400, detail=str(e))

    # 2) Parse payload
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in webhook")

    event = (payload.get("event") or "").lower().strip()

    # We only act on these events
    handled_events = {
        "payment.captured",
        "order.paid",
        "payment.failed",
        "order.payment_failed",
    }
    if event not in handled_events:
        return {"status": "ignored", "event": event}

    # 3) Extract order_id, payment_id, amount, currency
    order_id: Optional[str] = None
    payment_id: Optional[str] = None
    amount: Optional[int] = None
    currency: Optional[str] = None

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
        # Without order_id we can't map to PaymentRecord
        return {"status": "no_order_id", "event": event}

    # 4) Find PaymentRecord
    record = _find_payment_record(db, order_id)
    if not record:
        # Could be unrelated; don't crash.
        return {"status": "no_matching_record", "event": event, "order_id": order_id}

    # Idempotency
    if record.status == "completed":
        return {"status": "already_completed", "event": event, "order_id": order_id}

    # Always store raw payload for audit trail (small risk: payload can be big; acceptable for now)
    _safe_set_metadata(record, {"last_event": event, "raw_payload": payload})

    # 5) Handle failure events
    if event in {"payment.failed", "order.payment_failed"}:
        record.status = "failed"
        # Optional: store failure reason if available
        try:
            failure_reason = (
                (payload.get("payload") or {})
                .get("payment", {})
                .get("entity", {})
                .get("error_description")
            )
            if failure_reason:
                _safe_set_metadata(record, {"failure_reason": failure_reason})
        except Exception:
            pass

        db.commit()
        return {"status": "marked_failed", "event": event, "order_id": order_id}

    # 6) Success event (payment.captured or order.paid)
    # Sanity check amount/currency against PaymentRecord
    if amount is not None and currency:
        try:
            if int(amount) != int(record.amount_minor_units) or currency.upper() != record.currency.upper():
                record.status = "mismatch"
                _safe_set_metadata(
                    record,
                    {
                        "mismatch": {
                            "expected_amount_minor_units": record.amount_minor_units,
                            "expected_currency": record.currency,
                            "got_amount_minor_units": amount,
                            "got_currency": currency,
                        }
                    },
                )
                db.commit()
                return {"status": "amount_currency_mismatch", "event": event, "order_id": order_id}
        except Exception:
            # If conversion fails for any reason, fail safe
            record.status = "mismatch"
            _safe_set_metadata(record, {"mismatch_error": "Unable to compare amount/currency"})
            db.commit()
            return {"status": "amount_currency_mismatch", "event": event, "order_id": order_id}

    # 7) Load pack for credits
    pack = db.query(CreditPack).filter(CreditPack.pack_id == record.pack_id).first()
    if not pack:
        record.status = "pack_not_found"
        db.commit()
        return {"status": "pack_not_found", "event": event, "order_id": order_id}

    credits = int(pack.credits)

    # 8) Apply topup (atomic-ish)
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
        if payment_id:
            record.gateway_payment_id = payment_id

        _safe_set_metadata(
            record,
            {
                "credited": True,
                "credits_added": credits,
                "credited_for_pack": pack.pack_id,
            },
        )

        db.commit()

    except Exception as e:
        db.rollback()

        # Re-load record to safely mark error (session might have rolled back changes)
        record2 = _find_payment_record(db, order_id)
        if record2:
            record2.status = "error"
            _safe_set_metadata(record2, {"crediting_error": str(e)})
            db.commit()

        # Respond 200 anyway so Razorpay doesn't keep retrying forever.
        # (If you want retries, change this to HTTPException(500))
        return {"status": "error_while_crediting", "event": event, "order_id": order_id}

    return {
        "status": "credited",
        "event": event,
        "order_id": order_id,
        "user_id": record.user_id,
        "credits_added": credits,
    }
