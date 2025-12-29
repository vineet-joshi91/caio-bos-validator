# -*- coding: utf-8 -*-
"""
Razorpay webhooks handler for CAIO BOS credit packs.

Key design decisions:
- DO NOT use model attribute name `metadata` anywhere (SQLAlchemy reserves it).
- We persist JSON into PaymentRecord column named "metadata" but mapped as `tx_metadata`.
- Idempotent: if record already completed, no double credit.
- Fail-safe: if signature missing/invalid, reject.
"""

from __future__ import annotations

import json
import os
import hmac
import hashlib
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from db import get_db
from wallet import CreditPack, PaymentRecord, apply_credit_topup

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _get_webhook_secret() -> str:
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET") or os.getenv("RAZORPAY_WEBHOOK_SECRET_KEY")
    if not secret:
        # fail safe
        raise HTTPException(status_code=500, detail="RAZORPAY_WEBHOOK_SECRET not set on server")
    return secret


def _verify_signature(body: bytes, signature: Optional[str]) -> None:
    """
    Verify Razorpay webhook signature.

    Razorpay sends:
      X-Razorpay-Signature = HMAC_SHA256(body, webhook_secret)
    """
    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature header")

    secret = _get_webhook_secret()
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")


def _safe_get_record_meta(record: PaymentRecord) -> Dict[str, Any]:
    """
    PaymentRecord JSON column is mapped as `tx_metadata` (NOT `metadata`).
    Return a mutable dict always.
    """
    current = getattr(record, "tx_metadata", None)
    if isinstance(current, dict):
        return dict(current)  # copy
    return {}


def _safe_set_record_meta(record: PaymentRecord, data: Dict[str, Any]) -> None:
    setattr(record, "tx_metadata", data)


def _find_payment_record(db: Session, order_id: str) -> Optional[PaymentRecord]:
    return (
        db.query(PaymentRecord)
        .filter(
            PaymentRecord.gateway == "razorpay",
            PaymentRecord.gateway_order_id == order_id,
        )
        .first()
    )


def _extract_from_payload(payload: Dict[str, Any], event: str) -> Dict[str, Any]:
    """
    Return dict with: order_id, payment_id, amount, currency
    Razorpay amounts are minor units.
    """
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

    return {
        "order_id": order_id,
        "payment_id": payment_id,
        "amount": amount,
        "currency": currency,
    }


@router.post("/razorpay")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: Optional[str] = Header(default=None, alias="X-Razorpay-Signature"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    body = await request.body()

    # 1) Verify signature (reject if bad)
    _verify_signature(body, x_razorpay_signature)

    # 2) Parse JSON payload
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in webhook")

    event = (payload.get("event") or "").lower().strip()

    # Accept only events we care about
    interesting = {"payment.captured", "order.paid", "payment.failed", "order.payment_failed"}
    if event not in interesting:
        return {"status": "ignored", "event": event}

    extracted = _extract_from_payload(payload, event)
    order_id = extracted["order_id"]
    payment_id = extracted["payment_id"]
    amount = extracted["amount"]
    currency = extracted["currency"]

    if not order_id:
        return {"status": "no_order_id", "event": event}

    # 3) Find PaymentRecord
    record = _find_payment_record(db, order_id)
    if not record:
        return {"status": "no_matching_record", "event": event, "order_id": order_id}

    # 4) Idempotency
    if record.status == "completed":
        return {"status": "already_completed", "event": event, "order_id": order_id}

    # 5) Failure
    if event in {"payment.failed", "order.payment_failed"}:
        record.status = "failed"
        meta = _safe_get_record_meta(record)
        meta["failure_event"] = event
        meta["raw_payload"] = payload
        _safe_set_record_meta(record, meta)
        db.commit()
        return {"status": "marked_failed", "event": event, "order_id": order_id}

    # 6) Success sanity check (amount/currency)
    if amount is not None and currency:
        if int(amount) != int(record.amount_minor_units) or str(currency).upper() != str(record.currency).upper():
            record.status = "mismatch"
            meta = _safe_get_record_meta(record)
            meta["raw_payload"] = payload
            meta["expected_amount_minor_units"] = record.amount_minor_units
            meta["received_amount_minor_units"] = amount
            meta["expected_currency"] = record.currency
            meta["received_currency"] = currency
            _safe_set_record_meta(record, meta)
            db.commit()
            return {"status": "amount_currency_mismatch", "event": event, "order_id": order_id}

    # 7) Load pack
    pack = db.query(CreditPack).filter(CreditPack.pack_id == record.pack_id).first()
    if not pack:
        record.status = "pack_not_found"
        meta = _safe_get_record_meta(record)
        meta["raw_payload"] = payload
        _safe_set_record_meta(record, meta)
        db.commit()
        return {"status": "pack_not_found", "event": event, "order_id": order_id}

    credits = int(pack.credits)

    # 8) Apply credit topup
    try:
        apply_credit_topup(
            db=db,
            user_id=int(record.user_id),
            credits=credits,
            gateway="razorpay",
            payment_id=payment_id or "",
            reason=f"topup_{pack.pack_id}",
            metadata={"order_id": order_id, "payment_id": payment_id, "event": event},
        )
        record.status = "completed"
        record.gateway_payment_id = payment_id

        meta = _safe_get_record_meta(record)
        meta["raw_payload"] = payload
        meta["credited_credits"] = credits
        _safe_set_record_meta(record, meta)

        db.commit()
        return {"status": "credited", "event": event, "order_id": order_id, "user_id": record.user_id, "credits_added": credits}
    except Exception as e:
        db.rollback()
        record.status = "error"
        meta = _safe_get_record_meta(record)
        meta["error"] = str(e)
        meta["raw_payload"] = payload
        _safe_set_record_meta(record, meta)
        db.commit()
        # Return 200 so Razorpay doesn’t retry forever; we can reconcile manually.
        return {"status": "error_while_crediting", "event": event, "order_id": order_id}
    