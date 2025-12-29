# -*- coding: utf-8 -*-
"""
webhooks_razorpay.py

Razorpay webhook handler for CAIO BOS credit top-ups.

Critical rules:
- DO NOT use model attribute name `metadata` anywhere (SQLAlchemy reserves it).
- PaymentRecord stores JSON in DB column named "metadata" but mapped as `tx_metadata`
  (and aliased as `extra_metadata` via synonym in your model).
- Idempotent: do not credit twice for the same order.
- Fail-safe: reject if signature missing/invalid.
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
    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET") or os.getenv("RAZORPAY_WEBHOOK_SECRET_KEY")
    if not secret:
        # Don't crash app at import time; fail only when webhook is called.
        raise ValueError("RAZORPAY_WEBHOOK_SECRET is not set")
    return secret


def _verify_signature(body: bytes, signature: Optional[str]) -> None:
    """
    Razorpay sends header: X-Razorpay-Signature
    Signature = HMAC_SHA256(body, webhook_secret) hex digest
    """
    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature header")

    try:
        secret = _get_webhook_secret()
    except ValueError as e:
        # If secret isn't configured, we must reject (fail-safe).
        raise HTTPException(status_code=500, detail=str(e))

    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")


def _get_record_metadata(record: PaymentRecord) -> Dict[str, Any]:
    """
    Safe getter for PaymentRecord JSON metadata.

    IMPORTANT:
    - Never touch record.metadata (SQLAlchemy MetaData)
    - Use record.tx_metadata or record.extra_metadata (synonym)
    """
    data = None

    # Prefer tx_metadata
    if hasattr(record, "tx_metadata"):
        data = getattr(record, "tx_metadata", None)

    # Fallback to synonym name if present
    if data is None and hasattr(record, "extra_metadata"):
        data = getattr(record, "extra_metadata", None)

    if data is None:
        return {}

    if isinstance(data, dict):
        return data

    # If something weird got stored, coerce to dict safely
    try:
        return dict(data)  # type: ignore[arg-type]
    except Exception:
        return {}


def _set_record_metadata(record: PaymentRecord, new_meta: Dict[str, Any]) -> None:
    """
    Safe setter for PaymentRecord JSON metadata.
    Writes to tx_metadata (preferred), else extra_metadata.
    """
    if hasattr(record, "tx_metadata"):
        setattr(record, "tx_metadata", new_meta)
        return
    if hasattr(record, "extra_metadata"):
        setattr(record, "extra_metadata", new_meta)
        return

    # If neither exists, we cannot store; but don't crash webhook.
    return


def _find_payment_record(db: Session, order_id: str) -> Optional[PaymentRecord]:
    return (
        db.query(PaymentRecord)
        .filter(
            PaymentRecord.gateway == "razorpay",
            PaymentRecord.gateway_order_id == order_id,
        )
        .first()
    )


def _extract_event_ids(payload: Dict[str, Any], event: str) -> Dict[str, Optional[Any]]:
    """
    Extract order_id, payment_id, amount, currency from common Razorpay payload shapes.
    """
    order_id = None
    payment_id = None
    amount = None
    currency = None

    pl = payload.get("payload") or {}

    if event.startswith("payment."):
        payment = (pl.get("payment") or {}).get("entity") or {}
        order_id = payment.get("order_id")
        payment_id = payment.get("id")
        amount = payment.get("amount")
        currency = payment.get("currency")

    elif event.startswith("order."):
        order = (pl.get("order") or {}).get("entity") or {}
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

    # 2) Parse JSON
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in webhook")

    event = (payload.get("event") or "").lower().strip()

    # Keep this tight (you can expand later)
    relevant_events = {
        "payment.captured",
        "payment.failed",
        "order.paid",
        "order.payment_failed",
    }

    if event not in relevant_events:
        return {"status": "ignored", "event": event}

    ids = _extract_event_ids(payload, event)
    order_id = ids["order_id"]
    payment_id = ids["payment_id"]
    amount = ids["amount"]
    currency = ids["currency"]

    if not order_id:
        # Acknowledge but do nothing
        return {"status": "no_order_id", "event": event}

    # 3) Find PaymentRecord
    record = _find_payment_record(db, order_id)
    if not record:
        return {"status": "no_matching_record", "event": event, "order_id": order_id}

    # Idempotency: already completed -> do nothing
    if getattr(record, "status", None) == "completed":
        return {"status": "already_completed", "event": event, "order_id": order_id}

    # 4) Failure events
    if event in {"payment.failed", "order.payment_failed"}:
        record.status = "failed"
        meta = _get_record_metadata(record)
        meta.update(
            {
                "failure_event": event,
                "order_id": order_id,
                "payment_id": payment_id,
            }
        )
        # Store raw payload (optional - can be large; keep if you want audit)
        meta["raw_payload"] = payload
        _set_record_metadata(record, meta)
        db.commit()
        return {"status": "marked_failed", "event": event, "order_id": order_id}

    # 5) Success event: sanity check amount/currency if present
    if amount is not None and currency:
        try:
            if int(amount) != int(record.amount_minor_units) or str(currency).upper() != str(record.currency).upper():
                record.status = "mismatch"
                meta = _get_record_metadata(record)
                meta.update(
                    {
                        "event": event,
                        "order_id": order_id,
                        "payment_id": payment_id,
                        "expected_amount_minor_units": record.amount_minor_units,
                        "got_amount_minor_units": amount,
                        "expected_currency": record.currency,
                        "got_currency": currency,
                    }
                )
                meta["raw_payload"] = payload
                _set_record_metadata(record, meta)
                db.commit()
                return {"status": "amount_currency_mismatch", "event": event, "order_id": order_id}
        except Exception:
            # If comparison fails for any reason, mark suspicious
            record.status = "mismatch"
            meta = _get_record_metadata(record)
            meta["raw_payload"] = payload
            _set_record_metadata(record, meta)
            db.commit()
            return {"status": "amount_currency_check_failed", "event": event, "order_id": order_id}

    # 6) Find pack
    pack = db.query(CreditPack).filter(CreditPack.pack_id == record.pack_id).first()
    if not pack:
        record.status = "pack_not_found"
        meta = _get_record_metadata(record)
        meta.update({"event": event, "order_id": order_id, "payment_id": payment_id})
        meta["raw_payload"] = payload
        _set_record_metadata(record, meta)
        db.commit()
        return {"status": "pack_not_found", "event": event, "order_id": order_id}

    # 7) Credit wallet
    credits_to_add = int(pack.credits)

    try:
        apply_credit_topup(
            db=db,
            user_id=record.user_id,
            credits=credits_to_add,
            gateway="razorpay",
            payment_id=str(payment_id or ""),
            reason=f"topup_{pack.pack_id}",
            metadata={
                "event": event,
                "order_id": order_id,
                "payment_id": payment_id,
                "pack_id": pack.pack_id,
            },
        )

        record.status = "completed"
        record.gateway_payment_id = str(payment_id or "")

        meta = _get_record_metadata(record)
        meta.update({"event": event, "order_id": order_id, "payment_id": payment_id, "credited": credits_to_add})
        meta["raw_payload"] = payload
        _set_record_metadata(record, meta)

        db.commit()

    except Exception as e:
        db.rollback()
        record.status = "error"
        meta = _get_record_metadata(record)
        meta.update({"event": event, "order_id": order_id, "payment_id": payment_id, "error": str(e)})
        meta["raw_payload"] = payload
        _set_record_metadata(record, meta)
        db.commit()

        # Return 200 so Razorpay doesn't retry forever; you can monitor status='error'
        return {"status": "error_while_crediting", "event": event, "order_id": order_id}

    return {
        "status": "credited",
        "event": event,
        "order_id": order_id,
        "user_id": record.user_id,
        "credits_added": credits_to_add,
    }
