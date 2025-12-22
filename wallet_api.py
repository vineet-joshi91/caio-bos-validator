# wallet_api.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import time
from typing import List, Optional

import razorpay
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from wallet import (
    get_balance,
    list_transactions,
    CreditTransaction,
    CreditPack,
    PaymentRecord,
)

from routes_bos_auth import get_current_user, User as AuthUser

router = APIRouter(prefix="/wallet", tags=["wallet"])

# ---------------------------------------------------------------------------
# Razorpay client
# ---------------------------------------------------------------------------

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    razorpay_client = razorpay.Client(
        auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
    )
else:
    razorpay_client = None

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class WalletBalanceResponse(BaseModel):
    user_id: int
    balance_credits: int


class TransactionOut(BaseModel):
    id: int
    user_id: int
    amount: int
    reason: str
    gateway: Optional[str] = None
    gateway_payment_id: Optional[str] = None
    created_at: str
    metadata: Optional[dict] = None

    @classmethod
    def from_orm_tx(cls, tx: CreditTransaction) -> "TransactionOut":
        # Support either extra_metadata (new) or metadata (old), just in case
        meta = getattr(tx, "extra_metadata", None)
        if meta is None:
            meta = getattr(tx, "metadata", None)

        return cls(
            id=tx.id,
            user_id=tx.user_id,
            amount=int(tx.amount),
            reason=tx.reason,
            gateway=getattr(tx, "gateway", None),
            gateway_payment_id=getattr(tx, "gateway_payment_id", None),
            created_at=tx.created_at.isoformat() if tx.created_at else "",
            metadata=meta,
        )


class CreateOrderRequest(BaseModel):
    user_id: int
    pack_id: str
    gateway: str = "razorpay"  # we support razorpay for now


class CreateOrderResponse(BaseModel):
    gateway: str
    order_id: str
    amount_minor_units: int
    currency: str
    pack_id: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    razorpay_key_id: Optional[str] = None  # frontend needs this


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/balance", response_model=WalletBalanceResponse)
def wallet_balance(
    user_id: int = Query(..., description="User ID whose wallet to inspect"),
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
) -> WalletBalanceResponse:
    if not getattr(current_user, "is_admin", False) and user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed to view other users' wallet")

    balance = get_balance(db, user_id=user_id)
    return WalletBalanceResponse(user_id=user_id, balance_credits=balance)


@router.get("/transactions", response_model=List[TransactionOut])
def wallet_transactions(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> List[TransactionOut]:
    txs = list_transactions(db, user_id=user_id, limit=limit, offset=offset)
    return [TransactionOut.from_orm_tx(tx) for tx in txs]


@router.post("/create-credit-order", response_model=CreateOrderResponse)
def create_credit_order(
    payload: CreateOrderRequest,
    db: Session = Depends(get_db),
) -> CreateOrderResponse:
    """
    Start a credit-pack purchase flow.

    - Validate pack_id against credit_packs (is_active = true).
    - Create a Razorpay order (for now we only support 'razorpay').
    - Insert a payment_records row with status='initiated'.
    - Return order details (order_id, amount, currency, pack info) to frontend.
    """
    if payload.gateway.lower() != "razorpay":
        raise HTTPException(
            status_code=400,
            detail="Currently only Razorpay gateway is supported for credit packs.",
        )

    if razorpay_client is None:
        raise HTTPException(
            status_code=500,
            detail="Razorpay is not configured on the server.",
        )

    # 1) Find the pack
    pack = (
        db.query(CreditPack)
        .filter(
            CreditPack.pack_id == payload.pack_id,
            CreditPack.is_active == True,  # noqa: E712
        )
        .first()
    )
    if not pack:
        raise HTTPException(
            status_code=404,
            detail=f"Credit pack {payload.pack_id!r} not found or inactive.",
        )

    amount = int(pack.amount_minor_units)
    currency = pack.currency.upper()

    # 2) Create Razorpay order
    receipt = f"credits:{payload.user_id}:{payload.pack_id}:{int(time.time())}"

    try:
        order = razorpay_client.order.create(
            {
                "amount": amount,  # in the smallest unit (paise)
                "currency": currency,
                "receipt": receipt,
                "notes": {
                    "user_id": str(payload.user_id),
                    "pack_id": payload.pack_id,
                    "type": "credit_topup",
                },
            }
        )
    except razorpay.errors.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"Razorpay error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create order: {e}")

    order_id = order.get("id")
    if not order_id:
        raise HTTPException(
            status_code=500,
            detail="Razorpay order creation returned no order id.",
        )

    # 3) Store payment_records row
    record = PaymentRecord(
        user_id=payload.user_id,
        pack_id=payload.pack_id,
        gateway="razorpay",
        gateway_order_id=order_id,
        gateway_payment_id=None,
        currency=currency,
        amount_minor_units=amount,
        type="credit_topup",
        status="initiated",
        extra_metadata={"receipt": receipt},
    )
    db.add(record)
    db.commit()

    # 4) Return details to frontend
    return CreateOrderResponse(
        gateway="razorpay",
        order_id=order_id,
        amount_minor_units=amount,
        currency=currency,
        pack_id=payload.pack_id,
        display_name=pack.display_name,
        description=pack.description,
        razorpay_key_id=RAZORPAY_KEY_ID,
    )
