# wallet.py
# -*- coding: utf-8 -*-
"""
Credit wallet + usage helpers for CAIO BOS.

Matches existing Postgres tables:

- credit_wallets(
    id BIGSERIAL PK,
    user_id BIGINT UNIQUE NOT NULL,
    balance_credits BIGINT NOT NULL DEFAULT 0,
    lifetime_added BIGINT NOT NULL DEFAULT 0,
    lifetime_spent BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
  )

- credit_transactions(
    id BIGSERIAL PK,
    user_id BIGINT NOT NULL,
    api_key_id BIGINT,
    amount BIGINT NOT NULL,            -- used as delta credits in this module
    reason VARCHAR(50) NOT NULL,
    meta JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    gateway TEXT DEFAULT 'system',
    gateway_payment_id TEXT,
    metadata JSONB
  )

- usage_daily(
    id BIGSERIAL PK,
    user_id BIGINT NOT NULL,
    usage_date DATE NOT NULL,
    docs_processed INTEGER NOT NULL DEFAULT 0,
    analyses_run INTEGER NOT NULL DEFAULT 0,
    credits_spent INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )

- credit_packs(
    id BIGSERIAL PK,
    pack_id TEXT NOT NULL UNIQUE,
    currency TEXT NOT NULL,
    amount_minor_units INTEGER NOT NULL,
    credits INTEGER NOT NULL,
    display_name TEXT,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    gateway_product_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    BigInteger,
    Integer,
    Text,
    Date,
    TIMESTAMP,
    String,
    func,
    select,
    and_,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, synonym

from db import Base


# ---------------------------------------------------------------------------
# ORM MODELS (aligned with existing DB schema)
# ---------------------------------------------------------------------------


class CreditWallet(Base):
    __tablename__ = "credit_wallets"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, unique=True, nullable=False, index=True)
    balance_credits = Column(BigInteger, nullable=False, default=0)
    lifetime_added = Column(BigInteger, nullable=False, default=0)
    lifetime_spent = Column(BigInteger, nullable=False, default=0)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class CreditTransaction(Base):
    """
    Existing table, now reused for credit wallet history.

    IMPORTANT:
    - We interpret `amount` as "delta credits" for new records:
        +X = top-up, -Y = deduction
    - Old rows keep whatever semantics they had (API usage, etc.)
      but our logic is forward-compatible.
    """

    __tablename__ = "credit_transactions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    api_key_id = Column(BigInteger, nullable=True)
    delta_credits = Column(BigInteger, nullable=False)
    amount = Column(BigInteger, nullable=False)
    reason = Column(String(50), nullable=False)
    meta = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    gateway = Column(Text, nullable=True, server_default="system")
    gateway_payment_id = Column(Text, nullable=True)
    extra_metadata = Column("metadata", JSONB, nullable=True, default=dict)


class UsageDaily(Base):
    __tablename__ = "usage_daily"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    usage_date = Column(Date, nullable=False, index=True)
    docs_processed = Column(Integer, nullable=False, default=0)
    analyses_run = Column(Integer, nullable=False, default=0)
    credits_spent = Column(Integer, nullable=False, default=0)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class CreditPack(Base):
    __tablename__ = "credit_packs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pack_id = Column(Text, nullable=False, unique=True, index=True)
    currency = Column(Text, nullable=False, index=True)
    amount_minor_units = Column(Integer, nullable=False)
    credits = Column(Integer, nullable=False)
    display_name = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    is_active = Column(Integer, nullable=False, default=True)
    gateway_product_id = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# ERRORS
# ---------------------------------------------------------------------------


class InsufficientCreditsError(Exception):
    def __init__(self, user_id: int, required: int, available: int) -> None:
        super().__init__(
            f"User {user_id} has {available} credits, "
            f"but {required} credits are required."
        )
        self.user_id = user_id
        self.required = required
        self.available = available


class DailyLimitReachedError(Exception):
    def __init__(self, user_id: int, date_value: dt.date, cap: int) -> None:
        super().__init__(
            f"User {user_id} reached daily limit ({cap}) on {date_value}."
        )
        self.user_id = user_id
        self.date_value = date_value
        self.cap = cap


# ---------------------------------------------------------------------------
# CORE HELPERS
# ---------------------------------------------------------------------------


def get_or_create_wallet(db: Session, user_id: int) -> CreditWallet:
    """
    Ensure a wallet row exists for this user.
    """
    wallet = db.execute(
        select(CreditWallet).where(CreditWallet.user_id == user_id)
    ).scalar_one_or_none()

    if wallet is None:
        wallet = CreditWallet(
            user_id=user_id,
            balance_credits=0,
            lifetime_added=0,
            lifetime_spent=0,
        )
        db.add(wallet)
        db.flush()
    return wallet


def get_balance(db: Session, user_id: int) -> int:
    val = db.execute(
        select(CreditWallet.balance_credits).where(CreditWallet.user_id == user_id)
    ).scalar_one_or_none()
    return int(val or 0)


def _get_or_create_usage_daily(
    db: Session,
    user_id: int,
    for_date: Optional[dt.date] = None,
) -> UsageDaily:
    if for_date is None:
        for_date = dt.date.today()

    usage = db.execute(
        select(UsageDaily).where(
            and_(
                UsageDaily.user_id == user_id,
                UsageDaily.usage_date == for_date,
            )
        )
    ).scalar_one_or_none()

    if usage is None:
        usage = UsageDaily(
            user_id=user_id,
            usage_date=for_date,
            docs_processed=0,
            analyses_run=0,
            credits_spent=0,
        )
        db.add(usage)
        db.flush()

    return usage


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def apply_credit_topup(
    db: Session,
    user_id: int,
    credits: int,
    gateway: str = "system",
    payment_id: Optional[str] = None,
    reason: str = "topup",
    metadata: Optional[Dict[str, Any]] = None,
) -> CreditWallet:
    """
    Add credits to a user's wallet and record it in credit_transactions.

    - `credits` MUST be positive.
    - `amount` column is stored as +credits.
    """
    if credits <= 0:
        raise ValueError("credits must be positive for topup")

    wallet = get_or_create_wallet(db, user_id)

    wallet.balance_credits += credits
    wallet.lifetime_added += credits

    tx = CreditTransaction(
        user_id=user_id,
        api_key_id=None,
        delta_credits=credits,   # REQUIRED by DB
        amount=credits,          # keep for compatibility
        reason=reason,
        gateway=gateway,
        gateway_payment_id=payment_id,
        extra_metadata=metadata or {},
    )

    db.add(tx)

    return wallet


def consume_credits_and_record_usage(
    db: Session,
    user_id: int,
    credits_required: int,
    doc_increment: int = 1,
    reason: str = "bos_run",
    gateway: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
    daily_doc_cap: Optional[int] = None,
    for_date: Optional[dt.date] = None,
) -> CreditWallet:
    """
    Atomically:

    - enforce daily document cap (if provided)
    - ensure enough credits
    - deduct credits from credit_wallets
    - update lifetime_spent
    - update usage_daily (docs_processed, analyses_run, credits_spent)
    - insert a row into credit_transactions with `amount = -credits_required`
    """
    if credits_required <= 0:
        raise ValueError("credits_required must be positive")

    wallet = get_or_create_wallet(db, user_id)
    usage = _get_or_create_usage_daily(db, user_id, for_date)

    # 1) Daily cap (typically only for Demo)
    if daily_doc_cap is not None:
        if usage.docs_processed + doc_increment > daily_doc_cap:
            raise DailyLimitReachedError(user_id, usage.usage_date, daily_doc_cap)

    # 2) Check credits
    if wallet.balance_credits < credits_required:
        raise InsufficientCreditsError(
            user_id=user_id,
            required=credits_required,
            available=wallet.balance_credits,
        )

    # 3) Deduct
    wallet.balance_credits -= credits_required
    wallet.lifetime_spent += credits_required

    # 4) Usage row
    usage.docs_processed += doc_increment
    usage.analyses_run += 1
    usage.credits_spent += credits_required

    # 5) Transaction log (NEGATIVE amount)
    tx = CreditTransaction(
        user_id=user_id,
        api_key_id=None,
        amount=-credits_required,
        reason=reason,
        gateway=gateway,
        gateway_payment_id=None,
        extra_metadata=metadata or {},
    )
    db.add(tx)

    return wallet


def list_transactions(
    db: Session,
    user_id: int,
    limit: int = 20,
    offset: int = 0,
) -> List[CreditTransaction]:
    """
    Return a page of transactions for a user, newest first.
    This must match how wallet_api.py calls it.
    """
    return (
        db.query(CreditTransaction)
        .filter(CreditTransaction.user_id == user_id)
        .order_by(CreditTransaction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

def get_wallet(db: Session, user_id: int) -> CreditWallet:
    """Alias for get_or_create_wallet, matching earlier design."""
    return get_or_create_wallet(db, user_id)


def consume_credits(
    db: Session,
    user_id: int,
    credits: int,
    reason: str = "bos_run",
    daily_doc_cap: Optional[int] = None,
    doc_increment: int = 1,
    gateway: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
) -> CreditWallet:
    """
    Alias for consume_credits_and_record_usage, matching earlier design.
    """
    return consume_credits_and_record_usage(
        db=db,
        user_id=user_id,
        credits_required=credits,
        doc_increment=doc_increment,
        reason=reason,
        gateway=gateway,
        metadata=metadata,
        daily_doc_cap=daily_doc_cap,
    )

class PaymentRecord(Base):
    __tablename__ = "payment_records"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    pack_id = Column(Text, nullable=False, index=True)
    gateway = Column(Text, nullable=False)  # 'razorpay', 'paypal', etc.
    gateway_order_id = Column(Text, nullable=False, index=True)
    gateway_payment_id = Column(Text, nullable=True)
    currency = Column(Text, nullable=False)
    amount_minor_units = Column(Integer, nullable=False)
    type = Column(Text, nullable=False, default="credit_topup")
    status = Column(Text, nullable=False)  # 'initiated','processing','completed','failed','cancelled'
    tx_metadata = Column("metadata", JSONB, nullable=True, default=dict)
    extra_metadata = synonym("tx_metadata")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
