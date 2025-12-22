# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 16:58:24 2025

@author: Vineet
"""

# routes_bos_auth.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import Column, Integer, String, Boolean, Text, TIMESTAMP
from sqlalchemy.orm import Session

from db import Base, get_db

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MIN", "43200"))  # 30 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# We mount under /bos (matches your nginx prefix)
router = APIRouter(tags=["bos-auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


# ---------------------------------------------------------------------
# DB model (matches your existing public.users table)
# ---------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=True)
    is_admin = Column(Boolean, nullable=False, default=False)
    is_paid = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP(timezone=False), nullable=True)
    username = Column(Text, nullable=True)
    tier = Column(Text, nullable=False, default="demo")
    is_test = Column(Boolean, nullable=False, default=False)
    # (other columns exist in DB; we don’t need to map them all here)


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class MeResponse(BaseModel):
    id: int
    email: EmailStr
    is_admin: bool
    is_paid: bool
    tier: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    me: MeResponse


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


def _create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "iat": int(now.timestamp()), "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email.ilike(email)).first()


def _get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = int(sub)
    except (JWTError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = _get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@router.post("/login", response_model=LoginResponse)
def bos_login(req: LoginRequest, db: Session = Depends(get_db)):
    user = _get_user_by_email(db, req.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.hashed_password:
        raise HTTPException(status_code=401, detail="Password not set for this user")

    if not _verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_access_token(user.id)

    return LoginResponse(
        access_token=token,
        me=MeResponse(
            id=user.id,
            email=user.email,
            is_admin=bool(user.is_admin),
            is_paid=bool(user.is_paid),
            tier=user.tier or "demo",
        ),
    )


@router.get("/me", response_model=MeResponse)
def bos_me(user: User = Depends(get_current_user)):
    return MeResponse(
        id=user.id,
        email=user.email,
        is_admin=bool(user.is_admin),
        is_paid=bool(user.is_paid),
        tier=user.tier or "demo",
    )

@router.post("/token")
def bos_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Swagger sends username/password in form fields
    email = (form_data.username or "").strip()

    user = _get_user_by_email(db, email)
    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not _verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_access_token(user.id)

    # IMPORTANT: return the standard OAuth2 token shape
    return {"access_token": token, "token_type": "bearer"}
