# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 16:49:47 2026

@author: Vineet
"""

# api/utils/auth.py
"""
Authentication utilities
"""
import logging
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from routes_bos_auth import get_current_user, User
from db import get_db

logger = logging.getLogger(__name__)


def get_authenticated_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> tuple[User, Session]:
    """
    Get authenticated user and database session.
    
    Returns:
        (user, db_session)
    
    Raises:
        HTTPException 401 if not authenticated
    """
    return current_user, db


def get_user_tier(user: User) -> str:
    """
    Determine user's tier based on their account status.
    
    Args:
        user: Authenticated user object
    
    Returns:
        "premium" if paid/admin, "standard" otherwise
    """
    if user.is_admin or user.is_paid:
        return "premium"
    return "standard"


def log_user_action(user: User, action: str, details: dict = None):
    """
    Log user action with context.
    
    Args:
        user: User performing action
        action: Description of action
        details: Optional additional details
    """
    detail_str = f" {details}" if details else ""
    logger.info(f"👤 User {user.id} ({user.email}): {action}{detail_str}")