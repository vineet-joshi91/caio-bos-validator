# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 17:57:41 2025

@author: Vineet
"""

# db.py
# -*- coding: utf-8 -*-
"""
Minimal SQLAlchemy setup for CAIO BOS credits/usage.

This is intentionally small and self-contained so it doesn't disturb
the rest of the validator/SLM engine.
"""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/caio_bos",
)

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
