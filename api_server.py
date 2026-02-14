# api_server.py — drop-in replacement
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json, subprocess, sys, tempfile, shutil, os
import io
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from wallet_api import router as wallet_router
from webhooks_razorpay import router as razorpay_webhook_router
from routes_bos_auth import router as bos_auth_router
from routes_bos_auth import get_current_user, User
from middleware.security import SecurityHeadersMiddleware, RequestLoggingMiddleware
from db import get_db
from api.services.document_service import extract_text_with_meta
from api.utils.auth import get_user_tier, log_user_action

import re
import os
from dotenv import load_dotenv
from logging_config import setup_logging

load_dotenv()

DEBUG = os.getenv("DEBUG", "0") == "1"
logger = setup_logging(debug=DEBUG)

logger.info("🚀 CAIO BOS API Server starting...")

from wallet import (
    CreditWallet,
    CreditTransaction,
    CreditPack,
    PaymentRecord,
    get_balance,
    get_or_create_wallet,
    apply_credit_topup,
    InsufficientCreditsError,
)

from db import SessionLocal
from tier_config import TIER_CONFIG
from bos_credits import charge_bos_run

app = FastAPI(title="CAIO BOS – EA API")

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)

import requests
from datetime import datetime
from sqlalchemy import text

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "CAIO BOS API",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/ollama")
async def check_ollama():
    """Check if Ollama is running and models are available"""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.ok:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            return {
                "status": "healthy",
                "ollama_running": True,
                "available_models": model_names,
                "primary_available": PRIMARY_EA_MODEL in model_names,
                "fallback_available": FALLBACK_EA_MODEL in model_names
            }
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {
            "status": "unhealthy",
            "ollama_running": False,
            "error": str(e)
        }

@app.get("/health/database")
async def check_database():
    """Check database connectivity"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "healthy", "database_connected": True}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": str(e)
        }

# Wallet + Payments routers MUST be included first
app.include_router(wallet_router)
app.include_router(razorpay_webhook_router)
app.include_router(bos_auth_router, prefix="/bos-auth", tags=["bos-auth"])

PRIMARY_EA_MODEL = "qwen2.5:3b-instruct"
FALLBACK_EA_MODEL = "qwen2.5:1.5b-instruct"

def charge_or_pass(user_id: int, plan_tier: str | None, brain: str):
    """
    Attempt to charge credits for a BOS run.
    Logs failures but doesn't block execution.
    """
    try:
        db = SessionLocal()
        charge_bos_run(db, user_id=user_id, plan_tier=plan_tier, brain=brain)
        db.commit()
        logger.info(f"✅ Charged user {user_id} for {brain} analysis")
    except InsufficientCreditsError as e:
        logger.warning(f"⚠️  User {user_id} insufficient credits: {e}")
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        # Log the error but allow the run to continue
        logger.error(f"❌ Failed to charge user {user_id}: {e}", exc_info=True)
        # Don't raise - let the analysis proceed
    finally:
        try:
            db.close()
        except Exception:
            pass


# -------------------- Models --------------------
class EARequest(BaseModel):
    packet: dict
    user_id: int
    plan_tier: str = "demo"
    model: Optional[str] = None
    timeout_sec: int = 300
    num_predict: int = 512

class BrainRequest(BaseModel):
    packet: dict
    user_id: int
    plan_tier: str = "demo"
    model: Optional[str] = None
    timeout_sec: int = 300
    num_predict: int = 512
    brain: str

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Helpers --------------------
def repo_root() -> str:
    # Expect to run from /opt/caio-bos-validator
    return str(Path(__file__).resolve().parent)

def run_slm(
    input_json_path: str,
    brain: str,
    *,
    model: Optional[str],
    timeout_sec: int,
    num_predict: int,
) -> Dict[str, Any]:
    """
    Thin wrapper around `python -m slm.run_slm ...` so:
    - Works in dev and on Render
    - Always returns a dict (either result or structured error)
    """
    root = repo_root()
    cmd = [
        sys.executable,
        "-m",
        "slm.run_slm",
        "--input",
        input_json_path,
        "--brain",
        brain,
        "--timeout",
        str(timeout_sec),
        "--num_predict",
        str(num_predict),
    ]
    if model:
        cmd += ["--model", model]

    try:
        p = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_sec + 60,
        )
        out = {
            "stdout": p.stdout or "",
            "stderr": p.stderr or "",
            "returncode": p.returncode,
        }
        if p.returncode != 0:
            out["error"] = "SLM failed"
        # If run_slm prints JSON on stdout, prefer parsing it
        try:
            j = json.loads(p.stdout)
            if isinstance(j, dict):
                return j
        except Exception:
            pass
        return out
    except Exception as e:
        return {
            "error": "SLM failed",
            "stdout": "",
            "stderr": str(e),
        }