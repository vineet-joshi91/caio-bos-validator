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
    
# -------------------- Routes --------------------
@app.get("/")
def root():
    return {"ok": True, "service": "caio-bos"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/welcome")
def welcome():
    return {"ok": True, "message": "Welcome to CAIO BOS"}

@app.post("/run-ea")
def run_ea(payload: EARequest):
    # --- Guard: prevent empty Decision Review packets (avoid timeouts / fluff) ---
    pkt = payload.packet or {}

    findings = pkt.get("findings") or []
    insights_map = pkt.get("insights") or {}
    document_text = (pkt.get("document_text") or pkt.get("text") or "").strip()

    has_insights = False
    if isinstance(insights_map, dict):
        for b in ["cfo", "cmo", "coo", "chro", "cpo", "ea"]:
            if insights_map.get(b):
                has_insights = True
                break
    else:
        has_insights = bool(insights_map)

    if (not findings) and (not has_insights) and (not document_text):
        raise HTTPException(
            status_code=400,
            detail=(
                "Decision Review requires findings/insights or document_text. "
                "Upload a file (Analyze) or provide a populated validator packet."
            ),
        )

    # --- Charge credits (if configured) ---
    try:
        charge_bos_run(payload.user_id, payload.plan_tier)
    except InsufficientCreditsError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception:
        # If charging fails, still allow run; tighten later if you want
        pass

    # --- Save packet to temp file ---
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tf:
        json.dump(payload.packet, tf, ensure_ascii=False, indent=2)
        tmp_in = tf.name

    # --- Single-model policy: Primary + Fallback ---
    primary_model = "qwen2.5:3b-instruct"
    fallback_model = "qwen2.5:1.5b-instruct"

    # Use request overrides if you ever want later; for now enforce primary
    # (keeps behavior stable and prevents accidental weak models)
    timeout_sec = payload.timeout_sec
    num_predict = payload.num_predict

    # --- Run primary ---
    out = run_slm(
        tmp_in,
        "ea",
        model=primary_model,
        timeout_sec=timeout_sec,
        num_predict=num_predict,
    )

    # Detect failure (either top-level error or ui.error)
    ui_obj = out.get("ui") if isinstance(out, dict) else None
    is_fail = (
        (not isinstance(out, dict))
        or (isinstance(out, dict) and out.get("error"))
        or (isinstance(ui_obj, dict) and ui_obj.get("error"))
    )

    # --- If primary failed, retry once with fallback ---
    if is_fail:
        out2 = run_slm(
            tmp_in,
            "ea",
            model=fallback_model,
            timeout_sec=timeout_sec,
            num_predict=num_predict,
        )
        out = out2

    # --- Final safety: never return None / invalid types ---
    if out is None:
        return {"ui": {"error": "Empty response from run_slm", "stdout": "", "stderr": ""}}

    if not isinstance(out, dict):
        return {"ui": {"error": "Invalid response type from run_slm", "stdout": "", "stderr": str(out)}}

    # If run_slm returned {"error": "..."} without ui wrapper, wrap it
    if "error" in out and "ui" not in out:
        return {"ui": out}

    return out




@app.post("/run-brain")
def run_brain(payload: BrainRequest):
    # Charge credits (if configured)
    try:
        charge_bos_run(payload.user_id, payload.plan_tier)
    except InsufficientCreditsError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as tf:
        json.dump(payload.packet, tf, ensure_ascii=False, indent=2)
        tmp_in = tf.name

    out = run_slm(
        tmp_in,
        payload.brain,
        model=payload.model,
        timeout_sec=payload.timeout_sec,
        num_predict=payload.num_predict,
    )
    if "error" in out and "ui" not in out:
        return {"ui": out}
    return out

from routes_bos_auth import get_current_user

@app.post("/upload-and-ea")
async def upload_and_ea(
    file: UploadFile = File(...),
    timeout_sec: int = 300,
    num_predict: int = 256,
    model: Optional[str] = None,
    current_user: User = Depends(get_current_user),  # ADD THIS - automatically validates JWT
    db: Session = Depends(get_db),  # ADD THIS - for database access
):
    """
    Upload a file and run EA.
    """
    # Get user info from authenticated user
    # Get user info
    user_id = current_user.id
    plan_tier = get_user_tier(current_user)
    filename = file.filename or "upload"
        
    # Log the upload
    log_user_action(
        current_user, 
        "upload", 
        {"filename": filename, "tier": plan_tier, "admin": current_user.is_admin}
    )
        
    raw = await file.read()
        
    # Charge credits before processing
    try:
        charge_or_pass(user_id=user_id, plan_tier=plan_tier, brain="ea")
    except HTTPException as e:
        if e.status_code == 402:
            logger.warning(f"❌ User {user_id} insufficient credits")
        raise
    
    model = None
            
    # JSON packet path (backward compatible)
    if filename.lower().endswith(".json"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tf:
            tf.write(raw)
            tmp_in = tf.name
        out = run_slm(
            tmp_in,
            "ea",
            model=None,
            timeout_sec=timeout_sec,
            num_predict=None,
        )
        if "error" in out and "ui" not in out:
            return {"ui": out}
        return {"ui": out.get("ui") or out}

    # Document path (PDF/DOCX/TXT/other)
    text, extract_meta = extract_text_with_meta(filename, raw)
    
    print(
    f"[UPLOAD] filename={filename} "
    f"len={len(text)} "
    f"preview={text[:200]!r}"
)

    print(f"[EXTRACT] chosen={extract_meta.get('chosen_method')} flags={extract_meta.get('quality_flags')}")

    if not text or len(text.strip()) < 20:
        return {
            "ui": {
                "error": "No readable text extracted from upload",
                "stdout": "",
                "stderr": "",
            }
        }

    # Wrap extracted text into a packet JSON for EA
    packet: Dict[str, Any] = {
        "label": "Uploaded Document",
        "source": {
            "filename": filename,
            "content_type": file.content_type,
            "size_bytes": len(raw),
        },
        "document_text": text[:200000],  # safety cap
        "facts": {},
        "meta": {"ingest": "upload-and-ea"},
    }
    
    packet["meta"]["doc_text_len"] = len(text)
    packet["meta"]["doc_text_preview"] = text[:400]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as tf:
        json.dump(packet, tf, ensure_ascii=False, indent=2)
        tmp_in = tf.name

    out = run_slm(
        tmp_in,
        "ea",
        model=None,
        timeout_sec=timeout_sec,
        num_predict=num_predict,
    )
    
    # If primary fails, retry once with fallback model
    ui_obj = out.get("ui") if isinstance(out, dict) else None
    is_fail = (
        (isinstance(out, dict) and out.get("error"))
        or (isinstance(ui_obj, dict) and ui_obj.get("error"))
    )
    if is_fail:
        out2 = run_slm(
            tmp_in,
            "ea",
            model=FALLBACK_EA_MODEL,
            timeout_sec=timeout_sec,
            num_predict=num_predict,
        )
        out = out2

    
    # Attach extraction metadata
    packet["meta"]["extract"] = extract_meta
    
    # Build warnings for UI
    warnings = []
    flags = extract_meta.get("quality_flags") or []
    if "LIKELY_QUOTE_PRICING_NOT_EXTRACTED" in flags:
        warnings.append(
            "Pricing/quotation terms may be embedded as an image/table and were not extracted reliably. "
            "Upload the quotation as XLSX/CSV or a text-based PDF, or upload the quotation pages separately."
        )
    elif "LOW_TEXT_PDF" in flags:
        warnings.append(
            "This PDF contains limited extractable text (possibly scanned or table-heavy). "
            "Results may be incomplete; consider uploading a text-based PDF or an XLSX/CSV version."
        )
    
    packet["meta"]["warnings"] = warnings
    
    if "error" in out and "ui" not in out:
        return {"ui": out}
    ui_obj = out.get("ui") or out
    if isinstance(ui_obj, dict):
        ui_obj.setdefault("warnings", [])
        ui_obj["warnings"].extend(packet["meta"].get("warnings", []))
        ui_obj["extract_meta"] = extract_meta
    return {"ui": ui_obj}