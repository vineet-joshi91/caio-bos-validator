# api_server.py — drop-in replacement
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
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

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore

try:
    import docx  # python-docx
except Exception:
    docx = None  # type: ignore

try:
    import openpyxl
except Exception:
    openpyxl = None  # type: ignore

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

# Wallet + Payments routers MUST be included first
app.include_router(wallet_router)
app.include_router(razorpay_webhook_router)
app.include_router(bos_auth_router, tags=["bos-auth"])

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

def _extract_text_from_upload(filename: str, data: bytes) -> str:
    """
    Extract usable text from common file types.
    Never raises UnicodeDecodeError.
    """
    name = (filename or "").lower().strip()

    # PDF
    if name.endswith(".pdf"):
        if PdfReader is None:
            return ""
        try:
            reader = PdfReader(io.BytesIO(data))
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
            return "\\n\\n".join(parts).strip()
        except Exception:
            return ""

    # DOCX
    if name.endswith(".docx"):
        if docx is None:
            return ""
        try:
            d = docx.Document(io.BytesIO(data))
            parts = [p.text for p in d.paragraphs if p.text]
            return "\\n".join(parts).strip()
        except Exception:
            return ""

    # Plain text-like
    if name.endswith((".txt", ".md", ".csv", ".json", ".yaml", ".yml")):
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("utf-8", errors="ignore")
    # Excel file
    if name.endswith(".xlsx"):
        if openpyxl is None:
            return ""
        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
            parts = []
            for sheet in wb.worksheets:
                parts.append(f"[Sheet: {sheet.title}]")
                for row in sheet.iter_rows(values_only=True):
                    # join non-empty cells
                    cells = [str(c) for c in row if c is not None and str(c).strip() != ""]
                    if cells:
                        parts.append(" | ".join(cells))
            return "\n".join(parts).strip()
        except Exception:
            return ""

    # Fallback: never crash on binary
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

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
            detail="Decision Review requires findings/insights or document_text. Upload a file (Analyze) or provide a populated validator packet."
        )

    # Charge credits (if configured)
    try:
        charge_bos_run(payload.user_id, payload.plan_tier)
    except InsufficientCreditsError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception:
        # If charging fails, still allow run; you can tighten later
        pass

    # Save packet to temp file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as tf:
        json.dump(payload.packet, tf, ensure_ascii=False, indent=2)
        tmp_in = tf.name

    out = run_slm(
        tmp_in,
        "ea",
        model=payload.model,
        timeout_sec=payload.timeout_sec,
        num_predict=payload.num_predict,
    )
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

@app.post("/upload-and-ea")
async def upload_and_ea(
    file: UploadFile = File(...),
    timeout_sec: int = 300,
    num_predict: int = 512,
    model: Optional[str] = None,
):
    """
    Upload a file and run EA.

    - If file is JSON: treat as BOS/validator packet JSON (existing behavior).
    - Else: extract readable text and wrap it into a packet JSON for EA.

    Returns: {"ui": ...} compatible with BOSSummary.
    """
    filename = file.filename or "upload"
    raw = await file.read()

    # JSON packet path (backward compatible)
    if filename.lower().endswith(".json"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tf:
            tf.write(raw)
            tmp_in = tf.name
        out = run_slm(
            tmp_in,
            "ea",
            model=model,
            timeout_sec=timeout_sec,
            num_predict=num_predict,
        )
        if "error" in out and "ui" not in out:
            return {"ui": out}
        return {"ui": out.get("ui") or out}

    # Document path (PDF/DOCX/TXT/other)
    text = _extract_text_from_upload(filename, raw)

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

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as tf:
        json.dump(packet, tf, ensure_ascii=False, indent=2)
        tmp_in = tf.name

    out = run_slm(
        tmp_in,
        "ea",
        model=model,
        timeout_sec=timeout_sec,
        num_predict=num_predict,
    )
    if "error" in out and "ui" not in out:
        return {"ui": out}
    return {"ui": out.get("ui") or out}
