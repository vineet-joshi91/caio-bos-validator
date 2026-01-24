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

import csv

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image
except Exception:
    Image = None


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
    Extract usable text from common file types with fallbacks.
    - PDFs: pypdf -> pdfplumber -> OCR (optional) if still too short
    - DOCX: paragraphs + tables
    - XLSX: all sheets, row-wise
    - CSV/TSV: delimiter sniff + robust decoding
    - Images: OCR (optional)
    Always returns a string; never raises UnicodeDecodeError.
    """
    name = (filename or "").lower().strip()

    # --- Safety caps ---
    MAX_TEXT_CHARS = 250_000      # hard cap to keep packets sane
    PDF_MIN_TEXT_CHARS = 3000     # below this, try fallback (plumber/OCR)
    MAX_PDF_PAGES_OCR = 12        # OCR can be expensive; cap pages
    MAX_XLSX_ROWS_PER_SHEET = 2000
    MAX_XLSX_CELLS_PER_ROW = 50

    def _cap(s: str) -> str:
        s = (s or "").strip()
        return s[:MAX_TEXT_CHARS]

    def _decode_bytes(b: bytes) -> str:
        # Try common encodings. utf-8-sig handles BOM.
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return b.decode(enc)
            except Exception:
                continue
        return b.decode("utf-8", errors="ignore")

    def _is_probably_binary(b: bytes) -> bool:
        # Heuristic: lots of NUL bytes suggests binary
        return b.count(b"\x00") > 10

    def _extract_pdf_pypdf(b: bytes) -> str:
        if PdfReader is None:
            return ""
        try:
            reader = PdfReader(io.BytesIO(b))
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                t = t.strip()
                if t:
                    parts.append(t)
            return "\n\n".join(parts).strip()
        except Exception:
            return ""

    def _extract_pdf_pdfplumber(b: bytes) -> str:
        if pdfplumber is None:
            return ""
        try:
            parts = []
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    t = t.strip()
                    if t:
                        parts.append(t)
            return "\n\n".join(parts).strip()
        except Exception:
            return ""

    def _extract_image_ocr(img_bytes: bytes) -> str:
        if pytesseract is None or Image is None:
            return ""
        try:
            img = Image.open(io.BytesIO(img_bytes))
            # Basic OCR; keep it simple/reliable
            txt = pytesseract.image_to_string(img)
            return (txt or "").strip()
        except Exception:
            return ""

    def _extract_pdf_ocr(b: bytes) -> str:
        """
        OCR PDF by rendering pages to images via pdftoppm (poppler-utils)
        and running pytesseract. Uses a cap on pages.
        """
        if pytesseract is None or Image is None:
            return ""
        # If poppler isn't installed, this will fail; we catch below.
        import tempfile
        import subprocess
        import os
        try:
            with tempfile.TemporaryDirectory() as td:
                pdf_path = os.path.join(td, "in.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(b)

                # Render to PNGs: page-1.png, page-2.png, ...
                # -f 1 -l N caps pages
                out_prefix = os.path.join(td, "page")
                cmd = ["pdftoppm", "-png", "-f", "1", "-l", str(MAX_PDF_PAGES_OCR), pdf_path, out_prefix]
                subprocess.run(cmd, check=True, capture_output=True)

                parts = []
                # Collect rendered pages in order
                for i in range(1, MAX_PDF_PAGES_OCR + 1):
                    img_path = f"{out_prefix}-{i}.png"
                    if not os.path.exists(img_path):
                        break
                    with open(img_path, "rb") as imf:
                        img_bytes = imf.read()
                    t = _extract_image_ocr(img_bytes)
                    if t:
                        parts.append(f"[OCR Page {i}]\n{t}")
                return "\n\n".join(parts).strip()
        except Exception:
            return ""

    # -------------------------
    # PDF
    # -------------------------
    if name.endswith(".pdf"):
        # 1) pypdf
        t1 = _extract_pdf_pypdf(data)
        best = t1

        # 2) pdfplumber fallback
        if len(best) < PDF_MIN_TEXT_CHARS:
            t2 = _extract_pdf_pdfplumber(data)
            if len(t2) > len(best):
                best = t2

        # 3) OCR fallback (optional)
        if len(best) < PDF_MIN_TEXT_CHARS:
            t3 = _extract_pdf_ocr(data)
            if len(t3) > len(best):
                best = t3

        return _cap(best)

    # -------------------------
    # DOCX
    # -------------------------
    if name.endswith(".docx"):
        if docx is None:
            return ""
        try:
            d = docx.Document(io.BytesIO(data))
            parts = []

            # paragraphs
            for p in d.paragraphs:
                txt = (p.text or "").strip()
                if txt:
                    parts.append(txt)

            # tables
            for table in d.tables:
                for row in table.rows:
                    cells = []
                    for cell in row.cells:
                        ct = (cell.text or "").strip()
                        if ct:
                            cells.append(ct)
                    if cells:
                        parts.append(" | ".join(cells))

            return _cap("\n".join(parts))
        except Exception:
            return ""

    # -------------------------
    # Excel
    # -------------------------
    if name.endswith(".xlsx"):
        if openpyxl is None:
            return ""
        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
            parts = []
            for sheet in wb.worksheets:
                parts.append(f"[Sheet: {sheet.title}]")
                row_count = 0
                for row in sheet.iter_rows(values_only=True):
                    if row_count >= MAX_XLSX_ROWS_PER_SHEET:
                        parts.append("[TRUNCATED: too many rows]")
                        break
                    row_count += 1

                    cells = []
                    for c in row[:MAX_XLSX_CELLS_PER_ROW]:
                        if c is None:
                            continue
                        s = str(c).strip()
                        if s:
                            cells.append(s)
                    if cells:
                        parts.append(" | ".join(cells))
            return _cap("\n".join(parts))
        except Exception:
            return ""

    # -------------------------
    # CSV / TSV
    # -------------------------
    if name.endswith((".csv", ".tsv")):
        try:
            text = _decode_bytes(data)
            # Sniff delimiter (fallback to comma/tsv)
            sample = text[:4096]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                delim = dialect.delimiter
            except Exception:
                delim = "\t" if name.endswith(".tsv") else ","

            reader = csv.reader(io.StringIO(text), delimiter=delim)
            out_lines = []
            for i, row in enumerate(reader):
                if i > 3000:
                    out_lines.append("[TRUNCATED: too many rows]")
                    break
                # Keep rows bounded
                row = [cell.strip() for cell in row[:60] if cell and cell.strip()]
                if row:
                    out_lines.append(" | ".join(row))
            return _cap("\n".join(out_lines))
        except Exception:
            # Last resort decode
            return _cap(_decode_bytes(data))

    # -------------------------
    # Plain text-like
    # -------------------------
    if name.endswith((".txt", ".md", ".json", ".yaml", ".yml", ".log")):
        return _cap(_decode_bytes(data))

    # -------------------------
    # Images (OCR)
    # -------------------------
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        t = _extract_image_ocr(data)
        return _cap(t)

    # -------------------------
    # Fallback
    # -------------------------
    if _is_probably_binary(data):
        # Don't attempt to "decode" arbitrary binaries meaningfully
        return ""
    return _cap(_decode_bytes(data))


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
    
    print(
    f"[UPLOAD] filename={filename} "
    f"len={len(text)} "
    f"preview={text[:200]!r}"
)

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
        model=model,
        timeout_sec=timeout_sec,
        num_predict=num_predict,
    )
    if "error" in out and "ui" not in out:
        return {"ui": out}
    return {"ui": out.get("ui") or out}
