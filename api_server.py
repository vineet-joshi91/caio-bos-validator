# api_server.py — drop-in replacement
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json, subprocess, sys, tempfile, shutil, os
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from wallet_api import router as wallet_router
from webhooks_razorpay import router as razorpay_webhook_router


from wallet import (
    consume_credits_and_record_usage,
    InsufficientCreditsError,
    DailyLimitReachedError,
)
from .db import get_db
from .db import SessionLocal
from tier_config import TIER_CONFIG
from bos_credits import charge_bos_run

app = FastAPI(title="CAIO BOS – EA API")

# Wallet + Payments routers MUST be included first
app.include_router(wallet_router)
app.include_router(razorpay_webhook_router)

# -------------------- Models --------------------
class EARequest(BaseModel):
    packet: dict
    user_id: int
    plan_tier: str = "demo"  # 'demo','pro','premium','enterprise'
    model: Optional[str] = None
    timeout_sec: int = 300
    num_predict: int = 512


class BrainRequest(EARequest):
    brain: str  # one of: cfo,cmo,coo,chro,cpo,ea



# -------------------- Utils --------------------
def repo_root() -> Path:
    """
    Walk upwards until we find slm/run_slm.py; fallback to current dir.
    """
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        if (cur / "slm" / "run_slm.py").exists():
            return cur
        cur = cur.parent
    return Path(__file__).resolve().parent


def _last_top_level_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the last top-level JSON object by bracket counting.
    Works even if the model prints extra tokens before/after JSON.
    """
    start = -1
    depth = 0
    last_obj: Optional[Dict[str, Any]] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start : i + 1]
                    try:
                        last_obj = json.loads(candidate)
                    except Exception:
                        # ignore malformed candidates, keep scanning
                        pass
    return last_obj


def parse_json_loose(txt: str) -> Dict[str, Any]:
    """
    Try strict JSON first, then fallback to bracket-scan.
    If nothing parses, return a structured error so the UI can show raw logs.
    """
    # 1) strict
    try:
        return json.loads(txt)
    except Exception:
        pass

    # 2) bracket-scan
    obj = _last_top_level_json(txt)
    if obj is not None:
        return obj

    # 3) structured error
    return {
        "error": "Model response was not valid JSON",
        "raw": (txt or "")[:4000],
    }


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
        "--config",
        "slm/config/models.yaml",
        "--timeout",
        str(int(timeout_sec)),
        "--num_predict",
        str(int(num_predict)),
    ]
    if model:
        cmd += ["--model", model]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=int(timeout_sec) + 30,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"SLM timed out after {timeout_sec}s",
            "stdout": "",
            "stderr": "",
            "cmd": " ".join(cmd),
        }

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        # Make errors easy to see in diagnostics / frontend
        return {
            "error": "SLM failed",
            "stdout": stdout,
            "stderr": stderr or "(empty stderr)",
            "cmd": " ".join(cmd),
        }

    if not stdout:
        return {
            "error": "SLM produced no output",
            "stdout": "",
            "stderr": stderr or "(empty stderr)",
            "cmd": " ".join(cmd),
        }

    return parse_json_loose(stdout)


# -------------------- CORS --------------------
# Default to your known frontend / site origins.
DEFAULT_ORIGINS = [
    "https://caioinsights.com",
    "https://www.caioinsights.com",
    "https://caio-frontend.vercel.app",
    "https://caioai.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

env_origins = os.getenv("CORS_ALLOW_ORIGINS")
if env_origins:
    allow_list = [o.strip() for o in env_origins.split(",") if o.strip()]
else:
    allow_list = DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_list or ["*"],
    allow_credentials=False,  # no cookies / auth needed for this API
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Endpoints --------------------
@app.get("/")
def root():
    return {"message": "CAIO BOS – EA API root", "health": "ok"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/welcome")
def welcome():
    return {"message": "CAIO BOS API is up"}


@app.post("/run-ea")
def run_ea(req: EARequest, db: Session = Depends(get_db)):
    # 1) Run BOS credit gate
    credits_used = charge_bos_run(
        db=db,
        user_id=req.user_id,
        plan_tier=req.plan_tier,
        brain="ea",
        doc_increment=1,
    )
    # 2) Commit the credit changes / usage log
    db.commit()

    # 3) Proceed with EA analysis as before
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tf:
        json.dump(req.packet, tf, ensure_ascii=False, indent=2)
        tmp_in = tf.name

    out = run_slm(
        tmp_in,
        "ea",
        model=req.model,
        timeout_sec=req.timeout_sec,
        num_predict=req.num_predict,
    )

    if "error" in out and "ui" not in out:
        return {"ui": out}
    return {"ui": out.get("ui") or out}


@app.post("/run-brain")
def run_brain(req: BrainRequest, db: Session = Depends(get_db)):
    brain = req.brain.lower().strip()
    if brain not in {"cfo", "cmo", "coo", "chro", "cpo", "ea"}:
        raise HTTPException(status_code=400, detail="Invalid brain")

    # 1) BOS credit gate
    credits_used = charge_bos_run(
        db=db,
        user_id=req.user_id,
        plan_tier=req.plan_tier,
        brain=brain,
        doc_increment=1,
    )
    db.commit()

    # 2) Now run the actual brain
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tf:
        json.dump(req.packet, tf, ensure_ascii=False, indent=2)
        tmp_in = tf.name

    out = run_slm(
        tmp_in,
        brain,
        model=req.model,
        timeout_sec=req.timeout_sec,
        num_predict=req.num_predict,
    )
    return out

@app.post("/upload-and-ea")
async def upload_and_ea(
    file: UploadFile = File(...),
    timeout_sec: int = 300,
    num_predict: int = 512,
    model: Optional[str] = None,
):
    """
    Convenience path for your website: user uploads a validator packet JSON,
    we run EA on it and return the UI block.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="wb"
    ) as tf:
        shutil.copyfileobj(file.file, tf)
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
