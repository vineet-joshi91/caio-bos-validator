# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 12:18:03 2025

@author: Vineet
"""

# api.py
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict
from slm.brains.ea_slm import run as run_ea
from slm.postprocess.normalize import to_ui_payload
from wallet import router as wallet_router

app = FastAPI(title="CAIO SLM API")

# adjust for your MVP origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wallet_router, prefix="/api", tags=["wallet"])

@app.post("/run-ea")
def run_ea_endpoint(packet: Dict[str, Any] = Body(...)):
    """
    Body should be the same dict you pass to run_slm.py (validator packet with findings/insights).
    """
    # run EA SLM (uses your models.yaml and current OllamaRunner underneath)
    ea_raw = run_ea(packet)
    # normalize for UI
    ui_payload = to_ui_payload(ea_raw)
    return {"ok": True, "ui": ui_payload, "raw": ea_raw}

@app.get("/health")
def health():
    return {"ok": True}
