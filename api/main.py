# api/main.py
# -*- coding: utf-8 -*-
import json, os, sys, time
from typing import Any, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, RootModel

# --- import slm package safely ---
ROOT = Path(__file__).resolve().parents[1]
SLM_DIR = ROOT / "slm"
sys.path.extend([str(ROOT), str(SLM_DIR)])

from slm.run_slm import _ensure_meta, _to_jsonable  # you already have these
from slm.config import load_config, get_brain_effective
from slm.brains import cfo_slm, cmo_slm, coo_slm, chro_slm, cpo_slm, ea_slm

class Packet(RootModel[Dict[str, Any]]): pass

class Overrides(BaseModel):
    model: Optional[str] = None
    timeout_sec: Optional[int] = None
    num_predict: Optional[int] = None

class RunRequest(BaseModel):
    packet: Packet
    overrides: Optional[Overrides] = None

app = FastAPI(title="CAIO BOS – EA API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

CONFIG_PATH = os.getenv("SLM_CONFIG", str(SLM_DIR / "config" / "models.yaml"))
CFG = load_config(CONFIG_PATH)
BRAINS = ["cfo", "cmo", "coo", "chro", "cpo"]

def _eff(brain: str, ov: Optional[Overrides]) -> Dict[str, Any]:
    e = get_brain_effective(CFG, brain)
    if ov:
        if ov.model: e["model"] = ov.model
        if ov.timeout_sec: e["timeout_sec"] = int(ov.timeout_sec)
        if ov.num_predict: e["num_predict"] = int(ov.num_predict)
    return e

def _run_brain(name: str, pkt: Dict[str, Any], eff: Dict[str, Any]) -> Dict[str, Any]:
    fn = {
        "cfo": cfo_slm.run, "cmo": cmo_slm.run, "coo": coo_slm.run,
        "chro": chro_slm.run, "cpo": cpo_slm.run
    }[name]
    out = fn(
        pkt,
        host=eff["host"], model=eff["model"],
        timeout_sec=eff["timeout_sec"], num_predict=eff["num_predict"],
        temperature=eff["temperature"], top_p=eff["top_p"], repeat_penalty=eff["repeat_penalty"],
    )
    return _ensure_meta(_to_jsonable(out), eff)

@app.get("/health")
def health():
    return {"ok": True, "config": CONFIG_PATH}

@app.post("/run-ea")
def run_ea(req: RunRequest):
    t0 = time.time()
    pkt = req.packet.root
    body_len = len(json.dumps(pkt))
    if body_len > 1_200_000:
        raise HTTPException(413, "Packet too large")

    # parallel CXO passes
    per_brain: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(BRAINS)) as ex:
        futs = {ex.submit(_run_brain, b, pkt, _eff(b, req.overrides)): b for b in BRAINS}
        for fut in as_completed(futs):
            b = futs[fut]
            per_brain[b] = fut.result()

    # EA aggregation
    ea_eff = _eff("ea", req.overrides)
    ea_json = ea_slm.run(
        pkt, per_brain=per_brain,
        host=ea_eff["host"], model=ea_eff["model"],
        timeout_sec=ea_eff["timeout_sec"],
        num_predict=max(int(ea_eff["num_predict"]), 384),
        temperature=ea_eff["temperature"], top_p=ea_eff["top_p"],
        repeat_penalty=ea_eff["repeat_penalty"],
    )

    # normalize
    if ea_json is None: ea_json = {"executive_summary": "EA returned no content."}
    elif isinstance(ea_json, str):
        s = ea_json.strip()
        try: ea_json = json.loads(s) if s else {"executive_summary": "EA returned empty output."}
        except Exception: ea_json = {"executive_summary": s}
    elif not isinstance(ea_json, dict):
        ea_json = {"executive_summary": str(ea_json)}

    ea_json = _ensure_meta(_to_jsonable(ea_json), ea_eff)
    ea_json.setdefault("cross_brain_actions_7d", [])
    ea_json.setdefault("cross_brain_actions_30d", [])
    ea_json.setdefault("top_priorities", [])
    ea_json.setdefault("key_risks", [])
    ea_json.setdefault("owner_matrix", {})
    ea_json["_meta"]["elapsed_sec"] = round(time.time() - t0, 2)

    # Return the bundle contract your UI already handles
    return {"ui": ea_json, "per_brain": per_brain}
