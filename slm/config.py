# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:57:44 2025

@author: Vineet
"""

# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class Defaults:
    engine: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    timeout_sec: int = 120
    num_predict: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    fallback_model: str = "qwen2.5:1.5b-instruct"


@dataclass
class BrainCfg:
    model_path: str
    timeout_sec: Optional[int] = None


@dataclass
class ModelConfig:
    defaults: Defaults
    brains: Dict[str, BrainCfg]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


def load_config(path: str | Path) -> ModelConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[SLM] models.yaml not found at {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    d = raw.get("defaults", {}) or {}
    defaults = Defaults(
        engine=d.get("engine", "ollama"),
        base_url=d.get("base_url", "http://127.0.0.1:11434"),
        timeout_sec=int(d.get("timeout_sec", 120)),
        num_predict=int(d.get("num_predict", 256)),
        temperature=float(d.get("temperature", 0.2)),
        top_p=float(d.get("top_p", 0.9)),
        repeat_penalty=float(d.get("repeat_penalty", 1.05)),
        fallback_model=d.get("fallback_model", "qwen2.5:1.5b-instruct"),
    )

    brains_raw = raw.get("brains", {}) or {}
    brains: Dict[str, BrainCfg] = {}
    for k, v in brains_raw.items():
        brains[k] = BrainCfg(
            model_path=v.get("model_path"),
            timeout_sec=v.get("timeout_sec"),
        )

    return ModelConfig(defaults=defaults, brains=brains)


def get_brain_effective(cfg: ModelConfig, brain: str) -> dict:
    """
    Returns a flat dict of effective settings for a given brain,
    merging defaults with per-brain overrides.
    """
    b = cfg.brains.get(brain)
    if not b:
        raise KeyError(f"[SLM] No brain named '{brain}' in models.yaml")
    eff = {
        "model": b.model_path or cfg.defaults.fallback_model,
        "host": cfg.defaults.base_url,
        "timeout_sec": int(b.timeout_sec or cfg.defaults.timeout_sec),
        "num_predict": int(cfg.defaults.num_predict),
        "temperature": float(cfg.defaults.temperature),
        "top_p": float(cfg.defaults.top_p),
        "repeat_penalty": float(cfg.defaults.repeat_penalty),
    }
    return eff
