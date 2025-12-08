# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

__all__ = [
    "OllamaRunner",
    "run_brain",
    "build_brain_prompt",
    "call_ollama",
    "PROMPT_SYSTEM",
]

PROMPT_SYSTEM = (
    "You are a compact Strategic CXO reasoning engine. "
    "Respond ONLY with strict JSON that matches the requested schema. "
    "Do not include code fences or extra commentary."
)
# -----------------------------
# Low-level Ollama HTTP runner
# -----------------------------
@dataclass
class OllamaRunner:
    model: str
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    timeout_sec: int = 120
    num_predict: int = 256

    def infer(self, prompt: str, system: str = "") -> str:
        """
        Call Ollama /api/generate and return the 'response' text.
        Ensures prompt is a string, and format is inside options.
        """
        url = f"{self.host}/api/generate"
        print(f"[SLM] Calling Ollama model='{self.model}' at {self.host} (timeout {self.timeout_sec}s)")
    
        # 🔒 Enforce prompt as string (Ollama requires this)
        if not isinstance(prompt, str):
            try:
                prompt = json.dumps(prompt, ensure_ascii=False)
            except Exception:
                prompt = str(prompt)
    
        # same for system (it must be a string too)
        if not isinstance(system, str):
            system = str(system)
    
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "num_predict": self.num_predict,
                "format": "json"
            },
            "stream": False
        }
    
        try:
            r = requests.post(url, json=payload, timeout=self.timeout_sec)
            r.raise_for_status()
        except requests.exceptions.ReadTimeout:
            raise RuntimeError(f"[SLM] Ollama timed out after {self.timeout_sec}s.")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"[SLM] Could not reach Ollama at {self.host}. {e}")
        except requests.HTTPError as e:
            status = getattr(r, "status_code", None)
            body = getattr(r, "text", "")
            raise RuntimeError(f"[SLM] Ollama HTTP {status}: {body}") from e
    
        data = r.json()
        txt = data.get("response", "")
        print(f"[SLM] Ollama responded, bytes: {len(txt)}")
        return txt


# -----------------------------
# Robust JSON helpers
# -----------------------------
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def _strip_fences(s: str) -> str:
    # remove leading/trailing ``` and ```json
    return _CODE_FENCE_RE.sub("", s).strip()

def ensure_json_dict(raw: str) -> Dict[str, Any]:
    """
    Accept raw model text; strip code fences; return a dict.
    If parsing fails, return a safe default envelope.
    """
    if not isinstance(raw, str):
        return {"error": "non-string model output"}
    txt = _strip_fences(raw)
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
        return {"error": "model returned non-dict JSON", "raw": obj}
    except Exception as e:
        # last-chance: try to salvage the first {...} block
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                frag = txt[start : end + 1]
                obj2 = json.loads(frag)
                if isinstance(obj2, dict):
                    return obj2
        except Exception:
            pass
        return {"error": f"non-JSON output ({type(e).__name__})", "raw": raw[:2000]}


# -----------------------------
# Prompting
# -----------------------------
def build_brain_prompt(pkt: Dict[str, Any], brain: str) -> str:
    findings = pkt.get("findings", []) or []
    insights = (pkt.get("insights", {}) or {}).get(brain, []) or []
    bos_index = float(pkt.get("bos_index", 0.0) or 0.0)

    snippet = {
        "brain": brain.upper(),
        "bos_index": bos_index,
        "insights": insights[:6],
        "top_findings": [
            {
                "rule_id": f.get("rule_id"),
                "severity": f.get("severity"),
                "title": f.get("title"),
                "message": (f.get("message") or "")[:180],
            }
            for f in findings[:12]
            if str(f.get("rule_id", "")).lower().startswith(brain.lower())
        ],
    }

    schema_hint = {
        "plan": {
            "assumptions": [],
            "priorities": [],
            "queries_to_run": [],
            "data_gaps": [],
        },
        "recommendation": {
            "summary": "",
            "actions_7d": [],
            "actions_30d": [],
            "kpis_to_watch": [],
            "risks": [],
            "forecast_note": "",
        },
        "confidence": 0.8,
        "tools_used": [],
        "tools": {"metrics": {}, "needs": []},
    }

    prompt = (
        "You are a compact Strategic CXO reasoning engine. "
        "Respond ONLY with strict JSON matching the requested schema.\n\n"
        "DATA:\n" + json.dumps(snippet, ensure_ascii=False) + "\n\n"
        "SCHEMA:\n" + json.dumps(schema_hint, ensure_ascii=False) + "\n\n"
        "RULES:\n"
        "- Keep it specific, concise, and practical.\n"
        "- Do not add fields not present in SCHEMA.\n"
        "- Use arrays for actions/kpis (no long prose strings).\n"
        "- Confidence 0.0..1.0.\n"
        "- Unknown metric => null.\n"
        "Return ONLY the JSON."
    )
    return prompt


# -----------------------------
# Legacy shim for old brain code
# -----------------------------
def call_ollama(
    *args,
    **kwargs,
) -> str:
    """
    Backward/defensive shim to invoke Ollama regardless of how the caller
    ordered the positional args.

    Supported call shapes we've seen in this project:
      1) host, model, timeout_sec, num_predict, temperature, top_p, repeat_penalty, prompt, *, system=...
      2) prompt, host, model, timeout_sec, num_predict, temperature, top_p, repeat_penalty, *, system=...
    And variants where timeout_sec/num_predict accidentally receive the system prompt.

    This function normalizes everything before constructing OllamaRunner.
    """
    # Defaults (in case something is missing)
    host = "http://127.0.0.1:11434"
    model = "qwen2.5:1.5b-instruct"
    timeout_sec = 300
    num_predict = 512
    temperature = 0.2
    top_p = 0.9
    repeat_penalty = 1.05
    prompt = ""
    system = kwargs.pop("system", "")

    def _as_int(x, default):
        # Accept int, float, digit-string; otherwise return default
        if isinstance(x, bool):
            return default
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            if s.isdigit():
                return int(s)
        return default

    # First, try to pull any named kwargs we recognize
    # (lets callers use keyword style without pos args order worries)
    host = kwargs.pop("host", host)
    model = kwargs.pop("model", model)
    timeout_sec = _as_int(kwargs.pop("timeout_sec", timeout_sec), timeout_sec)
    num_predict = _as_int(kwargs.pop("num_predict", num_predict), num_predict)
    temperature = float(kwargs.pop("temperature", temperature))
    top_p = float(kwargs.pop("top_p", top_p))
    repeat_penalty = float(kwargs.pop("repeat_penalty", repeat_penalty))
    prompt = kwargs.pop("prompt", prompt)

    # If args are present, we need to map them. We detect whether the first arg
    # looks like a host (http://...) or is actually the prompt body.
    # Known correct order (host-first):
    #   0 host, 1 model, 2 timeout, 3 num_predict, 4 temperature, 5 top_p, 6 repeat_penalty, 7 prompt
    #
    # Prompt-first order we’ve seen:
    #   0 prompt, 1 host, 2 model, 3 timeout, 4 num_predict, 5 temperature, 6 top_p, 7 repeat_penalty
    pos = list(args)

    def _looks_like_host(x: str) -> bool:
        return isinstance(x, str) and (x.startswith("http://") or x.startswith("https://"))

    if pos:
        if _looks_like_host(pos[0]):
            # host-first shape
            if len(pos) >= 1: host = pos[0]
            if len(pos) >= 2: model = pos[1]
            if len(pos) >= 3: timeout_sec = _as_int(pos[2], timeout_sec)
            if len(pos) >= 4: num_predict = _as_int(pos[3], num_predict)
            if len(pos) >= 5: temperature = float(pos[4])
            if len(pos) >= 6: top_p = float(pos[5])
            if len(pos) >= 7: repeat_penalty = float(pos[6])
            if len(pos) >= 8: prompt = pos[7]
        else:
            # prompt-first shape
            if len(pos) >= 1: prompt = pos[0]
            if len(pos) >= 2: host = pos[1]
            if len(pos) >= 3: model = pos[2]
            if len(pos) >= 4: timeout_sec = _as_int(pos[3], timeout_sec)
            if len(pos) >= 5: num_predict = _as_int(pos[4], num_predict)
            if len(pos) >= 6: temperature = float(pos[5])
            if len(pos) >= 7: top_p = float(pos[6])
            if len(pos) >= 8: repeat_penalty = float(pos[7])

    # Safety: if timeout_sec somehow got a long string (prompt), coerce back to default
    timeout_sec = _as_int(timeout_sec, 300)
    num_predict = _as_int(num_predict, 512)

    runner = OllamaRunner(
        model=model,
        host=host,
        timeout_sec=timeout_sec,
        num_predict=num_predict,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    return runner.infer(prompt=prompt, system=system)



# -----------------------------
# Public entry for each brain
# -----------------------------
def run_brain(
    pkt: Dict[str, Any],
    brain: str,
    *,
    model: str,
    host: str,
    timeout_sec: int = 120,
    num_predict: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repeat_penalty: float = 1.05,
) -> Dict[str, Any]:
    runner = OllamaRunner(
        model=model,
        host=host,
        timeout_sec=timeout_sec,
        num_predict=num_predict,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    raw = runner.infer(prompt=build_brain_prompt(pkt, brain), system=PROMPT_SYSTEM)
    obj = ensure_json_dict(raw)

    # guarantee minimal shape for the UI and EA layer
    if "plan" not in obj:
        obj["plan"] = {"assumptions": [], "priorities": [], "queries_to_run": [], "data_gaps": []}
    if "recommendation" not in obj:
        obj["recommendation"] = {
            "summary": obj.get("error", "") or "",
            "actions_7d": [],
            "actions_30d": [],
            "kpis_to_watch": [],
            "risks": [],
            "forecast_note": "",
        }
    if "confidence" not in obj:
        obj["confidence"] = 0.8
    if "_meta" not in obj:
        obj["_meta"] = {"model": model, "engine": "ollama", "confidence": obj.get("confidence", 0.8)}

    return obj
