# -*- coding: utf-8 -*-
"""
CPO SLM module (Chief People Officer).

- Builds CPO prompt and calls the model.
- Normalises recommendation structure so it matches other brains
  (actions_7d, actions_30d, actions_quarter, actions_half_year, actions_year, etc.).
- Prepares a tools["charts"] container for future CPO-specific visuals
  (agency funnels, bench vs billable, external vs internal mix), but does not
  add any charts yet. Cross-brain budget/profit comparisons are handled in EA.
"""

import json
from typing import Dict, Any

from slm.core.slm_core import build_brain_prompt, call_ollama, PROMPT_SYSTEM
from slm.tools.common import ensure_recommendation_shape


def run(
    packet: Dict[str, Any],
    host: str,
    model: str,
    timeout_sec: int,
    num_predict: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
) -> Dict[str, Any]:
    """
    CPO SLM wrapper.

    - Builds CPO prompt from BOS packet.
    - Calls backend model via call_ollama.
    - Parses JSON / falls back to a safe structure if needed.
    - Normalises recommendation structure so EA and the frontend
      can rely on a consistent schema across all brains.
    - Ensures tools["charts"] exists for future CPO visuals,
      but does not attach any charts yet (budget/profit comparisons
      will be handled in EA brain components).
    """
    # Build the CPO prompt from the BOS packet
    prompt = build_brain_prompt(packet, "cpo")

    # Call the underlying model through the SLM core
    resp_text = call_ollama(
        host,
        model,
        prompt,
        timeout_sec,
        num_predict,
        temperature,
        top_p,
        repeat_penalty,
        system=PROMPT_SYSTEM,
    )

    # Try to parse JSON from the model
    try:
        obj = json.loads(resp_text)
    except Exception:
        # Fallback if the model does not return valid JSON
        obj = {
            "plan": {
                "assumptions": [],
                "priorities": [],
                "queries_to_run": [],
                "data_gaps": [],
            },
            "recommendation": {
                "summary": "Unstructured output",
                "actions_7d": [],
                "actions_30d": [],
                "kpis_to_watch": [],
                "risks": [],
                "forecast_note": "",
            },
            "confidence": 0.5,
            "_meta": {"model": model, "engine": "ollama", "confidence": 0.5},
            "raw_text": resp_text,
        }

    # Ensure metadata exists / is populated
    obj.setdefault(
        "_meta",
        {
            "model": model,
            "engine": "ollama",
            "confidence": obj.get("confidence", 0.7),
        },
    )

    # Normalise recommendation structure so it always has:
    #  - summary
    #  - actions_7d, actions_30d, actions_quarter,
    #    actions_half_year, actions_year
    #  - kpis_to_watch, risks, forecast_note
    ensure_recommendation_shape(obj)

    # Prepare tools["charts"] so frontend and EA code can rely on its presence.
    # We are NOT adding CPO charts here yet; those will be defined later
    # when you decide the exact visuals for agency/contract/BPO/bench views.
    tools: Dict[str, Any] = obj.setdefault("tools", {})
    tools.setdefault("charts", [])

    return obj
