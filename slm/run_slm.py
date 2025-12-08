# -*- coding: utf-8 -*-
import argparse, json, sys, os
from pathlib import Path
from typing import Dict, Any

# ---------------------------------------------------------------------
# Package-safe imports: works both as module and direct script
# ---------------------------------------------------------------------
if __package__ in (None, ""):
    PKG_DIR = Path(__file__).resolve().parent            # .../slm
    ROOT_DIR = PKG_DIR.parent                            # project root
    sys.path.append(str(PKG_DIR))                        # import "brains", "config"
    sys.path.append(str(ROOT_DIR))                       # import "slm" if needed

    from brains import cfo_slm, cmo_slm, coo_slm, chro_slm, cpo_slm, ea_slm
    from config import load_config, get_brain_effective
else:
    from .brains import cfo_slm, cmo_slm, coo_slm, chro_slm, cpo_slm, ea_slm
    from .config import load_config, get_brain_effective
# ---------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "slm/config/models.yaml"

def _print_json(obj: Any):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def _read_json(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)

def _to_jsonable(x):
    if hasattr(x, "model_dump_json"):  # pydantic v2
        return json.loads(x.model_dump_json())
    if hasattr(x, "dict"):             # pydantic v1
        return x.dict()
    if hasattr(x, "__dict__"):
        return {k: v for k, v in x.__dict__.items() if not k.startswith("_")}
    return x

def _ensure_meta(obj: Dict[str, Any], eff: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"ui": {"executive_summary": str(obj), "_meta": {"model": eff.get("model"), "engine": "ollama", "confidence": 0.0}}}
    meta = obj.setdefault("_meta", {})
    meta.setdefault("model", eff.get("model"))
    meta.setdefault("engine", "ollama")
    # float confidence if present
    try:
        meta.setdefault("confidence", float(obj.get("confidence", meta.get("confidence", 0.0))))
    except Exception:
        meta.setdefault("confidence", 0.0)
    return obj

def _call_brain(name: str, pkt: dict, eff: dict):
    fn_map = {
        "cfo": cfo_slm.run,
        "cmo": cmo_slm.run,
        "coo": coo_slm.run,
        "chro": chro_slm.run,
        "cpo": cpo_slm.run,
    }
    fn = fn_map[name]
    out = fn(
        pkt,
        host=eff["host"],
        model=eff["model"],
        timeout_sec=eff["timeout_sec"],
        num_predict=eff["num_predict"],
        temperature=eff["temperature"],
        top_p=eff["top_p"],
        repeat_penalty=eff["repeat_penalty"],
    )
    return _to_jsonable(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Phase 2.5 packet (e.g., bad_with_insights.json)")
    p.add_argument("--brain", required=True, choices=["cfo", "cmo", "coo", "chro", "cpo", "all", "ea"])
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to models.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--num_predict", type=int, default=None)
    args = p.parse_args()

    try:
        cfg = load_config(args.config)
        pkt = _read_json(Path(args.input))

        def eff_for(brain: str) -> dict:
            e = get_brain_effective(cfg, brain)
            if args.model:       e["model"] = args.model
            if args.timeout:     e["timeout_sec"] = int(args.timeout)
            if args.num_predict: e["num_predict"] = int(args.num_predict)
            return e

        # Single brain
        if args.brain in {"cfo", "cmo", "coo", "chro", "cpo"}:
            e = eff_for(args.brain)
            out = _ensure_meta(_call_brain(args.brain, pkt, e), e)
            _print_json(out)
            return

        # All brains
        if args.brain == "all":
            results = {}
            for b in ["cfo", "cmo", "coo", "chro", "cpo"]:
                e = eff_for(b)
                results[b] = _ensure_meta(_call_brain(b, pkt, e), e)
            _print_json(results)
            return

        # Executive Assistant aggregator (parallelize per-brain runs)
        if args.brain == "ea":
            from concurrent.futures import ThreadPoolExecutor, as_completed

            brain_list = ["cfo", "cmo", "coo", "chro", "cpo"]

            def run_one(brain_name: str):
                e = eff_for(brain_name)
                out = _ensure_meta(_call_brain(brain_name, pkt, e), e)
                return brain_name, out

            per_brain = {}
            max_workers = min(len(brain_list), max(2, (os.cpu_count() or 4)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(run_one, b): b for b in brain_list}
                for fut in as_completed(futures):
                    b, out = fut.result()
                    per_brain[b] = out

            ea_e = eff_for("ea")
            ea_json = ea_slm.run(
                pkt,
                per_brain=per_brain,
                host=ea_e["host"],
                model=ea_e["model"],
                timeout_sec=ea_e["timeout_sec"],
                num_predict=max(int(ea_e["num_predict"]), 256),
                temperature=ea_e["temperature"],
                top_p=ea_e["top_p"],
                repeat_penalty=ea_e["repeat_penalty"],
            )

            # normalize to dict
            if ea_json is None:
                ea_json = {"executive_summary": "EA returned no content."}
            elif isinstance(ea_json, str):
                msg = ea_json.strip()
                if msg:
                    try:
                        ea_json = json.loads(msg)
                    except Exception:
                        ea_json = {"executive_summary": msg}
                else:
                    ea_json = {"executive_summary": "EA returned empty output."}
            elif not isinstance(ea_json, dict):
                ea_json = {"executive_summary": str(ea_json)}

            ea_json = _ensure_meta(_to_jsonable(ea_json), ea_e)
            _print_json(ea_json)
            return

    except Exception as e:
        # Always emit JSON so the UI can show it in the EA Summary box
        err = {"ui": {"error": "SLM failed", "stdout": "", "stderr": f"{type(e).__name__}: {e}"}}
        _print_json(err)
        # Non-zero exit would bubble as HTTP 500 in your API subprocess runner;
        # keeping zero allows the diagnostics page to show the error content.
        return

if __name__ == "__main__":
    main()
