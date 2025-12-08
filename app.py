# -*- coding: utf-8 -*-
"""
CAIO BOS – Executive Assistant (EA) Demo
Local subprocess or API mode, robust JSON parsing + rich render
"""
import json, os, sys, subprocess, tempfile
from pathlib import Path
import requests
import streamlit as st

st.set_page_config(page_title="CAIO BOS – EA Demo", layout="wide")
st.title("CAIO BOS – Executive Assistant (EA) Demo")

# ----------------- Controls -----------------
mode = st.radio("Run mode", ["API (FastAPI server)", "Local (subprocess)"], horizontal=True, index=1)
api_base = st.text_input("API base URL (for API mode)", os.getenv("SLM_API", "http://127.0.0.1:8000"))
model_override = st.text_input("Optional model override (e.g., 'qwen2.5:1.5b-instruct')", "")
c1, c2 = st.columns(2)
with c1:
    timeout_sec = st.number_input("Timeout (sec)", min_value=30, max_value=900, value=300, step=30)
with c2:
    num_predict = st.number_input("num_predict", min_value=64, max_value=2048, value=512, step=64)

st.markdown("### Input packet")
tab1, tab2 = st.tabs(["Paste JSON", "Upload .json"])
pkt_text = ""

with tab1:
    pkt_text = st.text_area(
        "Validator packet (JSON)",
        height=240,
        placeholder='{"findings":[...],"insights":{...},"bos_index":0.0,...}'
    )

with tab2:
    up = st.file_uploader("Upload JSON file (validator packet)", type=["json"])
    if up:
        pkt_text = up.getvalue().decode("utf-8", errors="ignore")


# ----------------- Helpers -----------------
def find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        if (cur / "slm" / "run_slm.py").exists():
            return cur
        cur = cur.parent
    return start

def parse_json_loose(txt: str):
    """Return last balanced JSON object from arbitrary text."""
    try:
        return json.loads(txt)
    except Exception:
        pass

    start_idx, depth, last_valid = None, 0, None
    for i, ch in enumerate(txt):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidate = txt[start_idx:i+1]
                    try:
                        last_valid = json.loads(candidate)
                    except Exception:
                        pass
    if last_valid is not None:
        return last_valid
    raise ValueError("Could not parse JSON from subprocess output.")

def strip_slm_logs(s: str) -> str:
    """Drop lines like: [SLM] Calling Ollama ..."""
    return "\n".join(line for line in s.splitlines() if not line.strip().startswith("[SLM]"))

def ollama_up(base="http://127.0.0.1:11434"):
    try:
        r = requests.get(base.rstrip("/") + "/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False

def ensure_ui_shape(ui: dict) -> dict:
    if not isinstance(ui, dict):
        return {"executive_summary": str(ui), "_meta": {"engine": "ollama", "model": "unknown", "confidence": 0.0}}
    ui.setdefault("executive_summary", "—")
    ui.setdefault("top_priorities", [])
    ui.setdefault("cross_brain_actions_7d", [])
    ui.setdefault("cross_brain_actions_30d", [])
    ui.setdefault("key_risks", [])
    ui.setdefault("owner_matrix", {})
    ui.setdefault("_meta", {"engine": "ollama", "model": "unknown", "confidence": ui.get("confidence", 0.0)})

    # Accept common aliases / coercions
    if not ui["top_priorities"] and ui.get("priorities"):
        ui["top_priorities"] = ui.pop("priorities")
    if not ui["key_risks"] and ui.get("risks"):
        ui["key_risks"] = ui.pop("risks")
    for k in ("top_priorities", "cross_brain_actions_7d", "cross_brain_actions_30d", "key_risks"):
        if isinstance(ui[k], str):
            ui[k] = [ui[k]]
    if isinstance(ui["owner_matrix"], (list, str)):
        ui["owner_matrix"] = {"EA": ui["owner_matrix"] if isinstance(ui["owner_matrix"], list) else [ui["owner_matrix"]]}
    return ui


# ----------------- Rendering -----------------
def render_ui(payload: dict):
    """Accept either {ui, per_brain} bundle or a bare UI dict."""
    ui = payload.get("ui") if isinstance(payload, dict) else payload
    if not ui:
        ui = payload
    if not isinstance(ui, dict):
        ui = {"executive_summary": str(ui)}
    ui = ensure_ui_shape(ui)

    st.subheader("Executive Summary")
    st.write(ui["executive_summary"])

    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            pct = f"{int(float(ui.get('_meta', {}).get('confidence', 0.0))*100)}%"
        except Exception:
            pct = "—"
        st.metric("Confidence", pct)
    with c2:
        st.write("**Model:**", ui.get("_meta", {}).get("model", "—"))
    with c3:
        st.write("**Engine:**", ui.get("_meta", {}).get("engine", "—"))

    st.divider()
    st.subheader("Top Priorities by Brain")

    def normalize_top(x):
        out = []
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, dict):
                    d = {"brain": k}; d.update(v); out.append(d)
                elif isinstance(v, list):
                    out.append({"brain": k, "actions_7d": v})
                elif isinstance(v, str):
                    out.append({"brain": k, "actions_7d": [v]})
        elif isinstance(x, list):
            for it in x:
                if isinstance(it, dict): out.append(it)
                elif isinstance(it, str): out.append({"brain": "EA", "actions_7d": [it]})
        elif isinstance(x, str):
            out.append({"brain": "EA", "actions_7d": [x]})
        return out

    for item in normalize_top(ui.get("top_priorities", [])):
        brain = item.get("brain", "Unknown")
        with st.expander(f"🧠 {brain}"):
            a7  = item.get("actions_7d")  or []
            a30 = item.get("actions_30d") or []
            if isinstance(a7, str): a7 = [a7]
            if isinstance(a30, str): a30 = [a30]
            if a7:
                st.markdown("**Next 7 days**")
                st.markdown("\n".join(f"- {x}" for x in a7))
            if a30:
                st.markdown("**Next 30 days**")
                st.markdown("\n".join(f"- {x}" for x in a30))
            if not a7 and not a30:
                st.write("—")

    c7  = ui.get("cross_brain_actions_7d")  or []
    c30 = ui.get("cross_brain_actions_30d") or []
    if isinstance(c7, str): c7 = [c7]
    if isinstance(c30, str): c30 = [c30]
    if c7 or c30:
        st.divider(); st.subheader("Cross-Brain Actions")
        if c7:  st.markdown("**7-Day**");  st.markdown("\n".join(f"- {x}" for x in c7))
        if c30: st.markdown("**30-Day**"); st.markdown("\n".join(f"- {x}" for x in c30))

    risks = ui.get("key_risks") or []
    if isinstance(risks, str): risks = [risks]
    if risks:
        st.divider(); st.subheader("Key Risks")
        st.markdown("\n".join(f"- {x}" for x in risks))

    owners = ui.get("owner_matrix") or {}
    if isinstance(owners, (list, str)): owners = {"EA": owners if isinstance(owners, list) else [owners]}
    if owners:
        st.divider(); st.subheader("Owner Matrix")
        for brain, items in owners.items():
            items = items if isinstance(items, list) else [items]
            st.markdown(f"**{brain}**")
            st.markdown("\n".join(f"- {x}" for x in items if x))

    # Per-brain raw (if bundle provided)
    per_brain = payload.get("per_brain") if isinstance(payload, dict) else None
    if per_brain:
        st.divider(); st.subheader("Per-Brain Outputs (raw)")
        for b in ["cfo", "cmo", "coo", "chro", "cpo"]:
            if b in per_brain:
                with st.expander(b.upper()):
                    st.code(json.dumps(per_brain[b], indent=2, ensure_ascii=False))


# ----------------- Run -----------------
st.markdown("—")
go = st.button("Generate EA Plan", type="primary", use_container_width=True)

if go:
    if not pkt_text:
        st.error("Please paste or upload a validator JSON packet."); st.stop()
    try:
        pkt = json.loads(pkt_text)
    except Exception as e:
        st.error(f"Invalid JSON: {e}"); st.stop()

    with st.spinner("Running EA..."):
        if mode.startswith("API"):
            # New API contract
            payload = {
                "packet": pkt,
                "overrides": {
                    "model": model_override or None,
                    "timeout_sec": int(timeout_sec),
                    "num_predict": int(num_predict),
                }
            }
            try:
                r = requests.post(f"{api_base.rstrip('/')}/run-ea", json=payload, timeout=timeout_sec+15)
                r.raise_for_status()
                out = r.json()  # bundle or bare ui
                render_ui(out if isinstance(out, dict) else {"ui": out})
            except Exception as e:
                st.error(f"API error: {e}")
        else:
            # Local subprocess (module run)
            repo_root = find_repo_root(Path(__file__).resolve().parent)
            if not ollama_up():
                st.warning("Ollama not reachable at http://127.0.0.1:11434 — start it or pull the model.")

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as tf:
                json.dump(pkt, tf, indent=2); tf.flush()
                tmp_in = tf.name

            cmd = [
                sys.executable, "-m", "slm.run_slm",
                "--input", tmp_in,
                "--brain", "ea",
                "--config", "slm/config/models.yaml",
                "--timeout", str(int(timeout_sec)),
                "--num_predict", str(int(num_predict)),
            ]
            if model_override:
                cmd += ["--model", model_override]

            try:
                proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout_sec+30)
            except subprocess.TimeoutExpired:
                st.error("Local run timed out — try a smaller model or a longer timeout.")
                st.expander("Command used").write(" ".join(map(str, cmd))); st.stop()
            except Exception as e:
                st.error(f"Local run error: {e}")
                st.expander("Command used").write(" ".join(map(str, cmd))); st.stop()

            raw_out = (proc.stdout or "").strip()
            raw_err = (proc.stderr or "").strip()
            clean_out = strip_slm_logs(raw_out)

            if proc.returncode != 0:
                st.error(f"Subprocess error ({proc.returncode}). See raw logs below.")
                with st.expander("Raw stderr"): st.code(raw_err or "(empty)")
                with st.expander("Raw stdout"): st.code(raw_out or "(empty)")
                st.expander("Command used").write(" ".join(map(str, cmd))); st.stop()

            parsed = None
            try:
                parsed = parse_json_loose(clean_out)
            except Exception:
                try:
                    parsed = json.loads(clean_out)
                except Exception:
                    parsed = None

            if not parsed:
                st.warning("EA returned empty or non-JSON output. See raw logs below.")
                with st.expander("Raw stdout"): st.code(raw_out or "(empty)")
                with st.expander("Raw stderr"): st.code(raw_err or "(empty)")
                st.expander("Command used").write(" ".join(map(str, cmd))); st.stop()

            render_ui(parsed if isinstance(parsed, dict) else {"ui": parsed})
