
# caio-bos-validator (clean-room)

Deterministic validator for CAIO BOS. Mentor-editable **rules-as-data** (YAML), normalized **schemas** (JSON), and a tiny engine that emits **friendly labels**:

- **Authentic enough**
- **Needs attention**
- **Blocked (critical issues)**

It does **not** use ML/LLMs. It is an auditable referee for documents & queries. Later, CAIO can call it behind a feature flag.

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_validator.py --input fixtures/cfo/sample_payload_good.json --rules rules
python run_validator.py --input fixtures/cfo/sample_payload_bad.json --rules rules
```
