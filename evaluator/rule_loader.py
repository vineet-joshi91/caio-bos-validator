# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Dict, Any
import yaml

def load_brain_rules(rules_root: str, brain: str) -> List[Dict[str, Any]]:
    """
    Read all *.yaml under rules/<brain>/ and return a list of rule dicts.
    """
    base = Path(rules_root) / brain
    files = sorted(base.glob("*.yaml"))
    rules = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            rule = yaml.safe_load(f)
            rule["_filepath"] = str(fp)
            rules.append(rule)
    return rules
