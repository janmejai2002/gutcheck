"""Lease configuration (~/.cache/gutcheck/lease.json, or $GUTCHECK_LEASE_CONFIG)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "port": 8766,
    "device": "auto",
    "router": "laya-en",          # or a head trained with `gutcheck lease learn`
    "mode": "choice",
    "shortlist_k": 8,
    "threshold": 0.2,
    "max_lease": 3,
    "hint_k": 5,
    "none_silence": None,         # e.g. 0.9: inject nothing when the router is that sure nothing is needed
    # where leasable things live. Keep them OUT of the directories your agent scans at startup
    # (e.g. ~/.claude/skills) so they cost nothing until leased.
    "skill_dirs": ["~/.claude/skills-library"],
    "agent_dirs": ["~/.claude/agents-library"],
    "catalogs": [],               # extra JSON catalogs (e.g. MCP tools from the gateway)
    "pinned": [],
    "timeout_ms": 1500,           # the hook gives up after this and injects nothing
}


def config_path() -> str:
    from ..runtime.openvino_backend import cache_root

    return os.environ.get("GUTCHECK_LEASE_CONFIG", os.path.join(cache_root(), "lease.json"))


def load_config() -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    p = config_path()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg: Dict[str, Any]):
    p = config_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
