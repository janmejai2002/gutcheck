"""Synthesise labelled routing prompts for a catalog with any LLM command line.

    gutcheck lease synth --llm "claude -p" --out prompts.jsonl        # or "agy -p", "llm", "ollama run qwen3" ...
    gutcheck lease learn --data prompts.jsonl --name my-router

The LLM gets the catalog (id | kind | description) and returns JSON lines {"prompt", "needs": [ids]}.
Output is validated: unknown ids and duplicates are dropped.
"""
from __future__ import annotations

import json
import random
import shlex
import subprocess
from typing import Callable, Dict, List, Optional

from .catalog import Catalog

PROMPT = """Output ONLY JSON Lines, one object per line, no markdown fences, no commentary.
Below is a catalog of capabilities installed in a user's AI agent (id | kind | description).
Write {n} realistic messages the user might send their agent, each with the list of catalog ids genuinely needed.
Format: {{"prompt": "<message>", "needs": ["<id>", ...]}}
Mix: ~25% need nothing from the catalog; ~45% need exactly one item; ~30% need two or three that work together.
Make it realistic and hard: vary length and tone, paraphrase instead of repeating catalog words, include typos,
mention products without needing their tool, and include confusable cases where only one similar item is right.
Focus especially on these ids so each appears in at least one message: {focus}
Only use ids from the catalog.
CATALOG:
{catalog}
"""


def run_llm(cmd: str, prompt: str, timeout: int = 900) -> str:
    """`cmd` is a shell-style command; the prompt is passed as its final argument."""
    args = shlex.split(cmd, posix=True) + [prompt]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return r.stdout


def parse_lines(text: str, catalog: Catalog) -> List[Dict]:
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r.get("prompt"), str) and isinstance(r.get("needs"), list) and all(x in catalog for x in r["needs"]):
            out.append({"prompt": r["prompt"], "needs": r["needs"]})
    return out


def synth(catalog: Catalog, llm: str, per_call: int = 60, calls: Optional[int] = None, seed: int = 0,
          log: Callable = print, llm_fn: Optional[Callable[[str], str]] = None) -> List[Dict]:
    ids = [it.id for it in catalog]
    random.Random(seed).shuffle(ids)
    calls = calls or max(1, -(-len(ids) // 20))
    lines = "\n".join("%s | %s | %s" % (it.id, it.kind, it.description[:300]) for it in catalog)
    rows, seen = [], set()
    for c in range(calls):
        focus = ids[c * 20:(c + 1) * 20] or random.Random(seed + c).sample(ids, min(20, len(ids)))
        text = (llm_fn or (lambda p: run_llm(llm, p)))(PROMPT.format(n=per_call, focus=", ".join(focus), catalog=lines))
        new = []
        for r in parse_lines(text, catalog):
            key = r["prompt"].strip().lower()
            if key not in seen:  # dedupe within this response as well as across calls
                seen.add(key)
                new.append(r)
        rows.extend(new)
        log("  call %d/%d: %d valid prompts (%d total)" % (c + 1, calls, len(new), len(rows)))
    return rows
