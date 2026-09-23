"""MCP server: give any agent (Claude Code, Cursor, Codex, ...) local, calibrated System-1 decisions.

Four small tools, terse schemas - every tool description costs the agent context on every turn.
The model loads in a background thread at startup so the MCP handshake stays instant.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Union


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2
        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x
        return FastMCP


class _Lazy:
    def __init__(self, model: str, device: str):
        self.model, self.device = model, device
        self._d = None
        self._err: Optional[BaseException] = None
        self._ready = threading.Event()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from .engine import Decider
            self._d = Decider(self.model, device=self.device, verbose=False)
            self._d.warmup((128, 256))
        except BaseException as e:  # surfaced on first call
            self._err = e
        finally:
            self._ready.set()

    def get(self):
        self._ready.wait()
        if self._err:
            raise RuntimeError("gutcheck model failed to load: %s" % self._err)
        return self._d


def _compact(res: Dict[str, Any]) -> str:
    """Only what an agent needs to branch on; probabilities rounded to 3 places."""
    out = {}
    for qid, a in res["answers"].items():
        if a["type"] == "noul":
            out[qid] = {"p_true": round(a["noul"], 3)}
        elif a["type"] == "choice":
            # `p` is the chosen label's probability - what an agent should gate on. (Laya's entropy-based
            # "confidence" reads 0.48 for p=0.88 between two labels, which misleads threshold rules.)
            top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:4]
            out[qid] = {"choice": a["choice"], "p": round(top[0][1], 3), "top": {k: round(v, 3) for k, v in top}}
        else:
            probs = a["probabilities"]
            best = max(probs, key=probs.get)
            out[qid] = {"score": round(a["score"], 2), "max": len(a["legend"]) - 1, "level": int(best),
                        "p_level": round(probs[best], 3)}
    out["_ms"] = res.get("latency_ms")
    return json.dumps(out, ensure_ascii=False)


def build_server(model: str = "laya-en", device: str = "auto"):
    Server = _server_class()
    srv = Server("gutcheck", instructions=(
        "Local, calibrated System-1 decisions (~20-60 ms each, no LLM tokens). Use for classification, "
        "routing, triage, guardrails and yes/no checks over any text or JSON. Probabilities are calibrated: "
        "gate on them (act if p / p_true > 0.8, else ask). Prefer concrete, observable statements over abstract "
        "judgements. Zero-shot quality is modest on niche domains; "
        "`gutcheck train` fine-tunes on a few hundred labels."))
    lazy = _Lazy(model, device)

    @srv.tool()
    def classify(text: str, labels: Union[List[str], Dict[str, str]], question: str = "Which label fits best?") -> str:
        """Pick one label for `text`. `labels`: list of names or {name: description} (descriptions help). Max 64."""
        return _compact(lazy.get().decide(text, {"label": {"type": "choice", "instructions": question, "criteria": labels}}))

    @srv.tool()
    def check(text: str, statement: str) -> str:
        """Probability that a yes/no `statement` holds for `text`, e.g. 'The user is asking for a refund'."""
        return _compact(lazy.get().decide(text, {"check": {"type": "noul", "instructions": statement}}))

    @srv.tool()
    def rate(text: str, question: str, levels: List[str]) -> str:
        """Place `text` on an ordered scale. `levels` lowest first, e.g. ['none','minor','severe']."""
        return _compact(lazy.get().decide(text, {"rate": {"type": "score", "instructions": question, "criteria": levels}}))

    @srv.tool()
    def decide(state: Union[str, Dict[str, Any], List[Any]], questions: Dict[str, Dict[str, Any]]) -> str:
        """Several typed questions in one call (Jev format). questions: {id: {type: choice|score|noul,
        instructions, criteria}}; choice criteria {label: desc}, score criteria [levels], noul criteria optional."""
        return _compact(lazy.get().decide(state, questions))

    return srv


def main(model: Optional[str] = None, device: str = "auto"):
    model = model or os.environ.get("GUTCHECK_MODEL", "laya-en")
    build_server(model, os.environ.get("GUTCHECK_DEVICE", device)).run(transport="stdio")


if __name__ == "__main__":
    main()
