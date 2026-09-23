"""The lease daemon: a warm Leaser behind a localhost HTTP endpoint.

    POST /lease   {"prompt": "...", "context": "...optional"}  ->  {"leased": [...], "hinted": [...], "context": "<text>"}
    GET  /health
    POST /reload  re-read the catalog (after installing or editing skills)

The `context` field is ready-to-inject text for an agent (empty string when nothing is relevant).
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .catalog import Catalog, Item
from .config import load_config
from .leaser import LeaseResult, Leaser


def build_catalog(cfg: Dict[str, Any]) -> Catalog:
    cat = Catalog()
    for d in cfg.get("skill_dirs", []):
        cat.merge(Catalog.from_skill_dirs(d))
    for d in cfg.get("agent_dirs", []):
        cat.merge(Catalog.from_agent_dirs(d))
    for p in cfg.get("catalogs", []):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            cat.merge(Catalog.from_json(p))
    return cat


def _how_to_use(item: Item) -> str:
    path = item.payload.get("path")
    if item.kind == "skill" and path:
        return "read %s and follow it" % path
    if item.kind == "subagent" and path:
        return "spawn a general-purpose agent with the instructions in %s" % path
    if item.kind == "mcp_tool":
        return "gateway `call` tool=\"%s\" arguments=%s" % (item.id, item.payload.get("schema", "{}"))
    return path or ""


def render_context(res: LeaseResult) -> str:
    """Compact, agent-facing text. Empty when nothing is worth loading or hinting."""
    if not res.leased and not res.hinted:
        return ""
    lines = []
    if res.leased:
        lines.append("[gutcheck lease] Capabilities likely needed for this request (picked locally in %.0f ms):"
                     % sum(res.ms.values()))
        for l in res.leased:
            lines.append("- %s `%s` (p=%.2f): %s -> %s" % (l.item.kind.replace("_", " "), l.item.name, l.p,
                                                         l.item.description[:300], _how_to_use(l.item)))
    if res.hinted:
        lines.append("[gutcheck lease] Also available on request (not loaded): " +
                     ", ".join("`%s`" % h.item.name for h in res.hinted) +
                     ". Ask gutcheck (`gutcheck lease show <name>`) or read the file if one fits.")
    return "\n".join(lines)


class LeaseService:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or load_config()
        self.lock = threading.Lock()
        self.leaser: Optional[Leaser] = None
        self.loaded_at = 0.0
        self.reload()

    def reload(self):
        cfg = self.cfg
        cat = build_catalog(cfg)
        old = self.leaser
        leaser = Leaser(cat, embedder=old.embedder if old else None, decider=old.decider if old else None,
                        device=cfg["device"], shortlist_k=cfg["shortlist_k"], threshold=cfg["threshold"],
                        max_lease=cfg["max_lease"], hint_k=cfg["hint_k"], mode=cfg["mode"], router=cfg["router"],
                        pinned=cfg.get("pinned", []), none_silence=cfg.get("none_silence"))
        leaser.lease("warm up the router")  # compile the common buckets now, not on the first real prompt
        with self.lock:
            self.leaser = leaser
            self.loaded_at = time.time()

    def lease(self, prompt: str, context: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            leaser = self.leaser
        if leaser is None or not len(leaser.catalog):
            return {"leased": [], "hinted": [], "context": "", "catalog": 0}
        res = leaser.lease(prompt, context)
        return {
            "leased": [{"id": l.item.id, "name": l.item.name, "kind": l.item.kind, "p": round(l.p, 3),
                        "path": l.item.payload.get("path")} for l in res.leased],
            "hinted": [{"id": h.item.id, "name": h.item.name, "kind": h.item.kind} for h in res.hinted],
            "context": render_context(res),
            "tokens": res.tokens_leased, "catalog_tokens": res.tokens_catalog, "catalog": len(leaser.catalog),
            "ms": res.ms,
        }


def serve(cfg: Optional[Dict[str, Any]] = None, host: str = "127.0.0.1"):
    svc = LeaseService(cfg)
    port = svc.cfg["port"]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/health"):
                lz = svc.leaser
                return self._send(200, {"status": "ok", "catalog": len(lz.catalog) if lz else 0,
                                        "catalog_tokens": lz.catalog.total_tokens if lz else 0,
                                        "device": lz.embedder.device if lz else None, "pid": os.getpid()})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("content-length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path.startswith("/lease"):
                    return self._send(200, svc.lease(str(req.get("prompt", "")), req.get("context")))
                if self.path.startswith("/reload"):
                    svc.reload()
                    return self._send(200, {"status": "reloaded", "catalog": len(svc.leaser.catalog)})
                self._send(404, {"error": "not found"})
            except Exception as e:  # never take the agent down with us
                self._send(500, {"error": str(e)})

    httpd = ThreadingHTTPServer((host, port), H)
    print("gutcheck lease: %d items (%d tokens if all loaded) on %s -> http://%s:%d/lease"
          % (len(svc.leaser.catalog), svc.leaser.catalog.total_tokens, svc.leaser.embedder.device, host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
