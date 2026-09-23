"""Jev-compatible HTTP server using only the standard library.

    POST /v1/systemone   {"model": "laya-en", "state": ..., "questions": {...}}  -> Jev response
    POST /v1/batch       {"model": ..., "states": [...], "questions": {...}}     -> {"results": [...]}
    GET  /v1/models      installed + known models
    GET  /health

Point any TypeSafe/Jev client at http://127.0.0.1:8765 and it runs locally on your NPU/GPU.
`model: "jev-latest"` (what Jev SDKs send) maps to the server's default model.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

from .spec import QuestionError

JEV_ALIASES = {"jev-latest", "jev", "jev-1", ""}


class ModelPool:
    """Lazily loaded Deciders, one per model name."""

    def __init__(self, default_model: str, device: str):
        self.default_model = default_model
        self.device = device
        self._models: Dict[str, object] = {}
        self._lock = threading.Lock()

    def get(self, name: Optional[str]):
        from .engine import Decider

        name = self.default_model if (name is None or name in JEV_ALIASES or name.startswith("jev-")) else name
        with self._lock:
            d = self._models.get(name)
            if d is None:
                d = Decider(name, device=self.device, verbose=False)
                self._models[name] = d
            return d


def make_handler(pool: ModelPool, api_key: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gutcheck"

        def log_message(self, fmt, *args):  # quiet by default
            if os.environ.get("GUTCHECK_LOG"):
                super().log_message(fmt, *args)

        def _send(self, code: int, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _auth(self) -> bool:
            if not api_key:
                return True
            h = self.headers.get("authorization", "")
            ok = h == "Bearer " + api_key or self.headers.get("x-api-key") == api_key
            if not ok:
                self._send(401, {"error": {"type": "authentication_error", "message": "invalid or missing API key"}})
            return ok

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self._send(200, {"status": "ok", "default_model": pool.default_model, "device": pool.device,
                                        "loaded": list(pool._models)})
            if self.path.rstrip("/") == "/v1/models":
                from . import hub
                return self._send(200, {"default": pool.default_model, "installed": hub.list_local(),
                                        "known": {k: {kk: vv for kk, vv in v.items() if kk != "revision"}
                                                  for k, v in hub.REGISTRY.items()}})
            self._send(404, {"error": {"type": "not_found", "message": self.path}})

        def do_POST(self):
            # read the body before any early reply: answering 401 with the request still unread makes
            # the client's write fail (connection aborted) instead of it seeing the 401
            try:
                n = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(n) if n > 0 else b""
            except Exception:
                raw = b""
            if not self._auth():
                return
            try:
                req = json.loads(raw or b"{}")
            except Exception:
                return self._send(422, {"error": {"type": "invalid_request_error", "message": "body must be JSON"}})
            path = self.path.rstrip("/")
            try:
                if path == "/v1/systemone":
                    if "state" not in req or "questions" not in req:
                        return self._send(422, {"error": {"type": "invalid_request_error",
                                                          "message": "'state' and 'questions' are required"}})
                    d = pool.get(req.get("model"))
                    return self._send(200, d.decide(req["state"], req["questions"]))
                if path == "/v1/batch":
                    d = pool.get(req.get("model"))
                    t = time.perf_counter()
                    res = d.decide_batch(req["states"], req["questions"])
                    return self._send(200, {"results": res, "total_ms": round((time.perf_counter() - t) * 1000, 1)})
                return self._send(404, {"error": {"type": "not_found", "message": path}})
            except (QuestionError, TypeError, ValueError, KeyError) as e:
                return self._send(422, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except Exception as e:  # pragma: no cover
                traceback.print_exc()
                return self._send(500, {"error": {"type": "server_error", "message": str(e)}})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, model: str = "laya-en", device: str = "auto",
          preload: bool = True, api_key: Optional[str] = None):
    pool = ModelPool(model, device)
    if preload:
        d = pool.get(model)
        d.warmup((128, 256))
        print("gutcheck: %s ready on %s" % (d.name, d.device), flush=True)
    api_key = api_key or os.environ.get("GUTCHECK_API_KEY")
    httpd = ThreadingHTTPServer((host, port), make_handler(pool, api_key))
    print("gutcheck: serving Jev-compatible API at http://%s:%d/v1/systemone" % (host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
