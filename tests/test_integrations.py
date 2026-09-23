"""HTTP server, Claude Code hook installer, and the MCP gateway against a real (fake) downstream server."""
import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gutcheck import server as gserver


class FakeDecider:
    name, device = "fake", "CPU"

    def decide(self, state, questions):
        if not isinstance(questions, dict):
            raise TypeError("questions must be an object")
        return {"model": "fake", "answers": {k: {"type": "noul", "noul": 0.7} for k in questions},
                "usage": {"input_tokens": 1, "output_tokens": 0}, "device": "CPU", "latency_ms": 1.0}

    def decide_batch(self, states, questions):
        return [self.decide(s, questions) for s in states]


def _start(api_key=None):
    pool = gserver.ModelPool("fake", "CPU")
    pool._models["fake"] = FakeDecider()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), gserver.make_handler(pool, api_key))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, "http://127.0.0.1:%d" % httpd.server_address[1]


def _post(url, obj, headers=None):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_jev_compatible_endpoint_maps_jev_latest():
    httpd, url = _start()
    try:
        code, body = _post(url + "/v1/systemone", {"model": "jev-latest", "state": "hi",
                                                   "questions": {"q": {"type": "noul", "instructions": "x"}}})
        assert code == 200 and body["answers"]["q"]["noul"] == 0.7
        code, body = _post(url + "/v1/systemone", {"state": "hi"})
        assert code == 422
        code, body = _post(url + "/v1/batch", {"states": ["a", "b"], "questions": {"q": {"type": "noul", "instructions": "x"}}})
        assert code == 200 and len(body["results"]) == 2
    finally:
        httpd.shutdown()


def test_api_key_enforced():
    httpd, url = _start(api_key="s3cret")
    try:
        q = {"state": "hi", "questions": {"q": {"type": "noul", "instructions": "x"}}}
        assert _post(url + "/v1/systemone", q)[0] == 401
        assert _post(url + "/v1/systemone", q, {"authorization": "Bearer s3cret"})[0] == 200
    finally:
        httpd.shutdown()


def test_claude_install_is_idempotent_and_reversible(gutcheck_home, tmp_path):
    from gutcheck.lease import claude

    settings = tmp_path / "settings.json"
    original = {"permissions": {"allow": ["Bash(ls:*)"]},
                "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo mine"}]}]}}
    settings.write_text(json.dumps(original), encoding="utf-8")
    r1 = claude.install(str(settings))
    claude.install(str(settings))  # second install must not duplicate
    s = json.loads(settings.read_text(encoding="utf-8"))
    ups = s["hooks"]["UserPromptSubmit"]
    assert len(ups) == 2 and "echo mine" in json.dumps(ups[0])
    assert len(s["hooks"]["SessionStart"]) == 1
    assert s["permissions"] == original["permissions"]
    assert r1["backup"] and os.path.exists(r1["backup"])
    assert os.path.exists(r1["hook"])
    claude.uninstall(str(settings))
    assert json.loads(settings.read_text(encoding="utf-8")) == original


def test_hook_script_is_silent_when_daemon_down(gutcheck_home, tmp_path):
    from gutcheck.lease import claude

    script = claude.write_hook_script(port=1, timeout_ms=200, python=sys.executable, dest=str(tmp_path / "h.py"))
    # port 1 refuses; the hook must print nothing and exit 0. Use a python that cannot start gutcheck
    # so the "start daemon" fallback is harmless.
    body = open(script, encoding="utf-8").read().replace(repr(sys.executable), repr(sys.executable + "-missing"))
    open(script, "w", encoding="utf-8").write(body)
    r = subprocess.run([sys.executable, script, "prompt"], input=json.dumps({"prompt": "make a deck"}),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and r.stdout == ""


FAKE_SERVER = textwrap.dedent('''
    try:
        from mcp.server.mcpserver import MCPServer as S
    except ImportError:
        from mcp.server.fastmcp import FastMCP as S
    srv = S("fake")

    @srv.tool()
    def add(a: int, b: int) -> str:
        """Add two integers."""
        return str(a + b)

    @srv.tool()
    def shout(text: str) -> str:
        """Upper-case some text."""
        return text.upper()

    srv.run(transport="stdio")
''')


@pytest.mark.skipif(not __import__("importlib").util.find_spec("mcp"), reason="mcp not installed")
def test_gateway_sync_and_call(gutcheck_home, tmp_path):
    from gutcheck import gateway

    fake = tmp_path / "fake_server.py"
    fake.write_text(FAKE_SERVER, encoding="utf-8")
    gateway.save_gateway_config({"servers": {"fake": {"command": sys.executable, "args": [str(fake)]}}})
    cat = asyncio.run(gateway.sync_catalog(log=lambda *a: None))
    assert {"fake__add", "fake__shout"} <= {i.id for i in cat}
    assert cat["fake__add"].payload["schema"] == "{a:int!, b:int!}"
    from gutcheck.lease.config import load_config
    assert gateway.tools_catalog_path() in load_config()["catalogs"]

    async def run():
        ds = gateway.Downstream("fake", {"command": sys.executable, "args": [str(fake)]})
        try:
            res = await ds.call("add", {"a": 2, "b": 3})
            return [getattr(c, "text", None) for c in res.content]
        finally:
            await ds.close()

    assert asyncio.run(run()) == ["5"]


def test_compact_schema():
    from gutcheck.gateway import compact_schema

    s = {"type": "object", "required": ["title"], "properties": {
        "title": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}},
        "state": {"type": "string", "enum": ["open", "closed"]}}}
    assert compact_schema(s) == "{title:str!, labels:[str], state:open|closed}"
    assert compact_schema(None) == "{}"
