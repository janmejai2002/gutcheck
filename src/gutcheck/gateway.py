"""MCP gateway: one small MCP server in front of all your MCP servers.

Instead of every downstream tool schema sitting in the agent's context on every turn (often 10-80k
tokens), the agent sees two tools:

    find_tools(need)          -> the few downstream tools that fit, with compact schemas
    call(tool, arguments)     -> runs `server__tool` on the right downstream server

The prompt hook (`gutcheck lease install claude`) can also pre-lease the right tools per prompt, so the
agent usually calls `call` directly without a search round-trip.

Config: ~/.cache/gutcheck/gateway.json  {"servers": {"github": {"command": "npx", "args": [...], "env": {...}},
                                                      "docs": {"url": "https://..."}}}
`gutcheck gateway import` copies server entries from Claude Code's config; `gutcheck gateway sync` lists
every downstream tool into a lease catalog (mcp_tools.json).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .lease.catalog import Catalog, Item, estimate_tokens
from .runtime.openvino_backend import cache_root

SEP = "__"


def gateway_config_path() -> str:
    return os.environ.get("GUTCHECK_GATEWAY_CONFIG", os.path.join(cache_root(), "gateway.json"))


def tools_catalog_path() -> str:
    return os.path.join(cache_root(), "mcp_tools.json")


def load_gateway_config() -> Dict[str, Any]:
    p = gateway_config_path()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"servers": {}}


def save_gateway_config(cfg: Dict[str, Any]):
    os.makedirs(os.path.dirname(gateway_config_path()), exist_ok=True)
    with open(gateway_config_path(), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def import_claude_servers(claude_json: str = "~/.claude.json", include: Optional[List[str]] = None) -> Dict[str, Any]:
    """Copy user-level MCP servers from Claude Code's config (read-only on the source)."""
    with open(os.path.expanduser(claude_json), encoding="utf-8") as f:
        src = json.load(f).get("mcpServers", {})
    cfg = load_gateway_config()
    added = []
    for name, spec in src.items():
        if name == "gutcheck" or (include and name not in include):
            continue
        if "command" in spec or "url" in spec:
            cfg["servers"][name] = {k: v for k, v in spec.items() if k in ("command", "args", "env", "url", "type", "headers", "cwd")}
            added.append(name)
    save_gateway_config(cfg)
    return {"added": added, "config": gateway_config_path()}


def compact_schema(schema: Optional[Dict[str, Any]], limit: int = 400) -> str:
    """`{title:str!, body:str, labels:[str]}` - the smallest faithful rendering of a JSON schema."""
    if not schema:
        return "{}"
    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])

    def t(p):
        ty = p.get("type")
        if ty == "array":
            return "[%s]" % t(p.get("items") or {})
        if "enum" in p:
            return "|".join(map(str, p["enum"][:6]))
        return {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "object": "obj"}.get(ty, ty or "any")

    s = "{" + ", ".join("%s:%s%s" % (k, t(v), "!" if k in req else "") for k, v in props.items()) + "}"
    return s if len(s) <= limit else s[: limit - 3] + "..."


# ------------------------------------------------------------------------------------------------
# downstream connections
# ------------------------------------------------------------------------------------------------

class Downstream:
    """A persistent connection to one downstream MCP server (SDK v2 `Client`, or v1 session)."""

    def __init__(self, name: str, spec: Dict[str, Any]):
        self.name, self.spec = name, spec
        self._stack: Optional[contextlib.AsyncExitStack] = None
        self._client = None
        self._lock = asyncio.Lock()

    async def _connect(self):
        stack = contextlib.AsyncExitStack()
        spec = self.spec
        try:
            from mcp import Client  # v2
            from mcp.client.stdio import StdioServerParameters

            target = spec["url"] if "url" in spec else StdioServerParameters(
                command=spec["command"], args=spec.get("args", []), env={**os.environ, **spec.get("env", {})},
                cwd=spec.get("cwd"))
            self._client = await stack.enter_async_context(Client(target))
        except ImportError:  # mcp 1.x
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            if "url" in spec:
                from mcp.client.streamable_http import streamablehttp_client
                read, write, _ = await stack.enter_async_context(streamablehttp_client(spec["url"]))
            else:
                read, write = await stack.enter_async_context(stdio_client(StdioServerParameters(
                    command=spec["command"], args=spec.get("args", []), env={**os.environ, **spec.get("env", {})})))
            self._client = await stack.enter_async_context(ClientSession(read, write))
            await self._client.initialize()
        self._stack = stack

    async def client(self):
        async with self._lock:
            if self._client is None:
                await self._connect()
            return self._client

    async def list_tools(self) -> List[Any]:
        c = await self.client()
        return list((await c.list_tools()).tools)

    async def call(self, tool: str, arguments: Dict[str, Any]):
        c = await self.client()
        return await c.call_tool(tool, arguments)

    async def close(self):
        if self._stack:
            with contextlib.suppress(Exception):
                await self._stack.aclose()
        self._stack = self._client = None


def _tool_schema(t) -> Dict[str, Any]:
    return getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}


async def sync_catalog(cfg: Optional[Dict[str, Any]] = None, log=print) -> Catalog:
    """Connect to every configured server, list its tools, write the lease catalog."""
    cfg = cfg or load_gateway_config()
    cat = Catalog()
    for name, spec in cfg.get("servers", {}).items():
        ds = Downstream(name, spec)
        try:
            tools = await asyncio.wait_for(ds.list_tools(), timeout=60)
        except Exception as e:
            log("  %-20s FAILED: %s" % (name, str(e)[:120]))
            continue
        finally:
            await ds.close()
        for t in tools:
            schema = _tool_schema(t)
            desc = (t.description or "").strip()
            full_cost = estimate_tokens(json.dumps({"name": t.name, "description": desc, "input_schema": schema}))
            cat.add(Item(id=name + SEP + t.name, name=name + SEP + t.name, description=desc, kind="mcp_tool",
                         token_cost=full_cost, payload={"server": name, "tool": t.name, "schema": compact_schema(schema)}))
        log("  %-20s %d tools" % (name, len(tools)))
    cat.to_json(tools_catalog_path())
    # make the prompt hook aware of these tools
    from .lease.config import load_config, save_config

    lc = load_config()
    if tools_catalog_path() not in [os.path.expanduser(p) for p in lc.get("catalogs", [])]:
        lc["catalogs"] = list(lc.get("catalogs", [])) + [tools_catalog_path()]
        save_config(lc)
    return cat


# ------------------------------------------------------------------------------------------------
# the gateway server
# ------------------------------------------------------------------------------------------------

def _lease_via_daemon(need: str, port: int) -> Optional[List[str]]:
    import urllib.request

    try:
        req = urllib.request.Request("http://127.0.0.1:%d/lease" % port, data=json.dumps({"prompt": need}).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as r:
            res = json.loads(r.read())
        return [x["id"] for x in res.get("leased", []) + res.get("hinted", [])]
    except Exception:
        return None


def build_gateway(cfg: Optional[Dict[str, Any]] = None):
    from .lease.config import load_config
    from .mcp_server import _server_class

    cfg = cfg or load_gateway_config()
    servers = {n: Downstream(n, s) for n, s in cfg.get("servers", {}).items()}
    cat_path = tools_catalog_path()
    catalog = Catalog.from_json(cat_path) if os.path.exists(cat_path) else Catalog()
    lease_port = load_config()["port"]
    state: Dict[str, Any] = {"leaser": None}

    def local_rank(need: str, k: int) -> List[str]:
        if state["leaser"] is None:
            from .lease.leaser import Leaser
            state["leaser"] = Leaser(catalog, mode="dense", shortlist_k=k)
        return [c.item.id for c in state["leaser"].shortlist(need, k)]

    Server = _server_class()
    srv = Server("gutcheck-gateway", instructions=(
        "Gateway to %d tools on %d MCP servers (%s). Their schemas are NOT preloaded: call find_tools with what you "
        "need, then call(tool, arguments). If a [gutcheck lease] note already lists a tool, call it directly."
        % (len(catalog), len(servers), ", ".join(servers) or "none configured")))

    @srv.tool()
    def find_tools(need: str, k: int = 5) -> str:
        """Find downstream tools for a need, e.g. 'open a GitHub pull request'. Returns names + compact arg schemas."""
        if not len(catalog):
            return "no tools synced; run `gutcheck gateway sync`"
        ids = [i for i in (_lease_via_daemon(need, lease_port) or []) if i in catalog] or local_rank(need, k)
        lines = []
        for i in ids[:k]:
            it = catalog[i]
            lines.append("%s %s - %s" % (it.id, it.payload.get("schema", "{}"), it.description[:200]))
        return "\n".join(lines) or "no matching tools"

    @srv.tool()
    async def call(tool: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Run a downstream tool by its `server__tool` name with its arguments object."""
        server, _, name = tool.partition(SEP)
        if server not in servers or not name:
            return "unknown tool %r; use find_tools first" % tool
        res = await servers[server].call(name, arguments or {})
        parts = []
        for c in getattr(res, "content", None) or []:
            txt = getattr(c, "text", None)
            parts.append(txt if txt is not None else json.dumps(c.model_dump() if hasattr(c, "model_dump") else str(c))[:4000])
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            return "ERROR: " + "\n".join(parts)
        return "\n".join(parts) if parts else json.dumps(getattr(res, "structured_content", None) or {})

    return srv


def main():
    build_gateway().run(transport="stdio")


def cli(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="gutcheck gateway")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("run", help="run the gateway MCP server (stdio)")
    sp = sub.add_parser("import", help="copy MCP servers from Claude Code's ~/.claude.json")
    sp.add_argument("--only", nargs="*")
    sub.add_parser("sync", help="list every downstream tool into the lease catalog")
    sub.add_parser("list", help="show configured servers")
    a = p.parse_args(argv)
    if a.cmd == "import":
        print(json.dumps(import_claude_servers(include=a.only), indent=2))
        print("next: gutcheck gateway sync")
    elif a.cmd == "sync":
        cat = asyncio.run(sync_catalog())
        print("%d tools, %d tokens if all loaded -> %s" % (len(cat), cat.total_tokens, tools_catalog_path()))
    elif a.cmd == "list":
        print(json.dumps(load_gateway_config(), indent=2))
    elif a.cmd == "run":
        main()
    else:
        p.print_help()


if __name__ == "__main__":
    cli(sys.argv[1:])
