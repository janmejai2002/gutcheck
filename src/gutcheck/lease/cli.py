"""`gutcheck lease ...` commands."""
from __future__ import annotations

import json
import os
import sys


def _catalog(a):
    from .catalog import Catalog
    from .config import load_config
    from .daemon import build_catalog

    cfg = load_config()
    if getattr(a, "catalog", None):
        return Catalog.from_json(a.catalog), cfg
    return build_catalog(cfg), cfg


def cmd_serve(a):
    from .config import load_config
    from .daemon import serve

    cfg = load_config()
    if a.port:
        cfg["port"] = a.port
    serve(cfg)


def cmd_try(a):
    """Lease once. Uses the running daemon if there is one, otherwise loads everything in-process."""
    import urllib.request

    from .config import load_config

    cfg = load_config()
    if not a.catalog:
        try:
            req = urllib.request.Request("http://127.0.0.1:%d/lease" % cfg["port"],
                                         data=json.dumps({"prompt": a.prompt}).encode(),
                                         headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                res = json.loads(r.read())
            print(res["context"] or "(nothing leased)")
            print("\n%d of %d catalog tokens | %s" % (res["tokens"], res["catalog_tokens"], res["ms"]))
            return
        except Exception:
            pass
    from .daemon import render_context
    from .leaser import Leaser

    cat, cfg = _catalog(a)
    lz = Leaser(cat, device=a.device or cfg["device"], shortlist_k=cfg["shortlist_k"], threshold=cfg["threshold"],
                max_lease=cfg["max_lease"], hint_k=cfg["hint_k"], mode=cfg["mode"], router=a.router or cfg["router"])
    res = lz.lease(a.prompt)
    print(render_context(res) or "(nothing leased)")
    print("\nshortlist: " + ", ".join("%s=%.2f" % (c.item.name, c.p) for c in res.shortlist))
    print("%d of %d catalog tokens | %s" % (res.tokens_leased, res.tokens_catalog, res.ms))


def cmd_catalog(a):
    cat, cfg = _catalog(a)
    kinds = {}
    for it in cat:
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    print("%d items, %d tokens if all loaded | %s" % (len(cat), cat.total_tokens, kinds))
    print("sources: skills %s | agents %s | catalogs %s" % (cfg["skill_dirs"], cfg["agent_dirs"], cfg["catalogs"]))
    if a.list:
        for it in cat:
            print("  %-10s %-40s %5d  %s" % (it.kind, it.name[:40], it.token_cost, it.description[:70]))


def cmd_show(a):
    cat, _ = _catalog(a)
    hits = [it for it in cat if a.name in (it.name, it.id)]
    if not hits:
        sys.exit("no catalog item named %r" % a.name)
    it = hits[0]
    path = it.payload.get("path")
    print("%s `%s` (%s)\n%s" % (it.kind, it.name, it.id, it.description))
    if path and os.path.exists(path):
        print("\n--- %s ---" % path)
        with open(path, encoding="utf-8", errors="replace") as f:
            print(f.read())


def cmd_config(a):
    from .config import config_path, load_config, save_config

    cfg = load_config()
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        try:
            cfg[k] = json.loads(v)
        except Exception:
            cfg[k] = v
    if a.set:
        save_config(cfg)
    print(config_path())
    print(json.dumps(cfg, indent=2))


def cmd_install(a):
    if a.target != "claude":
        sys.exit("supported targets: claude")
    from .claude import install

    res = install(a.settings, dry_run=a.dry_run)
    if a.dry_run:
        print(json.dumps(res["would_write"].get("hooks"), indent=2))
        print("\n(dry run; %s not modified)" % res["settings"])
    else:
        print("installed hooks into %s\n  backup: %s\n  hook:   %s" % (res["settings"], res["backup"], res["hook"]))
        print("restart Claude Code. Skills in your lease skill_dirs are now leased per prompt.")


def cmd_uninstall(a):
    from .claude import uninstall

    print(uninstall(a.settings))


def cmd_learn(a):
    from .learn import learn

    cat, _ = _catalog(a)
    with open(a.data, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    unknown = {x for r in rows for x in r.get("needs", []) if x not in cat}
    if unknown:
        sys.exit("labels reference ids not in the catalog: %s" % sorted(unknown)[:10])
    out = learn(cat, rows, a.name, base=a.base, device=a.device or "auto", epochs=a.epochs, lr=a.lr, depth=a.depth)
    print("router ready: %s\nuse it:  gutcheck lease config router=%s" % (out, a.name))


def cmd_synth(a):
    from .synth import synth

    cat, _ = _catalog(a)
    rows = synth(cat, a.llm, per_call=a.per_call, calls=a.calls)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d labelled prompts to %s\nnext: gutcheck lease learn --data %s" % (len(rows), a.out, a.out))


def register(sub, common):
    p = sub.add_parser("lease", help="lease skills/tools/agents per prompt to save context tokens")
    ls = p.add_subparsers(dest="lease_cmd")

    sp = ls.add_parser("serve", help="run the warm lease daemon (hooks talk to it)")
    sp.add_argument("--port", type=int)
    sp.set_defaults(fn=cmd_serve)

    sp = ls.add_parser("try", help="see what a prompt would lease")
    sp.add_argument("prompt")
    sp.add_argument("--catalog", help="JSON catalog instead of the configured sources")
    sp.add_argument("--router")
    sp.add_argument("-d", "--device")
    sp.set_defaults(fn=cmd_try)

    sp = ls.add_parser("catalog", help="summarise the leasable catalog")
    sp.add_argument("--catalog")
    sp.add_argument("--list", action="store_true")
    sp.set_defaults(fn=cmd_catalog)

    sp = ls.add_parser("show", help="print one catalog item (and its SKILL.md)")
    sp.add_argument("name")
    sp.add_argument("--catalog")
    sp.set_defaults(fn=cmd_show)

    sp = ls.add_parser("config", help="show or set lease settings: key=value ...")
    sp.add_argument("set", nargs="*")
    sp.set_defaults(fn=cmd_config)

    sp = ls.add_parser("install", help="wire leasing into an agent (claude)")
    sp.add_argument("target")
    sp.add_argument("--settings", help="settings.json path (default ~/.claude/settings.json)")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=cmd_install)

    sp = ls.add_parser("uninstall", help="remove the hooks")
    sp.add_argument("target")
    sp.add_argument("--settings")
    sp.set_defaults(fn=cmd_uninstall)

    sp = ls.add_parser("learn", help="train a router for your catalog from labelled prompts")
    sp.add_argument("--data", required=True, help='JSONL of {"prompt": ..., "needs": [ids]}')
    sp.add_argument("--name", default="lease-router")
    sp.add_argument("--catalog")
    sp.add_argument("--base", default="laya-en")
    sp.add_argument("--epochs", type=int, default=3)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--depth", type=int, default=0, help="also train the top N encoder layers (slower, stronger)")
    sp.add_argument("-d", "--device")
    sp.set_defaults(fn=cmd_learn)
    sp = ls.add_parser("synth", help="generate labelled routing prompts for your catalog with any LLM CLI")
    sp.add_argument("--llm", required=True, help='command that takes a prompt as its last argument, e.g. "claude -p"')
    sp.add_argument("--out", default="lease_prompts.jsonl")
    sp.add_argument("--per-call", type=int, default=60)
    sp.add_argument("--calls", type=int, help="default: one call per 20 catalog items")
    sp.add_argument("--catalog")
    sp.set_defaults(fn=cmd_synth)
    p.set_defaults(fn=lambda a: p.print_help())
