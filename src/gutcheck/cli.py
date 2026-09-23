"""`gutcheck` command line."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _read_state(s: str):
    if s == "-":
        s = sys.stdin.read()
    elif s.startswith("@"):
        with open(s[1:], encoding="utf-8") as f:
            s = f.read()
    try:
        v = json.loads(s)
        if isinstance(v, (dict, list)):
            return v
    except Exception:
        pass
    return s


def _bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "#" * n + "." * (width - n)


def _print_answers(res: dict):
    for qid, a in res["answers"].items():
        if a["type"] == "noul":
            print("  %-18s noul  P(true) = %.3f  %s" % (qid, a["noul"], _bar(a["noul"])))
        elif a["type"] == "choice":
            print("  %-18s choice -> %s  (p %.2f)" % (qid, a["choice"], a["probabilities"][a["choice"]]))
            for k, p in sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:6]:
                print("      %-24s %.3f %s" % (k[:24], p, _bar(p, 14)))
        else:
            best = max(a["probabilities"], key=a["probabilities"].get)
            print("  %-18s score  %.2f / %d  (most likely level %s: %s, p %.2f)" % (
                qid, a["score"], len(a["legend"]) - 1, best, str(a["legend"][best])[:30], a["probabilities"][best]))
    print("  [%s on %s, %.0f ms]" % (res["model"], res["device"], res["latency_ms"]))


def cmd_doctor(a):
    import platform

    import openvino as ov

    from . import __version__, hub
    from .runtime.openvino_backend import cache_root, device_names

    print("gutcheck %s | Python %s | OpenVINO %s | %s" % (__version__, platform.python_version(), ov.__version__, platform.platform()))
    print("devices:")
    for d, n in device_names().items():
        print("  %-5s %s" % (d, n))
    from .engine import pick_device
    print("auto device -> %s" % pick_device("auto"))
    print("cache: %s" % cache_root())
    local = hub.list_local()
    print("installed models: %s" % (", ".join(m["name"] for m in local) or "none (run `gutcheck pull laya-en`)"))
    if a.test and local:
        from .engine import Decider
        d = Decider(local[0]["path"], device=a.device, verbose=False)
        t = time.perf_counter()
        r = d.decide("I was charged twice, please refund me", {"refund": {"type": "noul", "instructions": "Is this a refund request?"}})
        print("self-test on %s: P(refund)=%.3f (%.0f ms incl. compile)" % (d.device, r["answers"]["refund"]["noul"], (time.perf_counter() - t) * 1000))


def cmd_models(a):
    from . import hub

    local = {m["name"] for m in hub.list_local()}
    for name, spec in hub.REGISTRY.items():
        mark = "installed" if name in local else "         "
        print("%s  %-22s %5s  %-18s %s" % (mark, name, spec["params"], spec["languages"], spec["description"]))
    for m in hub.list_local():
        if m["name"] not in hub.REGISTRY:
            print("installed  %-22s  (local: %s)" % (m["name"], m["path"]))


def cmd_pull(a):
    from . import hub

    for m in a.models:
        p = hub.resolve(m, weights=a.weights)
        if a.onnx:
            hub.ensure_onnx(p)
        print("ready: %s -> %s" % (m, p))


def cmd_ask(a):
    from .engine import Decider

    state = _read_state(a.state)
    qs = {}
    if a.questions:
        with open(a.questions, encoding="utf-8") as f:
            qs = json.load(f)
    ins = a.instructions or ""
    if a.choice:
        opts = [o.strip() for o in a.choice.split(",") if o.strip()]
        crit = {}
        for o in opts:
            k, _, desc = o.partition("=")
            crit[k.strip()] = desc.strip() or None
        qs["choice"] = {"type": "choice", "instructions": ins or "Which option fits best?", "criteria": crit}
    if a.score:
        qs["score"] = {"type": "score", "instructions": ins or "Rate it.", "criteria": [s.strip() for s in a.score.split("|")]}
    if a.noul:
        qs["noul"] = {"type": "noul", "instructions": a.noul}
    d = Decider(a.model, device=a.device, verbose=not a.json)
    if not qs and d.default_questions is None:
        sys.exit("give at least one of --choice, --score, --noul or --questions "
                 "(only fine-tuned models have questions of their own)")
    res = d.decide(state, qs or None)  # a fine-tuned model answers the questions it was trained on
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        _print_answers(res)


def cmd_serve(a):
    from .server import serve

    serve(a.host, a.port, a.model, a.device, preload=not a.lazy)


def cmd_mcp(a):
    from .mcp_server import main as mcp_main

    mcp_main(model=a.model, device=a.device)


def cmd_bench(a):
    from .engine import Decider

    d = Decider(a.model, device=a.device)
    qs = {"intent": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": {"refund": "money back", "technical_help": "bug or outage", "billing_question": "invoice",
                                  "cancellation": "wants to cancel", "other": None}},
          "urgent": {"type": "noul", "instructions": "Is there time pressure?"},
          "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]}}
    state = {"message": "We were billed twice for March and support hasn't replied in 3 days. Fix it today or we cancel."}
    t = time.perf_counter()
    d.decide(state, qs)
    print("first call (includes compile/cache load): %.0f ms" % ((time.perf_counter() - t) * 1000))
    ts = []
    for _ in range(a.n):
        t = time.perf_counter()
        d.decide(state, qs)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print("%s on %s: %d calls x %d questions | p50 %.1f ms | p90 %.1f ms | %.1f ms/question"
          % (d.name, d.device, a.n, len(qs), ts[len(ts) // 2], ts[int(len(ts) * 0.9)], ts[len(ts) // 2] / len(qs)))


def cmd_warmup(a):
    """Compile every shape bucket now so first real calls never stall (matters on the NPU: ~30 s per
    bucket the first time, then cached on disk for good)."""
    from .engine import Decider

    d = Decider(a.model, device=a.device)
    be = d.backend
    if getattr(be, "dynamic", True):
        be.warmup()
        print("%s on %s uses dynamic shapes; one compile done" % (d.name, d.device))
        return
    t0 = time.perf_counter()
    for L in be.seq_buckets:
        if L > a.max_len:
            continue
        for B in be.batch_buckets:
            t = time.perf_counter()
            be.warmup([(B, L)])
            print("  batch %d x seq %4d: %.1fs" % (B, L, time.perf_counter() - t), flush=True)
    print("%s on %s: all buckets compiled and cached in %.0fs" % (d.name, d.device, time.perf_counter() - t0))


def main(argv=None):
    p = argparse.ArgumentParser(prog="gutcheck", description="Fast, calibrated System-1 decisions on your NPU/GPU/CPU.")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    def common(sp, model=True):
        if model:
            sp.add_argument("-m", "--model", default=os.environ.get("GUTCHECK_MODEL", "laya-en"))
        sp.add_argument("-d", "--device", default="auto",
                        help="auto | NPU | GPU | CPU (OpenVINO) | CUDA | DML | QNN | COREML (ONNX Runtime)")

    sp = sub.add_parser("doctor", help="show hardware, runtime and installed models")
    sp.add_argument("--test", action="store_true", help="run a one-question self-test")
    common(sp, model=False)
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("models", help="list known and installed models")
    sp.set_defaults(fn=cmd_models)

    sp = sub.add_parser("pull", help="download + build model packages")
    sp.add_argument("models", nargs="+")
    sp.add_argument("--weights", default="fp16", choices=["fp16", "int8"])
    sp.add_argument("--onnx", action="store_true", help="also export ONNX (NVIDIA/DirectML/QNN/CoreML backends)")
    sp.set_defaults(fn=cmd_pull)

    sp = sub.add_parser("ask", help="ask typed questions about a state (text, JSON, @file or -)")
    sp.add_argument("state")
    sp.add_argument("--choice", help="comma list: a,b,c  or  label=description,label2=description")
    sp.add_argument("--score", help="pipe list of levels, lowest first: 'calm|annoyed|furious'")
    sp.add_argument("--noul", help="a yes/no statement to test")
    sp.add_argument("-i", "--instructions", help="instructions for --choice/--score")
    sp.add_argument("-q", "--questions", help="JSON file with a Jev questions object")
    sp.add_argument("--json", action="store_true")
    common(sp)
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("serve", help="run the Jev-compatible HTTP API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--lazy", action="store_true", help="load the model on first request")
    common(sp)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("mcp", help="run the MCP server (stdio) for Claude Code, Cursor, etc.")
    common(sp)
    sp.set_defaults(fn=cmd_mcp)

    sp = sub.add_parser("warmup", help="precompile all shape buckets (run once per device; NPU first compile is slow)")
    sp.add_argument("--max-len", type=int, default=512)
    common(sp)
    sp.set_defaults(fn=cmd_warmup)

    sp = sub.add_parser("bench", help="measure latency on this machine")
    sp.add_argument("-n", type=int, default=20)
    common(sp)
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("gateway", help="one MCP server in front of all your MCP servers (import | sync | run | list)")
    sp.add_argument("rest", nargs=argparse.REMAINDER)
    sp.set_defaults(fn=lambda a: __import__("gutcheck.gateway", fromlist=["cli"]).cli(a.rest))

    for mod in ("train", "lease"):
        try:
            __import__("gutcheck.%s.cli" % mod)
            sys.modules["gutcheck.%s.cli" % mod].register(sub, common)
        except ImportError:
            pass

    a = p.parse_args(argv)
    if a.version:
        from . import __version__
        print(__version__)
        return
    if not getattr(a, "fn", None):
        p.print_help()
        return
    a.fn(a)


if __name__ == "__main__":
    main()
