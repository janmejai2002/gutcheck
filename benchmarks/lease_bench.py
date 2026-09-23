"""Lease benchmark: how much context does leasing save, and what does it miss?

    python benchmarks/lease_bench.py --split test --router lease-router-bench --device GPU

Data: data/lease/catalog.json (110 real-world skills / MCP tools / subagents) and labelled prompts
data/lease/{dev,test}.jsonl ({"prompt", "needs": [ids]}). Labels were synthesised with Gemini 3.8 Flash
and spot-checked; treat absolute numbers as indicative, comparisons between configs as the signal.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gutcheck.lease.catalog import Catalog  # noqa: E402
from gutcheck.lease.leaser import Leaser, estimate_hint_tokens  # noqa: E402


def evaluate(rows, results, cat, thr, max_lease=4, hint_k=5, silence=None):
    tp = fp = fn = toks = covered = reach = none_ok = nnone = 0
    for r, (sl, p_none) in zip(rows, results):
        g = set(r["needs"])
        ranked = sorted(sl, key=lambda c: -c.p)
        got = [c.item.id for c in ranked if c.p >= thr][:max_lease]
        hints = [c.item.id for c in sorted(sl, key=lambda c: c.rank) if c.item.id not in got][: max(0, hint_k - len(got))]
        if silence is not None and not got and p_none is not None and p_none >= silence:
            hints = []
        tp += len(set(got) & g)
        fp += len(set(got) - g)
        fn += len(g - set(got))
        toks += sum(cat[i].token_cost for i in got) + sum(estimate_hint_tokens(cat[i]) for i in hints)
        covered += g <= set(got)
        reach += len(g & (set(got) | set(hints)))
        if not g:
            nnone += 1
            none_ok += not got
    n_gold = sum(len(r["needs"]) for r in rows)
    return {"threshold": thr, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "reachable_recall": reach / max(1, n_gold), "prompts_fully_covered": covered / len(rows),
            "silent_when_nothing_needed": none_ok / max(1, nnone), "avg_tokens": toks / len(rows),
            "none_silence": silence,
            "saved_pct": 100 * (1 - toks / len(rows) / cat.total_tokens)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--router", default="laya-en")
    ap.add_argument("--mode", default="choice", choices=["choice", "noul", "dense"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--out")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cat = Catalog.from_json(os.path.join(root, "data/lease/catalog.json"))
    rows = [json.loads(l) for l in open(os.path.join(root, "data/lease/%s.jsonl" % a.split), encoding="utf-8")]
    lz = Leaser(cat, device=a.device, shortlist_k=a.k, mode=a.mode, router=a.router)
    lz.lease("warm up")

    results, ms = [], []
    recall_at = {k: [] for k in (1, 3, 5, 8)}
    for r in rows:
        t = time.perf_counter()
        sl = lz.shortlist(r["prompt"])
        ids = [c.item.id for c in sl]
        for k in recall_at:
            if r["needs"]:
                recall_at[k].append(len(set(ids[:k]) & set(r["needs"])) / len(r["needs"]))
        lz.last_p_none = None
        if a.mode == "choice":
            sl = lz.decide_choice(r["prompt"], sl)
        elif a.mode == "noul":
            sl = lz.decide(r["prompt"], sl)
        ms.append((time.perf_counter() - t) * 1000)
        results.append((sl, lz.last_p_none))

    thresholds = [float(x) for x in a.thresholds.split(",")]
    report = {"split": a.split, "n": len(rows), "router": a.router if a.mode != "dense" else None, "mode": a.mode,
              "device": lz.embedder.device, "catalog_items": len(cat), "catalog_tokens": cat.total_tokens,
              "shortlist_recall_at": {k: round(float(np.mean(v)), 3) for k, v in recall_at.items()},
              "latency_ms_p50": round(float(np.median(ms)), 1), "latency_ms_p90": round(float(np.percentile(ms, 90)), 1),
              "by_threshold": [evaluate(rows, results, cat, t) for t in thresholds],
              "with_none_silence": [evaluate(rows, results, cat, t, silence=s) for t in thresholds for s in (0.8, 0.9, 0.95)]
              if a.mode == "choice" else []}
    print("%s | %s/%s | n=%d | shortlist recall@k %s | p50 %.0f ms" % (
        a.split, a.mode, report["router"], len(rows), report["shortlist_recall_at"], report["latency_ms_p50"]))
    for e in report["by_threshold"]:
        print("  thr %.2f  P %.3f  R %.3f  reach %.3f  covered %.3f  silent %.3f  %5.0f tok  (%.1f%% saved)" % (
            e["threshold"], e["precision"], e["recall"], e["reachable_recall"], e["prompts_fully_covered"],
            e["silent_when_nothing_needed"], e["avg_tokens"], e["saved_pct"]))
    for e in report["with_none_silence"]:
        print("  thr %.2f silence %.2f  P %.3f  R %.3f  reach %.3f  %5.0f tok  (%.1f%% saved)" % (
            e["threshold"], e["none_silence"], e["precision"], e["recall"], e["reachable_recall"], e["avg_tokens"], e["saved_pct"]))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
