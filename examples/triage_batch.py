"""Triage a whole inbox in one batched pass and print a routing table.

    python examples/triage_batch.py data/support/test.jsonl
"""
import json
import sys
import time

import gutcheck

QUESTIONS = json.load(open("examples/tasks/support-triage.json", encoding="utf-8"))["questions"]
QUESTIONS = {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in QUESTIONS.items()}

path = sys.argv[1] if len(sys.argv) > 1 else "data/support/test.jsonl"
texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8")][:40]
d = gutcheck.load()
t = time.perf_counter()
results = d.decide_batch(texts, QUESTIONS)
ms = (time.perf_counter() - t) * 1000
for text, r in zip(texts, results):
    a = r["answers"]
    flag = "CHURN" if a["churn_risk"]["noul"] > 0.6 else ""
    print("%-16s urg %.1f %-5s | %s" % (a["intent"]["choice"], a["urgency"]["score"], flag, text[:70]))
print("\n%d messages x %d questions in %.0f ms on %s (%.1f ms per decision)"
      % (len(texts), len(QUESTIONS), ms, d.device, ms / (len(texts) * len(QUESTIONS))))
