"""Task specs, labelled datasets and evaluation metrics (no torch needed)."""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..spec import Question, parse_question


class Task:
    """A named set of questions plus how to read state and labels from dataset rows.

    {
      "name": "support-triage",
      "state": "text",                      # column holding the state, or a list of columns -> dict
      "questions": {
        "intent": {"type": "choice", "instructions": "...", "criteria": {...}, "label": "intent"},
        "urgency": {"type": "score", ..., "label": "urgency"},     # int level, or a level name
        "churn": {"type": "noul", ..., "label": "churn_risk"}      # bool, 0/1, "yes"/"no", or a probability
      }
    }
    A question without "label" is still asked but not trained or scored.
    """

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.name = spec.get("name", "task")
        self.state = spec.get("state", "text")
        self.question_defs = {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in spec["questions"].items()}
        self.labels = {k: v.get("label") for k, v in spec["questions"].items()}
        self.questions: List[Question] = [parse_question(k, v) for k, v in self.question_defs.items()]

    @classmethod
    def load(cls, path: str) -> "Task":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def state_of(self, row: Dict[str, Any]):
        if isinstance(self.state, list):
            return {c: row.get(c) for c in self.state}
        if self.state == "*":
            return {k: v for k, v in row.items() if k not in set(self.labels.values())}
        return row[self.state]

    def target(self, q: Question, row: Dict[str, Any]) -> Optional[np.ndarray]:
        """Target distribution over q's options, or None if the row has no label for q."""
        col = self.labels.get(q.qid)
        if not col or col not in row or row[col] is None or row[col] == "":
            return None
        v = row[col]
        k = len(q.keys)
        t = np.zeros(k, np.float32)
        if q.type == "noul":
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "yes", "y", "1"):
                    v = 1.0
                elif s in ("false", "no", "n", "0"):
                    v = 0.0
                else:
                    v = float(s)
            p = float(v)
            t[:] = [1.0 - p, p]
            return t
        if q.type == "score":
            if isinstance(v, str) and not v.strip().lstrip("-").isdigit():
                levels = [str(x) for x in q.legend.values()]
                v = levels.index(v)
            t[int(v)] = 1.0
            return t
        if isinstance(v, dict):  # soft label {label: prob}
            for i, key in enumerate(q.keys):
                t[i] = float(v.get(key, 0.0))
            return t / max(t.sum(), 1e-9)
        t[q.keys.index(str(v))] = 1.0
        return t


def read_rows(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8") as f:
        if ext in (".jsonl", ".ndjson"):
            return [json.loads(l) for l in f if l.strip()]
        if ext == ".json":
            data = json.load(f)
            return data if isinstance(data, list) else data["rows"]
        if ext in (".csv", ".tsv"):
            return list(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ","))
    raise ValueError("unsupported dataset format %r (use .jsonl, .json, .csv or .tsv)" % ext)


# ------------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------------

def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def macro_f1(y: List[int], p: List[int], k: int) -> float:
    f1s = []
    for c in range(k):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def score_predictions(task: Task, rows: List[Dict], probs: Dict[str, List[Optional[np.ndarray]]]) -> Dict[str, Dict[str, float]]:
    """probs[qid][i] = predicted distribution for row i (None if not predicted)."""
    report = {}
    for q in task.questions:
        ys, ps, confs, correct, briers, mae = [], [], [], [], [], []
        for row, p in zip(rows, probs[q.qid]):
            t = task.target(q, row)
            if t is None or p is None:
                continue
            y = int(t.argmax())
            yhat = int(np.argmax(p))
            ys.append(y)
            ps.append(yhat)
            confs.append(float(p.max()))
            correct.append(float(y == yhat))
            briers.append(float(((p - t) ** 2).sum()))
            if q.type == "score":
                mae.append(abs(float((np.arange(len(p)) * p).sum()) - y))
        if not ys:
            continue
        r = {"n": len(ys), "accuracy": round(float(np.mean(correct)), 4),
             "macro_f1": round(macro_f1(ys, ps, len(q.keys)), 4),
             "ece": round(ece(np.array(confs), np.array(correct)), 4),
             "brier": round(float(np.mean(briers)), 4)}
        if q.type == "score":
            r["mae"] = round(float(np.mean(mae)), 4)
        majority = max(np.bincount(ys, minlength=len(q.keys))) / len(ys)
        r["majority_baseline"] = round(float(majority), 4)
        report[q.qid] = r
    return report


def evaluate_decider(decider, task: Task, rows: List[Dict], batch: int = 16) -> Dict[str, Dict[str, float]]:
    """Run a Decider over labelled rows and score it."""
    probs: Dict[str, List[Optional[np.ndarray]]] = {q.qid: [] for q in task.questions}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        res = decider.decide_batch([task.state_of(r) for r in chunk], task.question_defs)
        for r in res:
            for q in task.questions:
                a = r["answers"][q.qid]
                if q.type == "noul":
                    probs[q.qid].append(np.array([1 - a["noul"], a["noul"]]))
                else:
                    probs[q.qid].append(np.array([a["probabilities"][k] for k in q.keys]))
    return score_predictions(task, rows, probs)


def format_report(report: Dict[str, Dict[str, float]]) -> str:
    lines = ["%-14s %5s %9s %9s %7s %7s %9s" % ("question", "n", "accuracy", "macro_f1", "ece", "brier", "majority")]
    for qid, r in report.items():
        lines.append("%-14s %5d %9.3f %9.3f %7.3f %7.3f %9.3f" % (
            qid[:14], r["n"], r["accuracy"], r["macro_f1"], r["ece"], r["brier"], r["majority_baseline"]))
    return "\n".join(lines)
