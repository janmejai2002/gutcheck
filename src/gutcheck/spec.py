"""Question schema, prompt construction and answer decoding.

The wire format matches TypeSafe's Jev `/v1/systemone` API: a `state` (string, object or array) and a
map of typed `questions` (`choice`, `score`, `noul`). The token layout reproduces Laya's exactly
(https://github.com/NandhaKishorM/laya, Apache-2.0) so checkpoints trained with Laya keep their
accuracy here:

    [CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]

The decision head scores the hidden state at each [MASK] marker; the softmax over markers is the
answer distribution.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

State = Union[str, dict, list]


class QuestionError(ValueError):
    """A question definition cannot be answered. The message names the question and the fix."""


# --------------------------------------------------------------------------------------------
# Validation / normalisation
# --------------------------------------------------------------------------------------------

@dataclass
class Question:
    """A validated, normalised question."""

    qid: str
    type: str
    instructions: str
    keys: List[str]                  # choice: labels; score: "0".."n-1"; noul: ["false", "true"]
    options: List[str]               # rendered option texts, in key order
    legend: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def qtype(self) -> int:
        return QTYPES[self.type]


def serialize_state(state: State) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _noul_labels(labels: Any) -> tuple:
    if labels is None:
        return "false", "true"
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("labels must map exactly 'false' and 'true' to distinct non-empty strings")
    f, t = labels["false"], labels["true"]
    if not isinstance(f, str) or not isinstance(t, str) or not f.strip() or not t.strip() or f.strip() == t.strip():
        raise ValueError("labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return f.strip(), t.strip()


def parse_question(qid: str, qdef: Any, max_options: int = 64) -> Question:
    """Validate one question definition and render its option texts (Laya-compatible)."""
    if not isinstance(qdef, dict):
        raise QuestionError("question %r: definition must be an object, got %s" % (qid, type(qdef).__name__))
    t = qdef.get("type")
    if t not in QTYPES:
        raise QuestionError("question %r: unknown type %r; use one of choice, score, noul" % (qid, t))
    if "instructions" not in qdef or qdef["instructions"] in (None, ""):
        raise QuestionError("question %r: missing 'instructions' (the question the model should answer)" % (qid,))
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    crit = qdef.get("criteria")

    if t == "choice":
        if isinstance(crit, list):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not crit:
            raise QuestionError("question %r: choice needs 'criteria' as {label: description} or [labels]" % (qid,))
        if len(crit) > max_options:
            raise QuestionError(
                "question %r: %d options exceeds the limit of %d. Accuracy falls off past ~20 options; "
                "use gutcheck.shortlist (embedding pre-filter) for large label sets." % (qid, len(crit), max_options))
        keys = [str(k) for k in crit]
        options = [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in zip(keys, crit.values())]
        return Question(qid, t, ins, keys, options)

    if t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise QuestionError("question %r: score needs 'criteria' as a list of at least 2 levels, lowest first" % (qid,))
        if len(crit) > max_options:
            raise QuestionError("question %r: too many score levels (%d)" % (qid, len(crit)))
        options = ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
        legend = {str(i): c for i, c in enumerate(crit)}
        return Question(qid, t, ins, [str(i) for i in range(len(crit))], options, legend=legend)

    # noul
    if crit is not None and not isinstance(crit, dict):
        raise QuestionError("question %r: noul 'criteria' must be an object with optional 'true'/'false'" % (qid,))
    crit = {str(k).lower(): v for k, v in (crit or {}).items()}
    try:
        f_label, t_label = _noul_labels(qdef.get("labels"))
    except ValueError as e:
        raise QuestionError("question %r: %s" % (qid, e)) from None
    fc, tc = crit.get("false"), crit.get("true")
    options = [
        f_label + ": " + (render_criterion(fc) if fc not in (None, "") else "no, the statement does not hold"),
        t_label + ": " + (render_criterion(tc) if tc not in (None, "") else "yes, the statement holds"),
    ]
    return Question(qid, t, ins, ["false", "true"], options)


def parse_questions(questions: Any, max_options: int = 64) -> List[Question]:
    if not isinstance(questions, dict):
        raise QuestionError("questions must be an object mapping question id -> definition")
    return [parse_question(str(qid), q, max_options) for qid, q in questions.items()]


# --------------------------------------------------------------------------------------------
# Token sequence construction
# --------------------------------------------------------------------------------------------

@dataclass
class Encoded:
    ids: List[int]
    markers: List[int]
    qtype: int


def build_sequence(tok, state: State, q: Question, max_len: int = 512, head_max_len: int = 192,
                   truncate_left: bool = False, state_ids: Optional[List[int]] = None) -> Encoded:
    """Laya's layout. `tok` is a `gutcheck.tokenizer.Tok`. `state_ids` lets callers tokenise a state once."""
    mask = tok.mask_token
    head_ids = tok.encode("%s question: %s" % (q.type, q.instructions.replace(mask, " ")))
    opt_ids = [[tok.mask_token_id] + tok.encode(" " + o.replace(mask, " "))[:48] for o in q.options]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = state_ids if state_ids is not None else tok.encode(serialize_state(state).replace(mask, " "))
    st = st[max(0, len(st) - room):] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    ids = ids[:max_len]
    markers = [m for m in markers if m < max_len]
    if len(markers) != len(q.options):
        raise QuestionError("question %r: options do not fit in the %d-token head budget" % (q.qid, head_max_len))
    return Encoded(ids, markers, q.qtype)


# --------------------------------------------------------------------------------------------
# Calibration + decoding
# --------------------------------------------------------------------------------------------

TEMP_MIN, TEMP_MAX = 0.5, 5.0


def clamp_temperature(t: Any) -> float:
    """Laya ships some sharpening temperatures (e.g. choice:11+ = 0.10) that turn coin flips into
    certainties. Refuse anything outside [0.5, 5]."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


class Calibration:
    """Per (question type, option-count bucket) temperatures."""

    def __init__(self, temperature: Sequence[float] = (1.0, 1.0, 1.0), by_options: Optional[Dict[str, float]] = None):
        self.temperature = [clamp_temperature(t) for t in temperature]
        self.by_options = {k: clamp_temperature(v) for k, v in (by_options or {}).items()}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Calibration":
        return cls(cfg.get("temperature", [1.0, 1.0, 1.0]), cfg.get("temperature_by_options", {}))

    def to_config(self) -> Dict[str, Any]:
        return {"temperature": list(self.temperature), "temperature_by_options": dict(self.by_options)}

    def get(self, qtype: int, k: int) -> float:
        return self.by_options.get(temp_bucket(qtype, k), self.temperature[qtype])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def entropy_confidence(p: np.ndarray) -> float:
    k = len(p)
    if k < 2:
        return 1.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def decode_answer(q: Question, logits: np.ndarray, calib: Calibration, act_prob: Optional[float] = None) -> Dict[str, Any]:
    """Turn one row of marker logits into a Jev-shaped answer."""
    k = len(q.keys)
    p = softmax(np.asarray(logits[:k], dtype=np.float64) / calib.get(q.qtype, k))
    conf = round(entropy_confidence(p), 4)
    if q.type == "choice":
        ans = {"type": "choice", "choice": q.keys[int(p.argmax())],
               "probabilities": {kk: round(float(v), 4) for kk, v in zip(q.keys, p)}, "confidence": conf}
    elif q.type == "score":
        ans = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4), "legend": q.legend,
               "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)}, "confidence": conf}
    else:
        pt = float(p[1])
        ans = {"type": "noul", "noul": round(pt, 4), "confidence": round(max(pt, 1.0 - pt), 4)}
    if act_prob is not None:
        ans["act_probability"] = round(float(act_prob), 4)
    return ans
