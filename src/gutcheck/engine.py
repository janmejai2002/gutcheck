"""The Decider: typed System-1 decisions on your NPU / GPU / CPU."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .spec import Calibration, Question, State, build_sequence, decode_answer, parse_questions, serialize_state
from .tokenizer import Tok

# Measured on an Intel Core Ultra 7 256V: GPU is fastest per call, NPU is close behind at a fraction
# of the power and leaves the GPU free. CPU is a correct but slow fallback.
DEVICE_PREFERENCE = ("GPU", "NPU", "CPU")


def pick_device(pref: str = "auto") -> str:
    from .runtime.openvino_backend import available_devices

    devs = available_devices()
    pref = (pref or "auto").upper()
    if pref != "AUTO":
        base = pref.split(".")[0]
        if base not in {d.split(".")[0] for d in devs}:
            raise RuntimeError("device %s not available; found %s" % (pref, ", ".join(devs)))
        return pref
    env = os.environ.get("GUTCHECK_DEVICE")
    if env:
        return pick_device(env)
    for d in DEVICE_PREFERENCE:
        for have in devs:
            if have.split(".")[0] == d:
                return have
    return "CPU"


ORT_DEVICES = {"CUDA", "TENSORRT", "DML", "DIRECTML", "QNN", "COREML", "ROCM", "ORT-CPU"}


def resolve_backend(device: str = "auto"):
    """-> ("openvino", OV device) or ("onnx", provider alias).

    auto: Intel GPU/NPU via OpenVINO if present, else an accelerated ONNX Runtime provider
    (CUDA, DirectML, QNN, CoreML), else OpenVINO on the CPU.
    """
    d = (device or "auto").upper()
    env = os.environ.get("GUTCHECK_DEVICE")
    if d == "AUTO" and env:
        d = env.upper()
    if d in ORT_DEVICES:
        return "onnx", d
    if d != "AUTO":
        return "openvino", pick_device(d)
    ov_dev = pick_device("auto")
    if ov_dev.split(".")[0] in ("GPU", "NPU"):
        return "openvino", ov_dev
    from .runtime.onnx_backend import available_providers

    # DirectML is opt-in only (device="DML"): its Reshape rejects our dynamic-shape graph on the builds
    # we have tested (onnxruntime-directml 1.24), so it must never be picked silently.
    for prov, alias in (("TensorrtExecutionProvider", "TENSORRT"), ("CUDAExecutionProvider", "CUDA"),
                        ("QNNExecutionProvider", "QNN"), ("CoreMLExecutionProvider", "COREML"),
                        ("ROCMExecutionProvider", "ROCM")):
        if prov in available_providers():
            return "onnx", alias
    return "openvino", ov_dev


class Decider:
    """Load a decision model and answer typed questions about any state.

    >>> d = Decider()                                   # laya-en on the best local accelerator
    >>> d.decide("I was billed twice, refund me or I cancel", {
    ...     "intent": {"type": "choice", "instructions": "What does the customer want?",
    ...                "criteria": {"refund": "money back", "cancel": "close account", "info": "a question"}},
    ...     "angry": {"type": "noul", "instructions": "Is the customer angry?"}})
    """

    def __init__(self, model: str = "laya-en", device: str = "auto", weights: str = "fp16",
                 batch_buckets: Sequence[int] = (1, 4), verbose: bool = True):
        from . import hub
        from .runtime.openvino_backend import OpenVINOBackend

        self.package_dir = hub.resolve(model, weights=weights, verbose=verbose)
        with open(os.path.join(self.package_dir, "gutcheck.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.name = self.manifest.get("name", model)
        head_xml, lower_xml, base_dir = None, None, self.package_dir
        # a fine-tuned head package runs its small head on top of the base model's shared encoder
        self.default_questions: Optional[Dict[str, Dict[str, Any]]] = None
        if self.manifest.get("format") == "gutcheck-head/1":
            head_xml = os.path.join(self.package_dir, "head.xml")
            base_dir = self.manifest.get("base_path")
            if not base_dir or not hub.is_package(base_dir):
                base_dir = hub.resolve(self.manifest["base"], verbose=verbose)
            task = self.manifest.get("task") or {}
            self.default_questions = {k: {kk: vv for kk, vv in v.items() if kk != "label"}
                                      for k, v in task.get("questions", {}).items()} or None
            depth = int(self.manifest.get("depth", 0) or 0)
            if depth:
                lower_xml = os.path.join(base_dir, "lower_d%d.xml" % depth)
                if not os.path.exists(lower_xml):
                    raise FileNotFoundError("%s needs %s; rebuild it with `gutcheck train --depth %d`"
                                            % (self.name, lower_xml, depth))
        self.base_dir = base_dir
        self.max_len = int(self.manifest.get("max_len", 512))
        self.head_max_len = int(self.manifest.get("head_max_len", 192))
        self.calibration = Calibration.from_config(self.manifest.get("calibration", {}))
        self._load_user_calibration()
        self.tok = Tok(os.path.join(base_dir, "tokenizer"))
        kind, dev = resolve_backend(device)
        if kind == "onnx":
            from .runtime.onnx_backend import OnnxBackend

            if not os.path.exists(os.path.join(base_dir, "decider.onnx")):
                hub.ensure_onnx(base_dir, verbose=verbose)
            self.backend = OnnxBackend(base_dir, dev, max_len=self.max_len, batch_buckets=batch_buckets,
                                       head_xml=head_xml)
            self.device = self.backend.device
        else:
            self.device = dev
            self.backend = OpenVINOBackend(base_dir, self.device, max_len=self.max_len,
                                           batch_buckets=batch_buckets, head_xml=head_xml, lower_xml=lower_xml)
        self._lock = threading.Lock()

    # --------------------------------------------------------------------------------------------
    def _load_user_calibration(self):
        p = os.path.join(self.package_dir, "calibration.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                self.calibration = Calibration.from_config(json.load(f))

    def warmup(self, seq_lens: Sequence[int] = (128,), batch: int = 1) -> "Decider":
        """Compile the given buckets now (first NPU compile ~30s, then cached on disk)."""
        self.backend.warmup([(batch, self.backend.seq_bucket(s)) for s in seq_lens])
        return self

    def encode(self, state: State, questions: List[Question]):
        text = serialize_state(state).replace(self.tok.mask_token, " ")
        st_ids = self.tok.encode(text)
        left = isinstance(state, list)  # conversations: keep the newest turns
        return [build_sequence(self.tok, state, q, self.max_len, self.head_max_len, left, st_ids) for q in questions]

    def decide_batch(self, states: Sequence[State], questions: Optional[Dict[str, Dict[str, Any]]] = None
                     ) -> List[Dict[str, Any]]:
        """Same questions over many states. Returns one Jev-shaped response per state.

        `questions` may be omitted for a fine-tuned model: it answers the questions it was trained on.
        """
        if isinstance(states, (str, dict)):
            raise TypeError("decide_batch takes a list of states; use decide() for one")
        if questions is None:
            if self.default_questions is None:
                raise TypeError("questions are required (only fine-tuned models have default questions)")
            questions = self.default_questions
        qs = parse_questions(questions)
        t0 = time.perf_counter()
        rows, per_state = [], []
        for st in states:
            enc = self.encode(st, qs)
            per_state.append(enc)
            rows.extend((e.ids, e.markers, e.qtype) for e in enc)
        if rows:
            with self._lock:
                logits, act = self.backend.run(rows, self.tok.pad_token_id)
        ms = (time.perf_counter() - t0) * 1000.0
        out, r = [], 0
        for enc in per_state:
            answers = {}
            for q, e in zip(qs, enc):
                # Laya's act head emits logits in the thousands; fp16 devices can overflow them to
                # inf/NaN. It is an optional extra (Jev has no such field), so drop it rather than lie.
                act_p = None
                if np.all(np.isfinite(act[r])):
                    a = act[r] - act[r].max()
                    act_p = float(np.exp(a[0]) / np.exp(a).sum())
                answers[q.qid] = decode_answer(q, logits[r, :len(e.markers)], self.calibration, act_p)
                r += 1
            out.append({
                "model": self.name,
                "answers": answers,
                "usage": {"input_tokens": sum(len(e.ids) for e in enc), "output_tokens": 0},
                "device": self.device,
                "latency_ms": round(ms / max(1, len(states)), 2),
            })
        return out

    def decide(self, state: State, questions: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Answer typed questions about one state. Jev `/v1/systemone` response shape."""
        return self.decide_batch([state], questions)[0]

    # conveniences -------------------------------------------------------------------------------
    def choice(self, state: State, instructions: str, options: Union[Dict[str, Optional[str]], List[str]]) -> Dict:
        return self.decide(state, {"q": {"type": "choice", "instructions": instructions, "criteria": options}})["answers"]["q"]

    def score(self, state: State, instructions: str, levels: List[str]) -> Dict:
        return self.decide(state, {"q": {"type": "score", "instructions": instructions, "criteria": levels}})["answers"]["q"]

    def noul(self, state: State, instructions: str, true: Optional[str] = None, false: Optional[str] = None) -> float:
        crit = {k: v for k, v in (("true", true), ("false", false)) if v}
        q = {"type": "noul", "instructions": instructions}
        if crit:
            q["criteria"] = crit
        return self.decide(state, {"q": q})["answers"]["q"]["noul"]

    predict = decide
    system_one = decide
