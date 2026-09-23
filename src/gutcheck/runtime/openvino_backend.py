"""OpenVINO backend: one IR, reshaped to static (batch, seq) buckets per device, compiled lazily and cached.

From the package's `decider.xml` (outputs: logits, act_logits, hidden) we derive, in memory:
  * fused   - encoder + base head -> logits      (base models; fastest)
  * encoder - encoder only        -> hidden      (fine-tuned heads, feature extraction for training)
A fine-tuned package adds `head.xml`, run on the same device after the encoder.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_SEQ_BUCKETS = (64, 128, 256, 512, 1024)
MAX_DYNAMIC_BATCH = 16
DEFAULT_BATCH_BUCKETS = (1, 4)
MAX_MARKERS = 64

Row = Tuple[List[int], List[int], int]  # (ids, markers, qtype)


def cache_root() -> str:
    return os.environ.get("GUTCHECK_HOME", os.path.join(os.path.expanduser("~"), ".cache", "gutcheck"))


def available_devices() -> List[str]:
    import openvino as ov

    return list(ov.Core().available_devices)


def device_names() -> Dict[str, str]:
    import openvino as ov

    core = ov.Core()
    out = {}
    for d in core.available_devices:
        try:
            out[d] = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            out[d] = d
    return out


class OpenVINOBackend:
    name = "openvino"

    def __init__(self, package_dir: str, device: str = "AUTO", max_len: int = 512,
                 seq_buckets: Sequence[int] = DEFAULT_SEQ_BUCKETS,
                 batch_buckets: Sequence[int] = DEFAULT_BATCH_BUCKETS,
                 head_xml: Optional[str] = None, lower_xml: Optional[str] = None, cache_dir: Optional[str] = None):
        import openvino as ov

        self.ov = ov
        self.core = ov.Core()
        self.cache_dir = cache_dir or os.path.join(cache_root(), "ov_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.core.set_property({"CACHE_DIR": self.cache_dir})
        self.device = device
        full = self.core.read_model(os.path.join(package_dir, "decider.xml"))
        names = {o.get_any_name() for o in full.outputs}
        params = {p.get_output_tensor(0).get_any_name(): p for p in full.get_parameters()}
        if "hidden" in names:
            self.fused = ov.Model([full.output("logits"), full.output("act_logits")], full.get_parameters(), "fused")
            self.encoder = ov.Model([full.output("hidden")], [params["input_ids"], params["attention_mask"]], "encoder")
        else:  # packages exported before the hidden output existed
            self.fused, self.encoder = full, None
        if lower_xml:  # deep adapters: shared lower layers (-> "mid") instead of the full encoder
            self.encoder = self.core.read_model(lower_xml)
        self.head = self.core.read_model(head_xml) if head_xml else None
        if self.head is not None and self.encoder is None:
            raise RuntimeError("this base package has no 'hidden' output; re-export it to use fine-tuned heads")
        # The NPU needs static shapes, so it pads to (batch, seq) buckets. CPU and GPU run exact dynamic shapes:
        # measured on Lunar Lake, padding 60-token prompts to a 128 bucket cost 2.4x on CPU and 1.5x on GPU.
        self.dynamic = not device.upper().startswith("NPU")
        self.seq_buckets = sorted({b for b in seq_buckets if b <= max_len} | {max_len})
        self.batch_buckets = sorted(set(batch_buckets))
        self._compiled: Dict[Tuple[str, int, int], object] = {}
        self._lock = threading.Lock()
        self.compile_log: List[Tuple[Tuple[str, int, int], float]] = []

    # --------------------------------------------------------------------------------------------
    def _config(self) -> Dict[str, str]:
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if self.device.startswith("CPU"):
            cfg["INFERENCE_PRECISION_HINT"] = "f32"
        return cfg

    def _get(self, kind: str, b: int, l: int):
        key = (kind, b, l)
        cm = self._compiled.get(key)
        if cm is not None:
            return cm
        with self._lock:
            cm = self._compiled.get(key)
            if cm is not None:
                return cm
            src = {"fused": self.fused, "encoder": self.encoder, "head": self.head}[kind]
            m = src.clone()
            k = MAX_MARKERS if b > 0 else -1
            shapes = {"input_ids": [b, l], "attention_mask": [b, l], "marker_pos": [b, k],
                      "marker_mask": [b, k], "qtype": [b], "hidden": [b, l, -1]}
            m.reshape({n: shapes[n] for n in (p.get_output_tensor(0).get_any_name() for p in m.get_parameters())})
            if kind == "head":  # hidden's last dim is fixed by the weights
                d = self.head.input("hidden").get_partial_shape()[2]
                m.reshape({"hidden": [b, l, d.get_length()]})
            t = time.perf_counter()
            cm = self.core.compile_model(m, self.device, self._config())
            self.compile_log.append((key, time.perf_counter() - t))
            self._compiled[key] = cm
            return cm

    def warmup(self, shapes: Optional[Sequence[Tuple[int, int]]] = None):
        if self.dynamic:
            shapes = [(-1, -1)]
        for b, l in shapes or [(1, self.seq_buckets[0])]:
            if self.head is None:
                self._get("fused", b, l)
            else:
                self._get("encoder", b, l)
                self._get("head", b, l)

    def seq_bucket(self, n: int) -> int:
        for b in self.seq_buckets:
            if n <= b:
                return b
        return self.seq_buckets[-1]

    def batch_bucket(self, n: int) -> int:
        for b in self.batch_buckets:
            if n <= b:
                return b
        return self.batch_buckets[-1]

    # --------------------------------------------------------------------------------------------
    def _chunks(self, rows: Sequence[Row]):
        """Group rows by length so a short question is not padded to a long neighbour.

        Yields (row indices, batch, seq) to pack to; for dynamic devices the compiled model is the (-1, -1) one.
        """
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
        if self.dynamic:
            for i in range(0, len(order), MAX_DYNAMIC_BATCH):
                chunk = order[i:i + MAX_DYNAMIC_BATCH]
                yield chunk, len(chunk), min(max(len(rows[j][0]) for j in chunk), self.seq_buckets[-1])
            return
        # static (NPU): fill each call up to the largest batch bucket, then pad to the bucket that fits the
        # longest row. One padded call is cheaper than splitting rows across several calls.
        max_b = self.batch_buckets[-1]
        for i in range(0, len(order), max_b):
            chunk = order[i:i + max_b]
            yield chunk, self.batch_bucket(len(chunk)), self.seq_bucket(max(len(rows[j][0]) for j in chunk))

    @staticmethod
    def _pack(rows: Sequence[Row], chunk: List[int], B: int, L: int, pad_id: int, K: int = MAX_MARKERS):
        ids = np.full((B, L), pad_id, np.int64)
        att = np.zeros((B, L), np.int64)
        mp = np.zeros((B, K), np.int64)
        mm = np.zeros((B, K), bool)
        mm[:, :2] = True  # padding rows keep >=2 valid markers so the act head's top-2 stays finite
        qt = np.zeros((B,), np.int64)
        for r, idx in enumerate(chunk):
            seq, markers, q = rows[idx]
            seq = seq[:L]
            ids[r, :len(seq)] = seq
            att[r, :len(seq)] = 1
            mm[r, :] = False
            mp[r, :len(markers)] = markers
            mm[r, :len(markers)] = True
            qt[r] = q
        return ids, att, mp, mm, qt

    def run(self, rows: Sequence[Row], pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (logits[n, MAX_MARKERS], act_logits[n, A])."""
        results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for chunk, B, L in self._chunks(rows):
            K = max(2, max(len(rows[i][1]) for i in chunk)) if self.dynamic else MAX_MARKERS
            ids, att, mp, mm, qt = self._pack(rows, chunk, B, L, pad_id, K)
            cb, cl = (-1, -1) if self.dynamic else (B, L)
            if self.head is None:
                res = self._get("fused", cb, cl)([ids, att, mp, mm, qt])
                lg, ac = res[0], res[1]
            else:
                hid = self._get("encoder", cb, cl)([ids, att])[0]
                res = self._get("head", cb, cl)({"hidden": hid, "attention_mask": att, "marker_pos": mp,
                                                 "marker_mask": mm, "qtype": qt})
                lg, ac = res["logits"], res["act_logits"]
            if lg.shape[1] < MAX_MARKERS:  # dynamic marker count: pad rows to a common width
                lg = np.concatenate([lg, np.full((lg.shape[0], MAX_MARKERS - lg.shape[1]), -1e4, lg.dtype)], 1)
            for r, idx in enumerate(chunk):
                results[idx] = (lg[r], ac[r])
        return (np.stack([results[i][0] for i in range(len(rows))]),
                np.stack([results[i][1] for i in range(len(rows))]))

    def features(self, rows: Sequence[Row], pad_id: int, dtype=np.float16) -> List[np.ndarray]:
        """Encoder hidden states per row, trimmed to the row's real length."""
        if self.encoder is None:
            raise RuntimeError("package has no 'hidden' output; re-export it")
        out: Dict[int, np.ndarray] = {}
        for chunk, B, L in self._chunks(rows):
            ids, att, _, _, _ = self._pack(rows, chunk, B, L, pad_id)
            cb, cl = (-1, -1) if self.dynamic else (B, L)
            hid = self._get("encoder", cb, cl)([ids, att])[0]
            for r, idx in enumerate(chunk):
                n = min(len(rows[idx][0]), L)
                out[idx] = hid[r, :n].astype(dtype)
        return [out[i] for i in range(len(rows))]
