"""ONNX Runtime backend: NVIDIA (CUDA/TensorRT), any Windows GPU (DirectML), Qualcomm NPU (QNN),
Apple (CoreML), or CPU. Same interface as OpenVINOBackend; uses `decider.onnx` from the package
(`gutcheck pull <model> --onnx`).

Shapes are dynamic for GPU providers; QNN needs static shapes, so we pad to the same (batch, seq) buckets.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .openvino_backend import DEFAULT_BATCH_BUCKETS, DEFAULT_SEQ_BUCKETS, MAX_MARKERS, OpenVINOBackend, Row

PROVIDER_PREFERENCE = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "QNNExecutionProvider",
                       "CoreMLExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider",
                       "CPUExecutionProvider")
ALIASES = {"CUDA": "CUDAExecutionProvider", "TENSORRT": "TensorrtExecutionProvider", "DML": "DmlExecutionProvider",
           "DIRECTML": "DmlExecutionProvider", "QNN": "QNNExecutionProvider", "COREML": "CoreMLExecutionProvider",
           "ROCM": "ROCMExecutionProvider", "ORT-CPU": "CPUExecutionProvider"}


def available_providers() -> List[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def pick_provider(pref: Optional[str] = None) -> str:
    have = available_providers()
    if pref:
        p = ALIASES.get(pref.upper(), pref)
        if p not in have:
            raise RuntimeError("ONNX Runtime provider %s not available; have %s" % (p, have))
        return p
    for p in PROVIDER_PREFERENCE:
        if p in have:
            return p
    raise RuntimeError("onnxruntime is not installed")


class OnnxBackend:
    name = "onnxruntime"

    def __init__(self, package_dir: str, provider: Optional[str] = None, max_len: int = 512,
                 seq_buckets: Sequence[int] = DEFAULT_SEQ_BUCKETS, batch_buckets: Sequence[int] = DEFAULT_BATCH_BUCKETS,
                 head_xml: Optional[str] = None):
        import onnxruntime as ort

        if head_xml:
            raise NotImplementedError("fine-tuned heads currently run on the OpenVINO backend")
        path = os.path.join(package_dir, "decider.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError("%s has no decider.onnx; run `gutcheck pull <model> --onnx`" % package_dir)
        self.provider = pick_provider(provider)
        self.device = self.provider.replace("ExecutionProvider", "")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=[self.provider, "CPUExecutionProvider"])
        self.static = self.provider == "QNNExecutionProvider"
        self.seq_buckets = sorted({b for b in seq_buckets if b <= max_len} | {max_len})
        self.batch_buckets = sorted(set(batch_buckets))
        self.outputs = [o.name for o in self.sess.get_outputs()]

    seq_bucket = OpenVINOBackend.seq_bucket
    batch_bucket = OpenVINOBackend.batch_bucket
    _chunks = OpenVINOBackend._chunks
    _pack = staticmethod(OpenVINOBackend._pack)

    def warmup(self, shapes=None):
        self.run([([1, 5, 6, 2], [1, 2], 2)], 0)

    def run(self, rows: Sequence[Row], pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
        results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for chunk, B, L in self._chunks(rows):
            if not self.static:  # dynamic providers: no need to pad to the bucket
                B = len(chunk)
                L = max(len(rows[i][0]) for i in chunk)
            ids, att, mp, mm, qt = self._pack(rows, chunk, B, L, pad_id)
            lg, ac = self.sess.run(["logits", "act_logits"], {"input_ids": ids, "attention_mask": att, "marker_pos": mp,
                                                             "marker_mask": mm, "qtype": qt})[:2]
            for r, idx in enumerate(chunk):
                results[idx] = (lg[r], ac[r])
        return (np.stack([results[i][0] for i in range(len(rows))]),
                np.stack([results[i][1] for i in range(len(rows))]))

    def features(self, rows, pad_id, dtype=np.float16):
        if "hidden" not in self.outputs:
            raise RuntimeError("decider.onnx has no hidden output")
        out = {}
        for chunk, _, _ in self._chunks(rows):
            B, L = len(chunk), max(len(rows[i][0]) for i in chunk)
            ids, att, mp, mm, qt = self._pack(rows, chunk, B, L, pad_id)
            hid = self.sess.run(["hidden"], {"input_ids": ids, "attention_mask": att, "marker_pos": mp,
                                             "marker_mask": mm, "qtype": qt})[0]
            for r, idx in enumerate(chunk):
                out[idx] = hid[r, :len(rows[idx][0])].astype(dtype)
        return [out[i] for i in range(len(rows))]
