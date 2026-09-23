"""Small sentence-embedding model on OpenVINO (NPU/GPU/CPU): the coarse "shortlist" stage.

Default: OpenVINO/bge-base-en-v1.5-int8-ov (MIT, 110 MB, pre-converted - no PyTorch needed).
Vectors for catalogs are cached on disk by content hash, so a catalog is embedded once.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, List, Optional, Sequence

import numpy as np

from .runtime.openvino_backend import cache_root

EMBEDDERS = {
    "bge-base-en": {"repo": "OpenVINO/bge-base-en-v1.5-int8-ov", "revision": "9a76c2f2cf23bb2fd4a20469a5170cb7ebe0d5d2",
                    "pooling": "cls", "query_prefix": "Represent this sentence for searching relevant passages: ",
                    "max_len": 512},
    "bge-base-en-fp16": {"repo": "OpenVINO/bge-base-en-v1.5-fp16-ov", "revision": None, "pooling": "cls",
                         "query_prefix": "Represent this sentence for searching relevant passages: ", "max_len": 512},
}
FILES = ["openvino_model.xml", "openvino_model.bin", "tokenizer.json", "tokenizer_config.json", "config.json"]
SEQ_BUCKETS = (32, 64, 128, 256, 512)


def ensure_embedder(name: str) -> str:
    """Install the embedder into gutcheck's own cache (like model packages), so after the first download it
    works offline and does not depend on where - or whether - the Hugging Face cache lives."""
    spec = EMBEDDERS[name]
    path = os.path.join(cache_root(), "embedders", name)
    if all(os.path.exists(os.path.join(path, f)) for f in ("openvino_model.xml", "openvino_model.bin", "tokenizer.json")):
        return path
    from .hub import download_with_retry

    download_with_retry(spec["repo"], revision=spec["revision"], allow_patterns=FILES, local_dir=path)
    return path
BATCH_BUCKETS = (1, 8, 32)


class Embedder:
    def __init__(self, name: str = "bge-base-en", device: str = "auto"):
        import openvino as ov
        from tokenizers import Tokenizer

        from .engine import pick_device

        spec = EMBEDDERS[name]
        self.name, self.spec = name, spec
        path = ensure_embedder(name)
        self.tk = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.tk.no_padding()
        self.tk.no_truncation()
        self.core = ov.Core()
        self.core.set_property({"CACHE_DIR": os.path.join(cache_root(), "ov_cache")})
        self.model = self.core.read_model(os.path.join(path, "openvino_model.xml"))
        self.inputs = [i.get_any_name() for i in self.model.inputs]
        self.device = pick_device(device)
        self.pad_id = self.tk.token_to_id("[PAD]") or 0
        self._compiled = {}
        self._lock = threading.Lock()
        self._cache_file = os.path.join(cache_root(), "embeddings", "%s.npz" % name)
        self._mem: Dict[str, np.ndarray] = {}
        self._load_cache()

    # --------------------------------------------------------------------------------------------
    def _get(self, b: int, l: int):
        key = (b, l)
        if key not in self._compiled:
            with self._lock:
                if key not in self._compiled:
                    m = self.model.clone()
                    m.reshape({n: [b, l] for n in self.inputs})
                    self._compiled[key] = self.core.compile_model(m, self.device, {"PERFORMANCE_HINT": "LATENCY"})
        return self._compiled[key]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        max_len = self.spec["max_len"]
        encs = [e.ids[: max_len - 1] + ([e.ids[-1]] if len(e.ids) > max_len else [])
                for e in self.tk.encode_batch(list(texts), add_special_tokens=True)]
        out = np.zeros((len(texts), 0), np.float32)
        order = sorted(range(len(encs)), key=lambda i: len(encs[i]))
        res: Dict[int, np.ndarray] = {}
        i = 0
        while i < len(order):
            L = next((b for b in SEQ_BUCKETS if len(encs[order[i]]) <= b), SEQ_BUCKETS[-1])
            j = i
            while j < len(order) and j - i < BATCH_BUCKETS[-1] and len(encs[order[j]]) <= L:
                j += 1
            chunk = order[i:j]
            B = next(b for b in BATCH_BUCKETS if len(chunk) <= b)
            ids = np.full((B, L), self.pad_id, np.int64)
            att = np.zeros((B, L), np.int64)
            for r, idx in enumerate(chunk):
                e = encs[idx][:L]
                ids[r, :len(e)] = e
                att[r, :len(e)] = 1
            feed = {"input_ids": ids, "attention_mask": att}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self._get(B, L)(feed)[0]
            if self.spec["pooling"] == "cls":
                v = h[:, 0]
            else:
                m = att[:, :, None].astype(np.float32)
                v = (h * m).sum(1) / np.maximum(m.sum(1), 1)
            v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
            for r, idx in enumerate(chunk):
                res[idx] = v[r].astype(np.float32)
            i = j
        return np.stack([res[k] for k in range(len(texts))]) if texts else out

    # --------------------------------------------------------------------------------------------
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.spec["query_prefix"] + t for t in texts])

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray:
        """Documents are cached by content hash (catalog descriptions rarely change)."""
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._mem]
        if missing:
            uniq = dict(missing)
            vecs = self._encode(list(uniq.values()))
            for k, v in zip(uniq.keys(), vecs):
                self._mem[k] = v
            self._save_cache()
        return np.stack([self._mem[k] for k in keys]) if keys else np.zeros((0, 768), np.float32)

    def _load_cache(self):
        try:
            with np.load(self._cache_file) as z:
                self._mem = {k: z[k] for k in z.files}
        except Exception:
            self._mem = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            tmp = self._cache_file + ".tmp.npz"
            np.savez(tmp, **self._mem)
            os.replace(tmp, self._cache_file)
        except Exception:
            pass
