"""Fine-tune the decision head on your labels, on a laptop, in minutes.

The encoder stays frozen and runs once per example on the NPU/GPU (OpenVINO) to cache its hidden
states; only the ~26M-parameter head (2 transformer layers + scorer) trains, in PyTorch on the CPU.
The result is a small adapter package (~50 MB) that reuses the base model's encoder at inference.

Loss: log score (cross-entropy against the label distribution, a strictly proper scoring rule) plus a
ranked probability score for ordinal `score` questions - the same family of rewards Laya's RLCD uses,
so probabilities stay honest. Temperatures are then fitted per (type, option bucket) on held-out data.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..spec import Calibration, QTYPES, clamp_temperature, temp_bucket
from .data import Task, score_predictions


def _log(msg: str):
    print(msg, flush=True)


@dataclass
class Item:
    hidden: np.ndarray      # [n, d] fp16
    markers: List[int]
    qtype: int
    target: np.ndarray      # [k]
    row: int
    qid: str


# ------------------------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------------------------

def extract_items(decider, task: Task, rows: List[Dict], log: Callable = None, batch_rows: int = 32) -> List[Item]:
    """Run the frozen encoder over every (row, labelled question) and keep its hidden states."""
    log = log or _log
    items: List[Item] = []
    t0 = time.perf_counter()
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start:start + batch_rows]
        enc_rows, meta = [], []
        for i, row in enumerate(chunk):
            encs = decider.encode(task.state_of(row), task.questions)
            for q, e in zip(task.questions, encs):
                t = task.target(q, row)
                if t is None:
                    continue
                enc_rows.append((e.ids, e.markers, e.qtype))
                meta.append((start + i, q.qid, e.markers, e.qtype, t))
        if not enc_rows:
            continue
        feats = decider.backend.features(enc_rows, decider.tok.pad_token_id)
        for f, (ri, qid, markers, qt, t) in zip(feats, meta):
            items.append(Item(f, list(markers), qt, t, ri, qid))
        done = min(len(rows), start + batch_rows)
        log("  encoded %d/%d rows (%d items, %.0fs)" % (done, len(rows), len(items), time.perf_counter() - t0))
    return items


# ------------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------------

def build_head(hidden_size: int, head_layers: int, n_act: int, weights_path: Optional[str] = None):
    import torch
    import torch.nn as nn

    from .model import DecisionModel

    class HeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            d = hidden_size
            layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.1, batch_first=True, norm_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))
            self.head_checkpointing = False

        decide = DecisionModel.decide

        def forward(self, hidden, attention_mask, marker_pos, marker_mask, qtype):
            return self.decide(hidden, attention_mask, marker_pos, marker_mask, qtype)

    m = HeadModel()
    if weights_path:
        from safetensors.torch import load_file

        m.load_state_dict(load_file(weights_path), strict=True)
    return m


def _collate(items: Sequence[Item]):
    import torch

    n = len(items)
    L = max(len(it.hidden) for it in items)
    d = items[0].hidden.shape[1]
    K = max(len(it.markers) for it in items)
    h = torch.zeros((n, L, d), dtype=torch.float32)
    att = torch.zeros((n, L), dtype=torch.long)
    mp = torch.zeros((n, max(K, 2)), dtype=torch.long)
    mm = torch.zeros((n, max(K, 2)), dtype=torch.bool)
    tgt = torch.zeros((n, max(K, 2)), dtype=torch.float32)
    qt = torch.zeros((n,), dtype=torch.long)
    for i, it in enumerate(items):
        m = len(it.hidden)
        h[i, :m] = torch.from_numpy(it.hidden.astype(np.float32))
        att[i, :m] = 1
        k = len(it.markers)
        mp[i, :k] = torch.tensor(it.markers)
        mm[i, :k] = True
        tgt[i, :k] = torch.from_numpy(it.target)
        qt[i] = it.qtype
    return h, att, mp, mm, qt, tgt


def _loss(logits, mm, tgt, qt, w_rps: float = 1.0):
    import torch

    logp = torch.log_softmax(logits.masked_fill(~mm, -1e4), -1)
    nll = -(tgt * logp).sum(-1)
    is_score = (qt == QTYPES["score"]).float()
    if is_score.any():
        p = logp.exp() * mm
        k = mm.sum(-1).clamp(min=2).float()
        rps = (((torch.cumsum(p, -1) - torch.cumsum(tgt, -1)) ** 2) * mm).sum(-1) / (k - 1)
        nll = nll + w_rps * rps * is_score
    return nll.mean()


def predict_logits(model, items: Sequence[Item], batch: int = 32) -> List[np.ndarray]:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            h, att, mp, mm, qt, _ = _collate(chunk)
            lg, _ = model(h, att, mp, mm, qt)
            for r, it in enumerate(chunk):
                out.append(lg[r, :len(it.markers)].numpy().astype(np.float64))
    return out


def fit_temperatures(items: Sequence[Item], logits: Sequence[np.ndarray]) -> Calibration:
    """One temperature per (type, option bucket) minimising NLL on held-out items."""
    grid = np.geomspace(0.5, 5.0, 64)
    groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    per_type: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {0: [], 1: [], 2: []}
    for it, z in zip(items, logits):
        groups.setdefault(temp_bucket(it.qtype, len(it.markers)), []).append((z, it.target))
        per_type[it.qtype].append((z, it.target))

    def best(pairs):
        if len(pairs) < 8:
            return None
        nll = []
        for T in grid:
            s = 0.0
            for z, t in pairs:
                zz = z / T
                zz = zz - zz.max()
                s -= float((t * (zz - math.log(np.exp(zz).sum()))).sum())
            nll.append(s)
        return float(grid[int(np.argmin(nll))])

    temps = [best(per_type[i]) or 1.0 for i in range(3)]
    by = {k: v for k, v in ((k, best(p)) for k, p in groups.items()) if v is not None}
    return Calibration(temps, by)


# ------------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------------

def train_head(model, train_items: List[Item], val_items: List[Item], epochs: int = 4, lr: float = 3e-4,
               batch: int = 16, seed: int = 0, log: Callable = None, threads: Optional[int] = None):
    log = log or _log
    import torch

    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train_items) / batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))

    def val_loss():
        if not val_items:
            return float("nan")
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_items), 32):
                h, att, mp, mm, qt, tgt = _collate(val_items[i:i + 32])
                lg, _ = model(h, att, mp, mm, qt)
                tot += float(_loss(lg, mm, tgt, qt)) * len(tgt)
                n += len(tgt)
        return tot / max(1, n)

    best_state, best_val = None, val_loss()
    log("  epoch 0: val loss %.4f" % best_val)
    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        # bucket by length so padding stays small, then shuffle the buckets
        idx = sorted(range(len(train_items)), key=lambda i: len(train_items[i].hidden) + rng.random() * 8)
        batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
        rng.shuffle(batches)
        t0, tl = time.perf_counter(), 0.0
        for b in batches:
            h, att, mp, mm, qt, tgt = _collate([train_items[i] for i in b])
            lg, _ = model(h, att, mp, mm, qt)
            loss = _loss(lg, mm, tgt, qt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tl += loss.item()
        v = val_loss()
        log("  epoch %d: train loss %.4f | val loss %.4f | %.0fs" % (ep, tl / max(1, len(batches)), v, time.perf_counter() - t0))
        if not val_items or v < best_val:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def export_head(model, out_dir: str, hidden_size: int):
    """Trace the head to OpenVINO IR (inputs: hidden, attention_mask, marker_pos, marker_mask, qtype)."""
    import openvino as ov
    import torch

    model.eval()
    B, L, K = 2, 64, 8
    h = torch.randn(B, L, hidden_size)
    att = torch.ones(B, L, dtype=torch.long)
    att[1, 40:] = 0
    mp = torch.zeros(B, K, dtype=torch.long)
    mp[:, :3] = torch.tensor([3, 7, 11])
    mm = torch.zeros(B, K, dtype=torch.bool)
    mm[:, :3] = True
    qt = torch.zeros(B, dtype=torch.long)
    inp = (h, att, mp, mm, qt)
    with torch.no_grad():
        traced = torch.jit.trace(model, inp, check_trace=False, strict=False)
        ref = model(*inp)[0].numpy()
    m = ov.convert_model(traced, example_input=inp)
    for i, n in enumerate(["hidden", "attention_mask", "marker_pos", "marker_mask", "qtype"]):
        m.inputs[i].get_tensor().set_names({n})
    m.outputs[0].get_tensor().set_names({"logits"})
    m.outputs[1].get_tensor().set_names({"act_logits"})
    m.reshape({"hidden": ov.PartialShape([-1, -1, hidden_size]), "attention_mask": ov.PartialShape([-1, -1]),
               "marker_pos": ov.PartialShape([-1, -1]), "marker_mask": ov.PartialShape([-1, -1]),
               "qtype": ov.PartialShape([-1])})
    ov.save_model(m, os.path.join(out_dir, "head.xml"), compress_to_fp16=True)
    got = ov.Core().compile_model(os.path.join(out_dir, "head.xml"), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(
        [t.numpy() for t in inp])[0]
    err = float(np.abs(got[:, :3] - ref[:, :3]).max())
    if not np.isfinite(err) or err > 0.1:
        raise RuntimeError("exported head does not match PyTorch (max |dlogit| = %s)" % err)
    return err


def _probs(items: Sequence[Item], logits: Sequence[np.ndarray], calib: Calibration, task: Task, n_rows: int):
    probs = {q.qid: [None] * n_rows for q in task.questions}
    for it, z in zip(items, logits):
        zz = z / calib.get(it.qtype, len(it.markers))
        p = np.exp(zz - zz.max())
        probs[it.qid][it.row] = p / p.sum()
    return probs


def finetune(task: Task, train_rows: List[Dict], name: str, base: str = "laya-en", eval_rows: Optional[List[Dict]] = None,
             epochs: int = 4, lr: float = 3e-4, batch: int = 16, val_fraction: float = 0.15, device: str = "auto",
             seed: int = 0, out_root: Optional[str] = None, log: Callable = None, depth: int = 0) -> Dict:
    """End to end: features on NPU/GPU -> train on CPU -> calibrate -> export -> evaluate.

    depth=0 trains only the decision head; depth=k also trains the top k encoder layers (see train/deep.py).
    """
    log = log or _log
    import torch

    from .. import hub
    from ..engine import Decider

    t_start = time.perf_counter()
    decider = Decider(base, device=device, verbose=False)
    if decider.manifest.get("format") == "gutcheck-head/1":
        raise ValueError("fine-tune from a base model, not another head (got %s)" % base)
    base_dir = decider.package_dir
    man = decider.manifest
    log("[1/5] base %s on %s | %d train rows%s" % (decider.name, decider.device, len(train_rows),
                                                  (" | %d eval rows" % len(eval_rows)) if eval_rows else ""))

    rng = random.Random(seed)
    rows = list(train_rows)
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_fraction)) if len(rows) >= 40 else 0
    val_rows, fit_rows = rows[:n_val], rows[n_val:]

    top_model = None
    if depth:
        from .. import hub as _hub
        from ..runtime.openvino_backend import OpenVINOBackend
        from .deep import build_top, ensure_lower_ir
        from .model import load_checkpoint

        ckpt = _hub.source_checkpoint(base_dir)
        lower = ensure_lower_ir(base_dir, ckpt, depth)
        decider.backend = OpenVINOBackend(base_dir, decider.device, max_len=decider.max_len, lower_xml=lower)
        full, _ = load_checkpoint(ckpt)
        top_model = build_top(full, depth)
        del full
    log("[2/5] encoding with the frozen %s" % ("lower %d layers" % (man["num_layers"] - depth) if depth else "encoder"))
    fit_items = extract_items(decider, task, fit_rows, log)
    val_items = extract_items(decider, task, val_rows, log) if val_rows else []
    eval_items = extract_items(decider, task, eval_rows, log) if eval_rows else []
    if not fit_items:
        raise ValueError("no labelled examples found - check the task's 'label' columns against the data")

    model = top_model or build_head(man["hidden_size"], man.get("head_layers", 2), len(man.get("act_costs", {})) + 1,
                                    os.path.join(base_dir, "head.safetensors"))
    before = None
    if eval_items:
        before = score_predictions(task, eval_rows, _probs(eval_items, predict_logits(model, eval_items),
                                                           decider.calibration, task, len(eval_rows)))

    log("[3/5] training %s (%d params) on %d items, %d epochs" % ("top %d layers + head" % depth if depth else "head",
        sum(p.numel() for p in model.parameters()), len(fit_items), epochs))
    train_head(model, fit_items, val_items, epochs=epochs, lr=lr, batch=batch, seed=seed, log=log)

    log("[4/5] calibrating on %d held-out items" % len(val_items))
    calib = fit_temperatures(val_items, predict_logits(model, val_items)) if val_items else Calibration()

    out = os.path.join(out_root or hub.models_dir(), name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    if depth:
        from .deep import export_top
        err = export_top(model, out, man["hidden_size"])
    else:
        err = export_head(model, out, man["hidden_size"])
    from safetensors.torch import save_file
    # fp16 is plenty for resuming training and halves the adapter on disk
    save_file({k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in model.state_dict().items()},
              os.path.join(out, "head.safetensors"))

    after = None
    if eval_items:
        after = score_predictions(task, eval_rows, _probs(eval_items, predict_logits(model, eval_items), calib, task,
                                                          len(eval_rows)))
    manifest = {
        "format": "gutcheck-head/1",
        "name": name,
        "base": man["name"],
        "base_path": base_dir,
        "depth": depth,
        "task": task.spec,
        "calibration": calib.to_config(),
        "max_len": man.get("max_len", 512),
        "head_max_len": man.get("head_max_len", 192),
        "hidden_size": man["hidden_size"],
        "head_layers": man.get("head_layers", 2),
        "act_costs": man.get("act_costs", {}),
        "train": {"rows": len(fit_rows), "val_rows": len(val_rows), "items": len(fit_items), "epochs": epochs,
                  "lr": lr, "seed": seed, "export_parity": err,
                  "seconds": round(time.perf_counter() - t_start, 1)},
        "eval": {"before": before, "after": after} if eval_items else None,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out, "gutcheck.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log("[5/5] wrote %s (%.0fs total)" % (out, time.perf_counter() - t_start))
    return manifest
