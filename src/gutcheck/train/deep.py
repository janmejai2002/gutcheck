"""Deep fine-tuning: also train the top `depth` encoder layers, not just the decision head.

Head-only training leaves the encoder frozen, and the encoder is where the question, the options and the
state actually interact - so it moves calibration more than accuracy. Unfreezing the top few layers adds
that capacity while staying laptop-sized:

    lower  = embeddings + layers[:N-depth]          frozen, exported once per base model as OpenVINO IR
                                                    ("lower_d{depth}.xml"), runs on the NPU/GPU to cache
                                                    mid-network features for every training example
    top    = layers[N-depth:] + final_norm + head   trained in PyTorch on the CPU, exported as "top.xml"

At inference the runtime chains lower (shared by every deep adapter of that depth) -> top (per task).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from ..export import MASK_PENALTY


def _masks(attention_mask, n: int, window: int):
    import torch

    pos = torch.arange(n, device=attention_mask.device)
    dist = (pos[:, None] - pos[None, :]).abs()
    eye = (dist == 0)[None, None]
    key = attention_mask.bool()[:, None, None, :]
    zero = torch.zeros((), dtype=torch.float32)
    neg = torch.full((), MASK_PENALTY, dtype=torch.float32)
    return {"full_attention": torch.where(key | eye, zero, neg),
            "sliding_attention": torch.where((key & (dist <= window)[None, None]) | eye, zero, neg)}


def _window(enc) -> int:
    c = enc.config
    return int(getattr(c, "sliding_window", None) or getattr(c, "local_attention", 128) // 2)


def build_lower(model, depth: int):
    """embeddings + first N-depth layers -> mid hidden states (the frozen, shared part)."""
    import torch

    enc = model.encoder
    n_keep = len(enc.layers) - depth
    window = _window(enc)

    class Lower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            n = input_ids.size(1)
            h = self.enc.embeddings(input_ids=input_ids)
            pos = torch.arange(n, device=input_ids.device).unsqueeze(0)
            masks = _masks(attention_mask, n, window)
            rope = {t: self.enc.rotary_emb(h, pos, t) for t in set(self.enc.config.layer_types)}
            for layer in self.enc.layers[:n_keep]:
                h = layer(h, attention_mask=masks[layer.attention_type], position_embeddings=rope[layer.attention_type])
            return h

    return Lower().eval()


def build_top(model, depth: int):
    """last `depth` layers + final norm + decision head: the trainable part."""
    import copy

    import torch

    enc = model.encoder
    window = _window(enc)

    class Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(copy.deepcopy(list(enc.layers[len(enc.layers) - depth:])))
            self.final_norm = copy.deepcopy(enc.final_norm)
            self.rotary_emb = copy.deepcopy(enc.rotary_emb)
            self.layer_types = list(set(enc.config.layer_types))
            self.head = copy.deepcopy(model.head)
            self.type_emb = copy.deepcopy(model.type_emb)
            self.scorer = copy.deepcopy(model.scorer)
            self.act_head = copy.deepcopy(model.act_head)
            self.register_buffer("temperature", torch.ones(3))
            self.head_checkpointing = False

        from .model import DecisionModel as _DM
        decide = _DM.decide

        def forward(self, hidden, attention_mask, marker_pos, marker_mask, qtype):
            n = hidden.size(1)
            pos = torch.arange(n, device=hidden.device).unsqueeze(0)
            masks = _masks(attention_mask, n, window)
            rope = {t: self.rotary_emb(hidden, pos, t) for t in self.layer_types}
            h = hidden
            for layer in self.layers:
                h = layer(h, attention_mask=masks[layer.attention_type], position_embeddings=rope[layer.attention_type])
            h = self.final_norm(h)
            return self.decide(h, attention_mask, marker_pos, marker_mask, qtype)

    return Top()


def ensure_lower_ir(base_dir: str, ckpt_dir: str, depth: int, verbose: bool = True) -> str:
    """Export the shared lower part for this depth into the base package (once)."""
    path = os.path.join(base_dir, "lower_d%d.xml" % depth)
    if os.path.exists(path):
        return path
    import openvino as ov
    import torch

    from .model import load_checkpoint

    model, _ = load_checkpoint(ckpt_dir)
    lower = build_lower(model, depth)
    ids = torch.randint(1000, 5000, (2, 64))
    att = torch.ones(2, 64, dtype=torch.long)
    att[1, 40:] = 0
    with torch.no_grad():
        traced = torch.jit.trace(lower, (ids, att), check_trace=False, strict=False)
        ref = lower(ids, att).numpy()
        full = model.encoder(input_ids=ids, attention_mask=_masks(att, 64, _window(model.encoder))).last_hidden_state
    m = ov.convert_model(traced, example_input=(ids, att))
    m.inputs[0].get_tensor().set_names({"input_ids"})
    m.inputs[1].get_tensor().set_names({"attention_mask"})
    m.outputs[0].get_tensor().set_names({"mid"})
    m.reshape({"input_ids": ov.PartialShape([-1, -1]), "attention_mask": ov.PartialShape([-1, -1])})
    ov.save_model(m, path, compress_to_fp16=True)
    got = ov.Core().compile_model(path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})([ids.numpy(), att.numpy()])[0]
    err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-6))
    # sanity: running the original top layers on `mid` must reproduce the full encoder
    top = build_top(model, depth).eval()
    with torch.no_grad():
        h = torch.from_numpy(ref)
        n = h.size(1)
        pos = torch.arange(n).unsqueeze(0)
        masks = _masks(att, n, _window(model.encoder))
        rope = {t: top.rotary_emb(h, pos, t) for t in top.layer_types}
        for layer in top.layers:
            h = layer(h, attention_mask=masks[layer.attention_type], position_embeddings=rope[layer.attention_type])
        h = top.final_norm(h)
    split_err = float((h - full).abs().max())
    if verbose:
        print("[gutcheck deep] lower_d%d: IR rel err %.4f, split err %.5f" % (depth, err, split_err), flush=True)
    if err > 0.05 or split_err > 1e-3:
        raise RuntimeError("lower/top split does not reproduce the encoder (rel err %s, split err %s)" % (err, split_err))
    return path


def export_top(top, out_dir: str, hidden_size: int) -> float:
    import openvino as ov
    import torch

    top.eval()
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
        traced = torch.jit.trace(top, inp, check_trace=False, strict=False)
        ref = top(*inp)[0].numpy()
    m = ov.convert_model(traced, example_input=inp)
    for i, n in enumerate(["hidden", "attention_mask", "marker_pos", "marker_mask", "qtype"]):
        m.inputs[i].get_tensor().set_names({n})
    m.outputs[0].get_tensor().set_names({"logits"})
    m.outputs[1].get_tensor().set_names({"act_logits"})
    m.reshape({"hidden": ov.PartialShape([-1, -1, hidden_size]), "attention_mask": ov.PartialShape([-1, -1]),
               "marker_pos": ov.PartialShape([-1, -1]), "marker_mask": ov.PartialShape([-1, -1]),
               "qtype": ov.PartialShape([-1])})
    path = os.path.join(out_dir, "head.xml")  # same slot as a head-only adapter; manifest records the depth
    ov.save_model(m, path, compress_to_fp16=True)
    got = ov.Core().compile_model(path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})([t.numpy() for t in inp])[0]
    err = float(np.abs(got[:, :3] - ref[:, :3]).max())
    if not np.isfinite(err) or err > 0.1:
        raise RuntimeError("exported top does not match PyTorch (max |dlogit| = %s)" % err)
    return err
