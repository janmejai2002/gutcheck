"""PyTorch decision model (needed only for export and training; inference does not import torch).

Architecture from Laya (https://github.com/NandhaKishorM/laya, Apache-2.0): a bidirectional encoder,
a question-type embedding, a 2-layer transformer head, a scorer applied at each option marker, and a
small "act" head. Kept weight-compatible so Laya checkpoints load with strict=True.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DecisionModel(nn.Module):
    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    # -- pieces, so export and head-only training can reuse them ---------------------------------
    def encode(self, input_ids, attention_mask, encoder_mask=None):
        am = encoder_mask if encoder_mask is not None else attention_mask
        return self.encoder(input_ids=input_ids, attention_mask=am).last_hidden_state

    def decide(self, h, attention_mask, marker_pos, marker_mask, qtype) -> Tuple[torch.Tensor, torch.Tensor]:
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values if p.size(-1) >= 2 else torch.cat([p, torch.zeros_like(p)], -1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
        return logits, act_logits

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, encoder_mask=None):
        h = self.encode(input_ids, attention_mask, encoder_mask)
        return self.decide(h, attention_mask, marker_pos, marker_mask, qtype)


def read_config(model_dir: str) -> Dict:
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg: Dict, encoder_dir: Optional[str] = None, pretrained_encoder: bool = False) -> DecisionModel:
    from transformers import AutoConfig, AutoModel

    if pretrained_encoder:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    else:
        ecfg = AutoConfig.from_pretrained(encoder_dir or cfg["encoder"])
        ecfg.reference_compile = False
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    return DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)


def load_checkpoint(model_dir: str) -> Tuple[DecisionModel, Dict]:
    """Load a Laya-format checkpoint directory (rl_agent_config.json, model.safetensors, encoder/)."""
    from safetensors.torch import load_file

    cfg = read_config(model_dir)
    enc_dir = os.path.join(model_dir, "encoder")
    model = build_model(cfg, enc_dir if os.path.isdir(enc_dir) else None)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    return model.eval(), cfg
