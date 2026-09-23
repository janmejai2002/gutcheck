"""Minimal tokenizer over the Rust `tokenizers` library (no `transformers` needed at runtime)."""
from __future__ import annotations

import json
import os
from typing import List

from tokenizers import Tokenizer


class Tok:
    """Exposes just what `spec.build_sequence` needs, with ids identical to HF `AutoTokenizer`."""

    def __init__(self, tok_dir: str):
        self._tk = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
        self._tk.no_padding()
        self._tk.no_truncation()
        cfg_path = os.path.join(tok_dir, "tokenizer_config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)

        def special(name: str, default: str) -> str:
            v = cfg.get(name, default)
            return v["content"] if isinstance(v, dict) else v

        self.cls_token = special("cls_token", "[CLS]")
        self.sep_token = special("sep_token", "[SEP]")
        self.mask_token = special("mask_token", "[MASK]")
        self.pad_token = special("pad_token", "[PAD]")
        self.cls_token_id = self._id(self.cls_token)
        self.sep_token_id = self._id(self.sep_token)
        self.mask_token_id = self._id(self.mask_token)
        self.pad_token_id = self._id(self.pad_token)

    def _id(self, token: str) -> int:
        i = self._tk.token_to_id(token)
        if i is None:
            raise ValueError("tokenizer has no id for special token %r" % token)
        return i

    def encode(self, text: str) -> List[int]:
        return self._tk.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self._tk.encode_batch(texts, add_special_tokens=False)]

    def __len__(self) -> int:
        return self._tk.get_vocab_size()
