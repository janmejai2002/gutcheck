"""Train a router head for *your* catalog: `gutcheck lease learn`.

Each labelled request ({"prompt": ..., "needs": [ids]}) becomes a stage-2 training example: the
shortlist the Leaser would build for it, as a choice over those candidates + "none", with the target
probability mass spread evenly over the needed items that made the shortlist (or on "none"). A copy with
the candidates shuffled is added so the head cannot lean on position. Labelled requests can come from
your logs or be synthesised by any LLM from the catalog descriptions (`gutcheck lease synth`).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
import shutil
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from ..spec import parse_question
from ..train.head import Item as TItem, build_head, export_head, fit_temperatures, predict_logits, train_head
from .catalog import Catalog
from .leaser import Leased, Leaser, choice_question


def _log(msg: str):
    print(msg, flush=True)


def build_items(leaser: Leaser, rows: List[Dict], shuffles: int = 1, seed: int = 0,
                log: Callable = None) -> List[TItem]:
    log = log or _log
    rng = random.Random(seed)
    d = leaser.decider
    enc_rows, meta = [], []
    for ri, r in enumerate(rows):
        sl = leaser.shortlist(r["prompt"])
        needs = set(r.get("needs") or [])
        orders = [list(range(len(sl)))]
        for _ in range(shuffles):
            o = list(range(len(sl)))
            rng.shuffle(o)
            orders.append(o)
        for oi, o in enumerate(orders):
            cands = [sl[i] for i in o]
            q = parse_question("x", choice_question(cands))
            hit = [j for j, k in enumerate(q.keys) if k in needs]
            t = np.zeros(len(q.keys), np.float32)
            if hit:
                t[hit] = 1.0 / len(hit)
            else:
                t[q.keys.index("none")] = 1.0
            e = d.encode(leaser.state(r["prompt"]), [q])[0]
            enc_rows.append((e.ids, e.markers, e.qtype))
            meta.append((ri, "x%d" % oi, e.markers, e.qtype, t))
    items: List[TItem] = []
    t0 = time.perf_counter()
    B = 64
    for s in range(0, len(enc_rows), B):
        feats = d.backend.features(enc_rows[s:s + B], d.tok.pad_token_id)
        for f, (ri, qid, markers, qt, t) in zip(feats, meta[s:s + B]):
            items.append(TItem(f, list(markers), qt, t, ri, qid))
        log("  encoded %d/%d examples (%.0fs)" % (min(len(enc_rows), s + B), len(enc_rows), time.perf_counter() - t0))
    return items


def learn(catalog: Catalog, rows: List[Dict], name: str, base: str = "laya-en", device: str = "auto",
          epochs: int = 3, lr: float = 3e-4, val_fraction: float = 0.15, shuffles: int = 1, seed: int = 0,
          out_root: Optional[str] = None, log: Callable = None, depth: int = 0) -> str:
    log = log or _log
    from .. import hub

    t_start = time.perf_counter()
    leaser = Leaser(catalog, device=device, router=base, mode="choice")
    d = leaser.decider
    man = d.manifest
    top_model = None
    if depth:  # same deep path as `gutcheck train --depth` (see train/deep.py)
        from .. import hub as _hub
        from ..runtime.openvino_backend import OpenVINOBackend
        from ..train.deep import build_top, ensure_lower_ir
        from ..train.model import load_checkpoint

        ckpt = _hub.source_checkpoint(d.package_dir)
        lower = ensure_lower_ir(d.package_dir, ckpt, depth)
        d.backend = OpenVINOBackend(d.package_dir, d.device, max_len=d.max_len, lower_xml=lower)
        full, _ = load_checkpoint(ckpt)
        top_model = build_top(full, depth)
        del full
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = int(round(len(rows) * val_fraction)) if len(rows) >= 40 else 0
    val_rows, fit_rows = rows[:n_val], rows[n_val:]
    log("[1/4] building shortlists + encoding %d train / %d val requests on %s" % (len(fit_rows), len(val_rows), d.device))
    fit_items = build_items(leaser, fit_rows, shuffles, seed, log)
    val_items = build_items(leaser, val_rows, 0, seed, log) if val_rows else []

    model = top_model or build_head(man["hidden_size"], man.get("head_layers", 2), len(man.get("act_costs", {})) + 1,
                                    os.path.join(d.package_dir, "head.safetensors"))
    log("[2/4] training router %s on %d examples" % ("top %d layers + head" % depth if depth else "head", len(fit_items)))
    train_head(model, fit_items, val_items, epochs=epochs, lr=lr, seed=seed, log=log)
    log("[3/4] calibrating")
    calib = fit_temperatures(val_items, predict_logits(model, val_items)) if val_items else None

    out = os.path.join(out_root or hub.models_dir(), name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    if depth:
        from ..train.deep import export_top
        err = export_top(model, out, man["hidden_size"])
    else:
        err = export_head(model, out, man["hidden_size"])
    from safetensors.torch import save_file
    save_file({k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in model.state_dict().items()},
              os.path.join(out, "head.safetensors"))
    manifest = {
        "format": "gutcheck-head/1", "name": name, "base": man["name"], "base_path": d.package_dir, "depth": depth,
        "task": {"name": "lease-router", "questions": {}},
        "calibration": calib.to_config() if calib else {"temperature": [1, 1, 1], "temperature_by_options": {}},
        "max_len": man.get("max_len", 512), "head_max_len": man.get("head_max_len", 192),
        "hidden_size": man["hidden_size"], "head_layers": man.get("head_layers", 2),
        "act_costs": man.get("act_costs", {}),
        "lease": {"shortlist_k": leaser.shortlist_k, "catalog_size": len(catalog)},
        "train": {"requests": len(fit_rows), "val_requests": len(val_rows), "examples": len(fit_items),
                  "epochs": epochs, "lr": lr, "export_parity": err, "seconds": round(time.perf_counter() - t_start, 1)},
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out, "gutcheck.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log("[4/4] wrote %s (%.0fs)" % (out, time.perf_counter() - t_start))
    return out
