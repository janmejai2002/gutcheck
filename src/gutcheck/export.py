"""Export a Laya-format checkpoint to a self-contained gutcheck model package (OpenVINO IR).

    <out>/gutcheck.json        manifest: config, calibration, source revision
    <out>/tokenizer/           tokenizer.json + tokenizer_config.json
    <out>/decider.xml|.bin     full model (encoder + decision head), dynamic [batch, seq]

The runtime reshapes the one IR to static buckets per device, so weights are stored once.

Why a custom attention mask: ModernBERT's sliding-window layers (|i-j| <= 64) leave a padded query far
into the padding with *no* visible key. Softmax over an all -inf row is NaN, and 0 * NaN poisons the real
rows in the next layer. The OpenVINO CPU plugin produces NaN logits; GPU/NPU happen to survive. We build
both masks here with a finite penalty and always let a token see itself, so no row is ever empty. Real
tokens' outputs are mathematically unchanged (their own key is already valid).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from typing import Dict, Optional

import numpy as np

MASK_PENALTY = -1.0e4
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]


def _wrapper(model, window: int):
    import torch

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            n = input_ids.size(1)
            pos = torch.arange(n, device=input_ids.device)
            dist = (pos[:, None] - pos[None, :]).abs()
            eye = (dist == 0)[None, None]
            key = attention_mask.bool()[:, None, None, :]
            full = key | eye
            slide = (key & (dist <= window)[None, None]) | eye
            zero = torch.zeros((), dtype=torch.float32)
            neg = torch.full((), MASK_PENALTY, dtype=torch.float32)
            masks = {"full_attention": torch.where(full, zero, neg),
                     "sliding_attention": torch.where(slide, zero, neg)}
            h = self.m.encoder(input_ids=input_ids, attention_mask=masks).last_hidden_state
            logits, act = self.m.decide(h, attention_mask, marker_pos, marker_mask, qtype)
            return logits, act, h

    return ExportWrapper(model).eval()


def example_inputs(tok_pad_id: int, batch: int = 2, seq: int = 128, markers: int = 8):
    import torch

    ids = torch.full((batch, seq), tok_pad_id, dtype=torch.long)
    ids[:, :40] = torch.randint(1000, 5000, (batch, 40))
    att = torch.zeros((batch, seq), dtype=torch.long)
    att[:, :40] = 1
    mp = torch.zeros((batch, markers), dtype=torch.long)
    mp[:, :3] = torch.tensor([5, 9, 13])
    mm = torch.zeros((batch, markers), dtype=torch.bool)
    mm[:, :3] = True
    qt = torch.zeros((batch,), dtype=torch.long)
    return ids, att, mp, mm, qt


def export_openvino(model_dir: str, out_dir: str, name: Optional[str] = None, source: Optional[Dict] = None,
                    weights: str = "fp16", verbose: bool = True) -> str:
    """Convert `model_dir` (Laya format) into a gutcheck package at `out_dir`.

    weights: "fp16" (default, most accurate) or "int8" (weight-only INT8 via NNCF; half the size).
    """
    import openvino as ov
    import torch

    from .tokenizer import Tok
    from .train.model import load_checkpoint

    def log(msg):
        if verbose:
            print("[gutcheck export] " + msg, flush=True)

    log("loading checkpoint from %s" % model_dir)
    model, cfg = load_checkpoint(model_dir)
    ecfg = model.encoder.config
    window = int(getattr(ecfg, "sliding_window", None) or getattr(ecfg, "local_attention", 128) // 2)
    tok = Tok(os.path.join(model_dir, "tokenizer"))
    w = _wrapper(model, window)
    inp = example_inputs(tok.pad_token_id)

    log("tracing (window=%d, hidden=%d, layers=%d)" % (window, ecfg.hidden_size, ecfg.num_hidden_layers))
    with torch.no_grad():
        traced = torch.jit.trace(w, inp, check_trace=False, strict=False)
        ref = [t.numpy() for t in w(*inp)]
    ovm = ov.convert_model(traced, example_input=inp)
    for i, n in enumerate(INPUT_NAMES):
        ovm.inputs[i].get_tensor().set_names({n})
    ovm.outputs[0].get_tensor().set_names({"logits"})
    ovm.outputs[1].get_tensor().set_names({"act_logits"})
    ovm.outputs[2].get_tensor().set_names({"hidden"})
    ovm.reshape({n: ov.PartialShape([-1, -1]) if n not in ("qtype",) else ov.PartialShape([-1]) for n in INPUT_NAMES})

    if weights == "int8":
        import nncf

        log("compressing weights to INT8")
        ovm = nncf.compress_weights(ovm, mode=nncf.CompressWeightsMode.INT8_ASYM)

    os.makedirs(out_dir, exist_ok=True)
    ov.save_model(ovm, os.path.join(out_dir, "decider.xml"), compress_to_fp16=(weights == "fp16"))

    # sanity check on CPU: the exported graph must reproduce the torch logits
    cm = ov.Core().compile_model(os.path.join(out_dir, "decider.xml"), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    got = cm([t.numpy() for t in inp])
    k = 3
    err = float(np.abs(got[0][:, :k] - ref[0][:, :k]).max())
    log("CPU parity check: max |dlogit| = %.5f" % err)
    if not np.isfinite(err) or err > 0.25:
        raise RuntimeError("exported model does not match PyTorch (max |dlogit| = %s)" % err)

    # head weights (everything but the encoder) so `gutcheck train` needs neither the source
    # checkpoint nor its 800 MB of encoder weights in PyTorch
    from safetensors.torch import save_file
    head_sd = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    save_file(head_sd, os.path.join(out_dir, "head.safetensors"))

    tdst = os.path.join(out_dir, "tokenizer")
    os.makedirs(tdst, exist_ok=True)
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(model_dir, "tokenizer", fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(tdst, fn))

    manifest = {
        "format": "gutcheck/1",
        "name": name or os.path.basename(os.path.normpath(out_dir)),
        "source": source or {"path": os.path.abspath(model_dir)},
        "encoder": cfg.get("encoder"),
        "hidden_size": ecfg.hidden_size,
        "num_layers": ecfg.num_hidden_layers,
        "max_len": cfg.get("max_len", 512),
        "head_max_len": cfg.get("head_max_len", 192),
        "act_costs": cfg.get("act_costs", {}),
        "head_layers": cfg.get("head_layers", 2),
        "outputs": ["logits", "act_logits", "hidden"],
        "calibration": {"temperature": cfg.get("temperature", [1.0, 1.0, 1.0]),
                        "temperature_by_options": cfg.get("temperature_by_options", {})},
        "weights": weights,
        "parity_max_abs_logit_err": err,
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "gutcheck.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log("wrote %s" % out_dir)
    return out_dir


def export_onnx(model_dir: str, out_dir: str, verbose: bool = True, opset: int = 18) -> str:
    """Also write `decider.onnx` (same graph and fixed masks as the IR) for the ONNX Runtime backend:
    CUDA / TensorRT, DirectML, QNN, CoreML. Weights are stored in fp32 external data next to it."""
    import torch

    from .tokenizer import Tok
    from .train.model import load_checkpoint

    def log(msg):
        if verbose:
            print("[gutcheck export] " + msg, flush=True)

    model, cfg = load_checkpoint(model_dir)
    ecfg = model.encoder.config
    window = int(getattr(ecfg, "sliding_window", None) or getattr(ecfg, "local_attention", 128) // 2)
    tok = Tok(os.path.join(model_dir, "tokenizer"))
    w = _wrapper(model, window)
    inp = example_inputs(tok.pad_token_id)
    path = os.path.join(out_dir, "decider.onnx")
    os.makedirs(out_dir, exist_ok=True)
    dyn = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
           "marker_pos": {0: "batch", 1: "markers"}, "marker_mask": {0: "batch", 1: "markers"}, "qtype": {0: "batch"},
           "logits": {0: "batch", 1: "markers"}, "act_logits": {0: "batch"}, "hidden": {0: "batch", 1: "seq"}}
    log("exporting ONNX (opset %d)" % opset)
    # nn.TransformerEncoderLayer's inference fast path is one fused op (aten::_transformer_encoder_layer_fwd)
    # that ONNX cannot express; the unfused path is numerically the same
    torch.backends.mha.set_fastpath_enabled(False)
    from torch.export import Dim

    batch, seq, markers = Dim("batch", min=1, max=64), Dim("seq", min=8, max=8192), Dim("markers", min=2, max=256)
    shapes = {"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq},
              "marker_pos": {0: batch, 1: markers}, "marker_mask": {0: batch, 1: markers}, "qtype": {0: batch}}
    import sys

    # torch.onnx prints progress marks like U+2705; on a Windows cp1252 console that raises
    # UnicodeEncodeError mid-export, so make the streams lossy instead of fatal
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    with torch.no_grad():
        # the dynamo exporter keeps sequence length symbolic; the legacy TorchScript exporter bakes
        # nn.MultiheadAttention's reshape sizes from the example input
        prog = torch.onnx.export(w, inp, input_names=INPUT_NAMES, output_names=["logits", "act_logits", "hidden"],
                                 dynamic_shapes={"input_ids": shapes["input_ids"], "attention_mask": shapes["attention_mask"],
                                                 "marker_pos": shapes["marker_pos"], "marker_mask": shapes["marker_mask"],
                                                 "qtype": shapes["qtype"]},
                                 opset_version=opset, dynamo=True, verbose=False)
        prog.save(path, external_data=True)
        ref = w(*inp)[0].numpy()
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {n: t.numpy() for n, t in zip(INPUT_NAMES, inp)})[0]
    err = float(np.abs(got[:, :3] - ref[:, :3]).max())
    # a different (batch, seq, markers) than the export example must also run and match
    inp2 = example_inputs(tok.pad_token_id, batch=3, seq=200, markers=5)
    with torch.no_grad():
        ref2 = w(*inp2)[0].numpy()
    got2 = sess.run(["logits"], {n: t.numpy() for n, t in zip(INPUT_NAMES, inp2)})[0]
    err = max(err, float(np.abs(got2[:, :3] - ref2[:, :3]).max()))
    log("ONNX CPU parity check: max |dlogit| = %.5f" % err)
    if not np.isfinite(err) or err > 0.25:
        raise RuntimeError("ONNX export does not match PyTorch (max |dlogit| = %s)" % err)
    return path
