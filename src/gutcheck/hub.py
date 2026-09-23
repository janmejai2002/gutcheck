"""Model registry: named models -> source checkpoints -> local gutcheck packages."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .runtime.openvino_backend import cache_root

LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8"

REGISTRY: Dict[str, Dict] = {
    "laya-en": {
        "repo": LAYA_REPO, "subfolder": None, "revision": LAYA_REVISION,
        "params": "421M", "base": "ModernBERT-large", "languages": "English", "max_len": 512,
        "description": "General English decisions. Best zero-shot on short English text.",
    },
    "laya-multilingual": {
        "repo": LAYA_REPO, "subfolder": "multilingual", "revision": None,
        "params": "322M", "base": "mmBERT-base", "languages": "100+", "max_len": 1024,
        "description": "Non-English or mixed-language text; 2x faster than laya-en.",
    },
    "laya-typed-decisions": {
        "repo": LAYA_REPO, "subfolder": "typed-decisions", "revision": None,
        "params": "421M", "base": "ModernBERT-large", "languages": "English", "max_len": 1024,
        "description": "Fine-tuned on support / invoice / security / agent-trace workflows.",
    },
}
ALIASES = {"default": "laya-en", "en": "laya-en", "laya": "laya-en", "multilingual": "laya-multilingual",
           "ml": "laya-multilingual", "typed-decisions": "laya-typed-decisions", "typed": "laya-typed-decisions"}


def models_dir() -> str:
    return os.path.join(cache_root(), "models")


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def package_path(name: str) -> str:
    return os.path.join(models_dir(), canonical(name))


def is_package(path: str) -> bool:
    """A base package (decider.xml) or a fine-tuned head package (head.xml)."""
    return os.path.exists(os.path.join(path, "gutcheck.json")) and (
        os.path.exists(os.path.join(path, "decider.xml")) or os.path.exists(os.path.join(path, "head.xml")))


def list_local() -> List[Dict]:
    out = []
    d = models_dir()
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if is_package(p):
                with open(os.path.join(p, "gutcheck.json"), encoding="utf-8") as f:
                    m = json.load(f)
                out.append({"name": n, "path": p, "kind": "head" if m.get("format") == "gutcheck-head/1" else "base",
                            "base": m.get("base"), "weights": m.get("weights"), "source": m.get("source"),
                            "task": (m.get("task") or {}).get("name")})
    return out


def download_with_retry(repo: str, **kw) -> str:
    """snapshot_download, retried once over plain HTTP if the Xet transfer backend fails (seen on Windows as
    'Cannot create a file when that file already exists', os error 183)."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, **kw)
    except OSError as e:
        if os.environ.get("HF_HUB_DISABLE_XET") == "1":
            raise
        old = os.environ.get("HF_HUB_DISABLE_XET")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            import huggingface_hub.constants as c

            c.HF_HUB_DISABLE_XET = True
        except Exception:
            pass
        try:
            return snapshot_download(repo, **kw)
        except Exception:
            raise e
        finally:
            if old is None:
                os.environ.pop("HF_HUB_DISABLE_XET", None)


def snapshot(repo: str, revision: Optional[str] = None, allow_patterns: Optional[List[str]] = None) -> str:
    """Local-first snapshot_download: use the cache when the files are there (works offline and with
    HF_HUB_OFFLINE=1), and only touch the network when something is missing."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, revision=revision, allow_patterns=allow_patterns, local_files_only=True)
    except Exception:
        return download_with_retry(repo, revision=revision, allow_patterns=allow_patterns)


def download_checkpoint(name: str) -> str:
    """Fetch the source checkpoint for a registry model; returns the checkpoint directory."""
    spec = REGISTRY[canonical(name)]
    prefix = (spec["subfolder"] + "/") if spec["subfolder"] else ""
    path = snapshot(spec["repo"], revision=spec["revision"], allow_patterns=[
        prefix + p for p in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
    return os.path.join(path, spec["subfolder"]) if spec["subfolder"] else path


def resolve(model: str, weights: str = "fp16", auto_build: bool = True, verbose: bool = True) -> str:
    """Return a local package directory for `model` (a registry name, a package dir, or a Laya checkpoint dir).

    Registry models are downloaded and exported on first use (needs the `export` extra: torch + transformers).
    """
    if os.path.isdir(model):
        if is_package(model):
            return model
        if os.path.exists(os.path.join(model, "rl_agent_config.json")):
            out = os.path.join(models_dir(), os.path.basename(os.path.normpath(model)))
            if not is_package(out):
                _export(model, out, os.path.basename(os.path.normpath(model)), {"path": os.path.abspath(model)}, weights, verbose)
            return out
        raise FileNotFoundError("%r is neither a gutcheck package nor a Laya checkpoint directory" % model)

    name = canonical(model)
    if name not in REGISTRY and is_package(os.path.join(models_dir(), name)):
        return os.path.join(models_dir(), name)  # a model you trained or imported
    suffix = "" if weights == "fp16" else "-" + weights
    out = os.path.join(models_dir(), name + suffix)
    if is_package(out):
        return out
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError("unknown model %r. Known models: %s (or pass a directory)" % (model, known))
    if not auto_build:
        raise FileNotFoundError("model %r is not installed; run `gutcheck pull %s`" % (name, name))
    ckpt = download_checkpoint(name)
    spec = REGISTRY[name]
    _export(ckpt, out, name + suffix, {"repo": spec["repo"], "subfolder": spec["subfolder"],
                                       "revision": spec["revision"]}, weights, verbose)
    return out


def _export(ckpt: str, out: str, name: str, source: Dict, weights: str, verbose: bool):
    try:
        from .export import export_openvino
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Building a model package needs PyTorch once: pip install \"gutcheck[export]\" (%s)" % e) from None
    export_openvino(ckpt, out, name=name, source=source, weights=weights, verbose=verbose)


def source_checkpoint(package_dir: str) -> str:
    """The Laya-format checkpoint a base package was built from (downloaded again if needed)."""
    with open(os.path.join(package_dir, "gutcheck.json"), encoding="utf-8") as f:
        man = json.load(f)
    src = man.get("source") or {}
    if src.get("path") and os.path.isdir(src["path"]):
        return src["path"]
    name = man.get("name", "").split("-int8")[0]
    if name in REGISTRY:
        return download_checkpoint(name)
    raise FileNotFoundError("cannot find the source checkpoint for %s" % package_dir)


def ensure_onnx(package_dir: str, verbose: bool = True) -> str:
    """Add decider.onnx to an existing base package (re-downloads the source checkpoint if needed)."""
    path = os.path.join(package_dir, "decider.onnx")
    if os.path.exists(path):
        return path
    ckpt = source_checkpoint(package_dir)
    try:
        from .export import export_onnx
    except ImportError as e:
        raise ImportError("ONNX export needs PyTorch once: pip install \"gutcheck[export]\" (%s)" % e) from None
    return export_onnx(ckpt, package_dir, verbose=verbose)
