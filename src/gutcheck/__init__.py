"""gutcheck: fast, calibrated System-1 decisions on your NPU, GPU or CPU."""
__version__ = "0.1.0"

from .spec import QuestionError  # noqa: E402


def __getattr__(name):
    # keep `import gutcheck` cheap: OpenVINO loads only when a Decider is created
    if name == "Decider":
        from .engine import Decider
        return Decider
    raise AttributeError(name)


def load(model: str = "laya-en", device: str = "auto", **kw):
    """`gutcheck.load()` -> a ready Decider (downloads/builds the model on first use)."""
    from .engine import Decider

    return Decider(model, device=device, **kw)


__all__ = ["Decider", "load", "QuestionError", "__version__"]
