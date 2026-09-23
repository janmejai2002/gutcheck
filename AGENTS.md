# Working on gutcheck (notes for coding agents)

Layout (`src/gutcheck`):

| file | role | imports torch? |
|---|---|---|
| `spec.py` | question schema (Jev wire format), Laya token layout, calibration, answer decoding | no |
| `tokenizer.py` | `tokenizers`-only adapter; ids must equal HF `AutoTokenizer` | no |
| `engine.py` | `Decider` (public API), backend selection | no |
| `runtime/openvino_backend.py` | static (batch, seq) buckets, lazy compile, fused vs encoder+head | no |
| `runtime/onnx_backend.py` | ONNX Runtime providers (CUDA, DirectML, QNN, CoreML) | no |
| `hub.py` | registry, download, `resolve()` builds packages on first use | only when exporting |
| `export.py` | torch -> OpenVINO IR / ONNX; custom attention masks (see docstring) | yes |
| `train/` | task specs, metrics, head fine-tuning | `head.py`, `model.py` yes |
| `lease/` | catalog, hybrid shortlist, Leaser, daemon, Claude Code hooks, router training, synth | `learn.py` yes |
| `gateway.py` | MCP gateway (find_tools / call) over downstream MCP servers | no |
| `server.py`, `mcp_server.py`, `cli.py` | surfaces | no |

Rules:
- Runtime paths must not import torch or transformers. Keep them lazy.
- Never change `spec.build_sequence` without re-running `benchmarks/parity.py`: Laya checkpoints depend on the
  exact layout.
- Anything that edits user config (`~/.claude/settings.json`) backs it up first and must stay idempotent;
  see `tests/test_integrations.py`.
- Hooks must never block or fail a user's prompt: on any error, print nothing and exit 0.
- Numbers in README come from `benchmarks/`; do not edit them by hand.

Tests: `pytest -q` (fast, fakes). Model tests: `GUTCHECK_TEST_MODEL=1 pytest -q` with `laya-en` installed.
