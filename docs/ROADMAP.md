# Roadmap and open work

Where gutcheck stands, what is unfinished, and the concrete next step for each item. Items are ordered by value per
hour of work. Every result claimed here was measured; anything not yet measured says so.

---

## 1. Deep lease router: finish the promising run

**What it is.** `gutcheck lease learn` trains the model that decides which skills / MCP tools / subagents a prompt
needs. By default it trains only the 26M-parameter decision head. `--depth 4` also trains the top 4 encoder layers
(75M parameters), the same recipe that lifted support-ticket accuracy from 0.815 to 0.905 (intent) and 0.470 to 0.705
(urgency) in `gutcheck train`.

**What we know.** On the benchmark catalog (110 items, 862 training prompts, 1,466 training examples after
shuffle augmentation):

| router | validation loss (lower is better) | test-split results |
|---|---:|---|
| zero-shot `laya-en` | 0.765 | P 0.74 / R 0.65 / reachable 0.954 at thr 0.2 |
| head-only, 3 epochs | 0.653 (best) | P 0.60 / R 0.78 / reachable 0.958 at thr 0.1 |
| **depth 4, after epoch 1 of 3** | **0.478** | **not measured: run stopped** |

After a single epoch the deep router's validation loss was already 27% below the best head-only router. For
support triage, a similar validation-loss gap turned into +9 and +23 accuracy points on the test set, so this is
the most promising unfinished experiment in the repo.

**Why it stopped.** On a 16 GB laptop the run holds the cached features (~1,466 examples x ~300 tokens x 1024 x fp16,
about 0.9 GB), the source checkpoint loaded in fp32 to build the trainable top (~1.7 GB, freed after), the trainable
top plus Adam state (~1 GB) and activations for 4 transformer layers at ~300 tokens. Together with a browser and other
apps the machine hit critical memory pressure during epoch 2, and the host process was stopped.

**How to rerun it.**

1. Free memory: close browsers and other heavy apps, and run nothing else (no NPU compiles, no benchmarks).
2. Run it in a normal terminal, not as a background job of an agent session that may reap it under memory pressure
   (for Claude Code, start it with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`, or just use a terminal):
   ```bash
   gutcheck lease learn --data data/lease/train.jsonl --catalog data/lease/catalog.json \
       --name lease-router-d4 --depth 4 --epochs 3 -d GPU
   ```
   Expect ~20 minutes per epoch on a Lunar Lake CPU, ~1 hour total.
3. If it still runs out of memory, in order of least impact on quality:
   - `--epochs 2` (the best head-only epoch was epoch 2 anyway);
   - train on one ordering per prompt instead of two (`shuffles=0` in `learn()`; halves the examples);
   - `--depth 2`;
   - implement the memory fix below.
4. Evaluate on the held-out split and compare with the rows above:
   ```bash
   python benchmarks/lease_bench.py --split test --mode choice --router lease-router-d4 \
       --thresholds 0.1,0.2,0.3 --device GPU --out benchmarks/results/lease_test_choice_lease-router-d4.json
   ```
5. If it beats zero-shot on reachable recall at equal or lower tokens, add a row to the README lease table (numbers
   from the JSON, never by hand) and document `--depth 4` as the recommended `lease learn` setting.

**Memory fix worth doing anyway.** `extract_items` keeps every example's hidden states in RAM. Writing them to a
`numpy.memmap` on disk (one file per run, rows addressed by offset) would cap feature memory near zero, make
`--depth 4` routers practical on 8-16 GB machines, and let runs resume after a crash. Files:
`src/gutcheck/train/head.py` (`Item`, `extract_items`, `_collate`) and `src/gutcheck/lease/learn.py`
(`build_items`).

**Then: a router for your own library.** The benchmark router is specific to its 110-item catalog. For a real
library, synthesise labelled prompts from that catalog and train on them:
```bash
gutcheck lease synth --llm "claude -p" --out my_prompts.jsonl   # any LLM CLI that takes a prompt argument
gutcheck lease learn --data my_prompts.jsonl --name my-router --depth 4
gutcheck lease config router=my-router
```
Hold out 20% of the synthesised prompts and run `lease_bench.py` on them before switching the default.

---

## 2. SSD models (Mamba-2) on the NPU

**Idea.** Mamba-2 ("state space duality") models keep a fixed-size recurrent state, so they can follow an entire
session at constant cost per token, whereas the decision model reads at most 512 tokens per call. Pairing a
~1.3B Mamba-2 model on the NPU with the decision models gives the lease layer a memory of the whole session.

**Uses, in order to test.**
1. Session-aware leasing: "now do the same for the Q3 file" should lease what the earlier turn used.
2. A local draft model when the decision model is unsure, before escalating to a cloud LLM.
3. Compact hand-over notes when switching agents.

**Status.** Not started. OpenVINO merged native Mamba-2 support on 2026-09-21
([openvinotoolkit/openvino#38168](https://github.com/openvinotoolkit/openvino/pull/38168)); NPU support for Mamba
models is not documented yet.

**First steps.**
1. Convert a Mamba-2 checkpoint of ~1.3B parameters (e.g. `state-spaces/mamba2-1.3b`) and run it on CPU, then GPU,
   then NPU. Record tokens/second and memory for each.
2. Measure the simplest integration first: summarise the last N turns with the SSD model into one line, pass it as
   `recent_context` to `Leaser.lease()` (the parameter already exists), and extend `lease_bench.py` with multi-turn
   prompts where the right tool depends on an earlier turn.
3. Only then try using the SSD hidden state directly as a feature for the router.

---

## 3. Smaller items

| item | status | next step |
|---|---|---|
| DirectML backend (AMD / any Windows GPU) | fails: DirectML's Reshape rejects the dynamic-shape ONNX graph, at every optimisation level and with static dimension overrides | find the failing `node_view` op in the dynamo export and replace it with a DML-friendly equivalent, or export a static-shape graph per bucket for DML |
| CUDA / QNN / CoreML backends | same graph as the verified ONNX CPU path, never run on real hardware | community test on each; `benchmarks/parity.py` works for any device string |
| Prebuilt model packages on the Hugging Face Hub | not started | publish the converted `laya-en` package so `gutcheck pull` needs no PyTorch at all |
| PyPI release | not started | `python -m build`, check the wheel in a clean venv (already verified locally), publish |
| INT8 on the NPU | not started | NNCF post-training quantisation with a calibration set; watch GeGLU outliers (the Hexagon port needed clamping) |
| Churn-risk label | no training method improved it | relabel a sample by hand to check whether the synthetic labels are consistent |
| Multilingual | `laya-multilingual` registered, not benchmarked | run parity and speed benchmarks on it |
