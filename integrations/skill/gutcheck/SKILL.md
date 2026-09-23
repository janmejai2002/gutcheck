---
name: gutcheck
description: Add fast local classification / routing / yes-no checks to code you are building, using gutcheck (System-1 decision models on the user's NPU/GPU/CPU, ~20-60 ms, no LLM tokens). Use when a program needs to classify text or JSON, triage or route requests, gate on a calibrated probability, add a guardrail, or pick a model/tool - instead of calling an LLM for a label.
---

# gutcheck: typed decisions in ~50 ms, locally

Use gutcheck when code needs a **label, a score or a yes/no** about text or JSON. Do not use it to generate text,
extract spans, or reason over many steps - that is still an LLM's job.

## Install / check
```bash
pip install "gutcheck[export,mcp] @ git+https://github.com/janmejai2002/gutcheck"   # model converts once (~850 MB)
gutcheck doctor --test           # shows NPU/GPU/CPU and runs one decision
```

## Python API (the whole surface you need)
```python
import gutcheck
d = gutcheck.load()                      # default model laya-en on the best device (GPU > NPU > CPU)

res = d.decide(state, questions)         # state: str | dict | list (a chat transcript is a list)
res["answers"]["<id>"]                   # one typed answer per question id
d.decide_batch([s1, s2, ...], questions) # many states, same questions: much faster than a loop

# shortcuts
d.choice("text", "Which team?", {"billing": "invoices, refunds", "tech": "bugs"})  # -> {"choice", "probabilities", "confidence"}
d.noul("text", "The user asks for a refund")                                     # -> float P(true)
d.score("text", "How urgent?", ["none", "soon", "blocking"])                     # -> {"score": expected level, ...}
```

Question types (Jev wire format):
```python
questions = {
  "team":    {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "invoices, refunds", "tech": "bugs, outages"}},   # <= ~20 options works best
  "urgency": {"type": "score",  "instructions": "How urgent?", "criteria": ["none", "soon", "blocking"]},  # lowest first
  "refund":  {"type": "noul",   "instructions": "Does the customer ask for money back?"},
}
```
Answers: choice -> `choice`, `probabilities`, `confidence`; score -> `score` (expected level), `probabilities`;
noul -> `noul` = P(true).

## Rules that make it work well
- **Ask concrete, observable questions; split abstract ones.** "Is this high-stakes?" scored 0.09 on a medication
  question; "The request is about health, medicine or a medical decision." scored 0.82. Ask several concrete
  `noul`s (medical / money / legal / safety) and combine them in code (e.g. `max`).
- **Phrase `noul` as a statement to verify**, not an open question.
- **Describe options.** `{"billing": "invoices, refunds"}` beats `["billing"]`.
- **Gate on probability, don't trust argmax blindly**: act when the chosen label's probability
  (`probabilities[choice]`) or `noul` is >= 0.8, otherwise escalate to an LLM or a human. Probabilities are
  calibrated; that is the point. (The `confidence` field is entropy-based and reads much lower - e.g. 0.48 for a
  0.88 vs 0.12 split - so do not threshold it like a probability.)
- **Batch** with `decide_batch`; one `Decider` per process (loading costs seconds, calls cost milliseconds).
- **Zero-shot is decent on common tasks and weak on niche ones.** If accuracy matters, fine-tune (below).
- More than ~20 labels: shortlist first (`gutcheck.lease.Leaser` does embedding shortlist + decision).

## Fine-tune on the user's labels (minutes on a laptop)
Write a task spec, then train on a JSONL/CSV with one column per label:
```json
{"name": "ticket-router", "state": "text",
 "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "...", "tech": "..."}, "label": "team"}}}
```
```bash
gutcheck train --task task.json --data train.jsonl --eval test.jsonl --name ticket-router
```
```python
d = gutcheck.load("ticket-router"); d.decide("text")   # trained models answer their own questions by default
```

## Other surfaces
- HTTP (Jev-compatible): `gutcheck serve` -> `POST http://127.0.0.1:8765/v1/systemone {"state", "questions"}`
- MCP tools for agents: `gutcheck mcp` (classify / check / rate / decide)
