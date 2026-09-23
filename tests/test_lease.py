import json
import os

import numpy as np

from gutcheck.lease.catalog import Catalog, Item, parse_frontmatter
from gutcheck.lease.daemon import render_context
from gutcheck.lease.leaser import BM25, Leased, LeaseResult, Leaser


def test_frontmatter_folded_and_quoted():
    fm = parse_frontmatter("---\nname: \"docx\"\ndescription: >\n  Create Word docs.\n  Edit them too.\n---\nbody")
    assert fm == {"name": "docx", "description": "Create Word docs. Edit them too."}
    fm = parse_frontmatter("---\nname: x\ndescription: |-\n  line one\n---\n")
    assert fm["description"] == "line one"
    assert parse_frontmatter("no frontmatter here") == {}


def test_catalog_from_skill_dirs(tmp_path):
    for n, d in (("pptx", "Make slide decks"), ("pdf", "Read PDFs")):
        os.makedirs(tmp_path / n)
        (tmp_path / n / "SKILL.md").write_text("---\nname: %s\ndescription: %s\n---\n# body\n" % (n, d), encoding="utf-8")
    cat = Catalog.from_skill_dirs(str(tmp_path))
    assert len(cat) == 2 and cat["skill:pptx"].description == "Make slide decks"
    assert cat["skill:pdf"].payload["path"].endswith("SKILL.md")
    p = tmp_path / "cat.json"
    cat.to_json(str(p))
    assert len(Catalog.from_json(str(p))) == 2


def test_bm25_prefers_exact_terms():
    b = BM25(["github create issue", "linear create issue", "slack post message"])
    s = b.scores("open a linear issue")
    assert int(np.argmax(s)) == 1


class FakeEmbedder:
    device = "CPU"

    def _v(self, texts):
        vocab = ["slide", "deck", "pdf", "slack", "message", "sql"]
        out = np.array([[t.lower().count(w) for w in vocab] for t in texts], np.float32) + 1e-3
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    embed_docs = _v

    def embed_queries(self, texts):
        return self._v(texts)


def _catalog():
    return Catalog([Item("pptx", "pptx", "slide deck builder"), Item("pdf", "pdf", "pdf reader"),
                    Item("slack", "slack__post", "post slack message", kind="mcp_tool", payload={"schema": "{text:str!}"})])


def test_leaser_dense_mode_and_tiers():
    lz = Leaser(_catalog(), embedder=FakeEmbedder(), mode="dense", threshold=0.5, hint_k=2, shortlist_k=3)
    res = lz.lease("make a slide deck")
    assert res.ids == ["pptx"]
    assert len(res.hinted) == 1 and res.hinted[0].item.id != "pptx"
    assert res.tokens_leased < res.tokens_catalog


class FakeDecider:
    def decide(self, state, questions):
        q = questions["x"]
        keys = list(q["criteria"])
        p = {k: (0.9 if k == "slack" else 0.1 / (len(keys) - 1)) for k in keys}
        return {"answers": {"x": {"type": "choice", "choice": "slack", "probabilities": p, "confidence": 0.8}}}


def test_leaser_choice_mode_uses_decider_probabilities():
    lz = Leaser(_catalog(), embedder=FakeEmbedder(), decider=FakeDecider(), mode="choice", threshold=0.2)
    res = lz.lease("tell the team on slack")
    assert res.ids == ["slack"] and abs(res.leased[0].p - 0.9) < 1e-6


def test_pinned_items_always_leased():
    lz = Leaser(_catalog(), embedder=FakeEmbedder(), mode="dense", threshold=0.99, pinned=["pdf"])
    assert "pdf" in lz.lease("anything").ids


def test_render_context_empty_when_nothing():
    assert render_context(LeaseResult("x", [], [], [], 0, 100)) == ""
    it = Item("slack", "slack__post", "post a message", kind="mcp_tool", payload={"schema": "{text:str!}"})
    txt = render_context(LeaseResult("x", [Leased(it, 0.9, 0.5, 0)], [], [], 50, 100, {"shortlist": 3, "decide": 40}))
    assert "slack__post" in txt and "{text:str!}" in txt and "gateway" in txt


def test_synth_validates_llm_output():
    from gutcheck.lease.synth import synth

    cat = _catalog()
    fake = lambda prompt: ('```json\n{"prompt": "make slides", "needs": ["pptx"]}\n'
                           '{"prompt": "bad id", "needs": ["nope"]}\nnot json\n'
                           '{"prompt": "make slides", "needs": ["pptx"]}\n{"prompt": "hi", "needs": []}\n')
    rows = synth(cat, llm="unused", calls=2, llm_fn=fake, log=lambda *a: None)
    assert rows == [{"prompt": "make slides", "needs": ["pptx"]}, {"prompt": "hi", "needs": []}]


class NoneDecider:
    def decide(self, state, questions):
        keys = list(questions["x"]["criteria"])
        p = {k: (0.9 if k == "none" else 0.1 / (len(keys) - 1)) for k in keys}
        return {"answers": {"x": {"type": "choice", "choice": "none", "probabilities": p, "confidence": 0.8}}}


def test_confident_none_injects_nothing():
    lz = Leaser(_catalog(), embedder=FakeEmbedder(), decider=NoneDecider(), mode="choice", none_silence=0.6)
    res = lz.lease("what's the capital of France")
    assert res.leased == [] and res.hinted == [] and render_context(res) == ""
    lz.none_silence = 0.95  # not confident enough -> hints still offered
    assert lz.lease("what's the capital of France").hinted
