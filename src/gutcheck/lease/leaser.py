"""Predictive leasing: decide, per request, which catalog items an agent actually needs.

    stage 1  shortlist   BM25 (exact names) + dense embeddings (paraphrases), fused by reciprocal rank
    stage 2  decide      the decision model answers a calibrated yes/no per shortlisted item

Only leased items reach the agent's context. Everything runs locally on the NPU/GPU in tens of ms,
with no LLM tokens spent on routing.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .catalog import Catalog, Item

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the to of for and or in on at by with from is are be it this that as into your you my me i we our "
            "use using when what which how do does can should will".split())


def _toks(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower().replace("_", " ")) if t not in _STOP]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [_toks(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        q = _toks(query)
        out = np.zeros(len(self.docs), np.float32)
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out[i] = s
        return out


@dataclass
class Leased:
    item: Item
    p: float                     # calibrated probability the item is needed (stage 2) or dense score
    dense: float
    rank: int


@dataclass
class LeaseResult:
    request: str
    leased: List[Leased]          # load these in full
    hinted: List[Leased]          # mention by name only (~10 tokens each); the agent can ask for them
    shortlist: List[Leased]
    tokens_leased: int
    tokens_catalog: int
    ms: Dict[str, float] = field(default_factory=dict)

    @property
    def ids(self) -> List[str]:
        return [l.item.id for l in self.leased]

    @property
    def tokens_saved(self) -> int:
        return self.tokens_catalog - self.tokens_leased


NOUL_TEMPLATE = "Is the {kind} \"{name}\" needed to carry out the request? It does: {description}"
CHOICE_INSTRUCTIONS = "Which capability must be loaded to carry out the request?"
NONE_OPTION = "none of these; the request can be handled without loading any of them"


def choice_question(cands: Sequence["Leased"]) -> Dict[str, Any]:
    """The stage-2 question: pick the needed capability among the shortlist, or none."""
    crit = {c.item.id: c.item.description[:160] for c in cands}
    crit["none"] = NONE_OPTION
    return {"type": "choice", "instructions": CHOICE_INSTRUCTIONS, "criteria": crit}


class Leaser:
    """
    >>> leaser = Leaser(Catalog.from_skill_dirs("~/.claude/skills-library"))
    >>> leaser.lease("turn these meeting notes into a slide deck").ids
    ['skill:pptx']
    """

    def __init__(self, catalog: Catalog, embedder=None, decider=None, device: str = "auto",
                 shortlist_k: int = 8, threshold: float = 0.2, max_lease: int = 4, hint_k: int = 5,
                 mode: str = "choice", router: str = "laya-en", pinned: Sequence[str] = (),
                 none_silence: Optional[float] = None):
        """mode: "choice" (one pass, recommended), "noul" (one pass per candidate), or "dense" (no
        decision model; threshold applies to cosine similarity). `router` may be a model trained with
        `gutcheck lease learn` for your catalog."""
        from ..embed import Embedder

        self.catalog = catalog
        self.items = list(catalog)
        self.embedder = embedder or Embedder(device=device)
        self.mode = mode
        self.decider = decider
        if self.decider is None and mode != "dense":
            from ..engine import Decider

            self.decider = Decider(router, device=device, verbose=False)
        self.shortlist_k, self.threshold, self.max_lease, self.hint_k = shortlist_k, threshold, max_lease, hint_k
        self.pinned = set(pinned)
        # Opt-in: when P("none") >= none_silence, drop hints too and inject nothing. Off by default because on
        # the lease benchmark any setting cost 1-8 points of reachability to save 1-7 tokens per prompt.
        self.none_silence = none_silence
        self.last_p_none: Optional[float] = None
        self.bm25 = BM25([i.text for i in self.items])
        self.doc_vecs = self.embedder.embed_docs([i.text for i in self.items])

    def shortlist(self, request: str, k: Optional[int] = None) -> List[Leased]:
        k = k or self.shortlist_k
        dense = self.doc_vecs @ self.embedder.embed_queries([request])[0]
        sparse = self.bm25.scores(request)
        rd = np.argsort(-dense)
        rs = np.argsort(-sparse)
        rrf = np.zeros(len(self.items))
        rrf[rd] += 1.0 / (60 + np.arange(len(rd)))
        has_sparse = sparse > 0
        rrf[rs] += np.where(has_sparse[rs], 1.0 / (60 + np.arange(len(rs))), 0.0)
        order = np.argsort(-rrf)[:k]
        return [Leased(self.items[i], float(dense[i]), float(dense[i]), r) for r, i in enumerate(order)]

    @staticmethod
    def state(request: str, context: Optional[str] = None):
        return {"request": request} if not context else {"request": request, "recent_context": context[-1500:]}

    def decide_choice(self, request: str, cands: List[Leased], context: Optional[str] = None) -> List[Leased]:
        a = self.decider.decide(self.state(request, context), {"x": choice_question(cands)})["answers"]["x"]
        for c in cands:
            c.p = float(a["probabilities"][c.item.id])
        self.last_p_none = float(a["probabilities"]["none"])
        return cands

    def decide(self, request: str, cands: List[Leased], context: Optional[str] = None) -> List[Leased]:
        state = self.state(request, context)
        qs = {"q%d" % j: {"type": "noul", "instructions": NOUL_TEMPLATE.format(
            kind=c.item.kind.replace("_", " "), name=c.item.name, description=c.item.description[:400])}
            for j, c in enumerate(cands)}
        ans = self.decider.decide(state, qs)["answers"]
        for j, c in enumerate(cands):
            c.p = float(ans["q%d" % j]["noul"])
        return cands

    def lease(self, request: str, context: Optional[str] = None) -> LeaseResult:
        t0 = time.perf_counter()
        cands = self.shortlist(request)
        self.last_p_none = None
        t1 = time.perf_counter()
        if self.mode == "choice":
            cands = self.decide_choice(request, cands, context)
        elif self.mode == "noul":
            cands = self.decide(request, cands, context)
        t2 = time.perf_counter()
        ranked = sorted(cands, key=lambda c: -c.p)
        leased = [c for c in ranked if c.p >= self.threshold][: self.max_lease]
        have = {c.item.id for c in leased}
        for pid in self.pinned:
            if pid in self.catalog and pid not in have:
                leased.append(Leased(self.catalog[pid], 1.0, 0.0, -1))
                have.add(pid)
        # hints come from the shortlist's own rank order (stage-1 recall is the safety net)
        hinted = [c for c in sorted(cands, key=lambda c: c.rank) if c.item.id not in have][: max(0, self.hint_k - len(leased))]
        if (not leased and self.none_silence is not None and self.last_p_none is not None
                and self.last_p_none >= self.none_silence):
            hinted = []  # the router is confident nothing here is needed: cost the agent zero tokens
        tokens = sum(l.item.token_cost for l in leased) + sum(estimate_hint_tokens(h.item) for h in hinted)
        return LeaseResult(request, leased, hinted, ranked, tokens, self.catalog.total_tokens,
                           {"shortlist": round((t1 - t0) * 1000, 1), "decide": round((t2 - t1) * 1000, 1)})


def estimate_hint_tokens(item: Item) -> int:
    return 4 + len(item.name) // 4
