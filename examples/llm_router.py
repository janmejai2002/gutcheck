"""Route each request to the cheapest LLM that can handle it - decided locally in ~50 ms, zero LLM tokens.

    python examples/llm_router.py "write a haiku about rain"
"""
import sys

import gutcheck

TIERS = ["small", "medium", "large"]  # map to your providers, e.g. haiku / sonnet / opus

QUESTIONS = {
    "difficulty": {"type": "score", "instructions": "How hard is this request for a language model?",
                   "criteria": ["trivial lookup or one-liner", "short answer, little reasoning",
                                "several steps of reasoning", "long multi-step reasoning or specialist knowledge"]},
    "needs_tools": {"type": "noul", "instructions": "Does answering need web search, files or other tools?"},
    # "Is this high-stakes?" is an abstract judgement and a System-1 model scores it poorly (0.09 for a
    # medication question). Concrete, observable statements work far better, so ask several and take the max.
    "medical": {"type": "noul", "instructions": "The request is about health, medicine or a medical decision."},
    "money": {"type": "noul", "instructions": "The request is about money, payments, investing or financial transfers."},
    "legal": {"type": "noul", "instructions": "The request is about laws, contracts or legal rights."},
    "safety": {"type": "noul", "instructions": "The request is about physical danger, chemicals, weapons or injury risk."},
}
STAKES = ("medical", "money", "legal", "safety")


def route(d, request: str) -> dict:
    a = d.decide({"request": request}, QUESTIONS)["answers"]
    score = a["difficulty"]["score"]                       # expected level, 0..3
    tier = 0 if score < 1.0 else 1 if score < 2.2 else 2
    stakes = max(a[k]["noul"] for k in STAKES)
    if stakes > 0.5:                                        # escalate on risk, not on vibes
        tier = 2
    return {"tier": TIERS[tier], "difficulty": round(score, 2), "tools": a["needs_tools"]["noul"] > 0.5,
            "stakes": round(stakes, 2)}


if __name__ == "__main__":
    d = gutcheck.load()
    # the last one is a known miss (stakes ~0.2): zero-shot models have gaps - fine-tune on your own traffic
    for req in sys.argv[1:] or ["what's 2+2", "refactor this 2k-line module into services with tests",
                                "should I stop taking my blood pressure meds before surgery?",
                                "how do I wire money to my landlord overseas?",
                                "can I mix bleach and ammonia to clean my bathroom?"]:
        print("%-70s -> %s" % (req[:70], route(d, req)))
