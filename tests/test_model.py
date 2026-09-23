"""Real-model tests. Opt in: GUTCHECK_TEST_MODEL=1 pytest -q  (needs `gutcheck pull laya-en`)."""
import numpy as np
import pytest

from conftest import model_available, requires_model

Q = {
    "team": {"type": "choice", "instructions": "Which team handles this?",
             "criteria": {"billing": "invoices, refunds", "tech": "bugs, outages", "sales": "pricing, demos"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["none", "soon", "blocking"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}
STATES = ["I was billed twice for March, please refund one charge.",
          {"body": "Production is down with 502s since 9am, we are blocked."},
          [{"role": "user", "content": "hi, what does the pro plan cost per seat?"}]]


@pytest.fixture(scope="module")
def decider():
    from gutcheck.engine import Decider

    return Decider("laya-en", verbose=False)


@requires_model
def test_answers_are_well_formed(decider):
    r = decider.decide(STATES[0], Q)
    a = r["answers"]
    assert set(a) == set(Q)
    assert abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= a["urgency"]["score"] <= 2 and 0 <= a["refund"]["noul"] <= 1
    assert r["usage"]["input_tokens"] > 0 and r["device"]


@requires_model
def test_obvious_cases(decider):
    a = decider.decide(STATES[0], Q)["answers"]
    assert a["team"]["choice"] == "billing" and a["refund"]["noul"] > 0.5
    a = decider.decide(STATES[1], Q)["answers"]
    assert a["team"]["choice"] == "tech"


@requires_model
def test_batch_equals_single(decider):
    batch = decider.decide_batch(STATES, Q)
    for s, b in zip(STATES, batch):
        one = decider.decide(s, Q)
        for qid in Q:
            x, y = one["answers"][qid], b["answers"][qid]
            key = "noul" if x["type"] == "noul" else "probabilities"
            if key == "noul":
                assert abs(x["noul"] - y["noul"]) < 0.02
            else:
                assert max(abs(x[key][k] - y[key][k]) for k in x[key]) < 0.02


@requires_model
def test_cpu_matches_accelerator(decider):
    from gutcheck.engine import Decider

    if decider.device == "CPU":
        pytest.skip("no accelerator")
    cpu = Decider("laya-en", device="CPU", verbose=False)
    a = decider.decide(STATES[0], Q)["answers"]["team"]["probabilities"]
    b = cpu.decide(STATES[0], Q)["answers"]["team"]["probabilities"]
    assert max(abs(a[k] - b[k]) for k in a) < 0.05


@requires_model
@pytest.mark.skipif(not model_available("support-triage"), reason="support-triage head not trained")
def test_finetuned_head_uses_default_questions():
    from gutcheck.engine import Decider

    d = Decider("support-triage", verbose=False)
    r = d.decide("Please cancel our subscription at the end of the month.")
    assert set(r["answers"]) == {"intent", "urgency", "churn_risk"}
