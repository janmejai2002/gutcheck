import numpy as np
import pytest

from gutcheck.spec import (Calibration, QuestionError, build_sequence, clamp_temperature, decode_answer,
                           parse_question, parse_questions, temp_bucket)


class FakeTok:
    """Whitespace tokenizer with Laya-style special tokens."""
    mask_token, cls_token_id, sep_token_id, mask_token_id, pad_token_id = "[MASK]", 1, 2, 3, 0

    def encode(self, text):
        return [10 + (hash(w) % 1000) for w in text.split()]


def test_choice_list_and_dict():
    q = parse_question("dept", {"type": "choice", "instructions": "team?", "criteria": ["billing", "tech"]})
    assert q.keys == ["billing", "tech"] and q.options == ["billing", "tech"]
    q = parse_question("dept", {"type": "choice", "instructions": "team?", "criteria": {"billing": "refunds", "tech": None}})
    assert q.options == ["billing: refunds", "tech"]


def test_score_and_noul_rendering():
    q = parse_question("u", {"type": "score", "instructions": "urgent?", "criteria": ["low", "high"]})
    assert q.options == ["level 0: low", "level 1: high"] and q.legend == {"0": "low", "1": "high"}
    q = parse_question("n", {"type": "noul", "instructions": "spam?", "criteria": {"true": "spam"}})
    assert q.keys == ["false", "true"]
    assert q.options[0].startswith("false: no") and q.options[1] == "true: spam"
    q = parse_question("n", {"type": "noul", "instructions": "x?", "labels": {"false": "B", "true": "A"}})
    assert q.options[0].startswith("B:") and q.options[1].startswith("A:")


@pytest.mark.parametrize("bad", [
    {"type": "nope", "instructions": "x"},
    {"type": "choice", "instructions": "x"},
    {"type": "choice", "instructions": "x", "criteria": {}},
    {"type": "score", "instructions": "x", "criteria": ["only one"]},
    {"type": "noul", "instructions": "x", "criteria": ["not", "a", "dict"]},
    {"type": "noul"},
    {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(65)}},
])
def test_invalid_questions_name_the_question(bad):
    with pytest.raises(QuestionError) as e:
        parse_question("myq", bad)
    assert "myq" in str(e.value)


def test_structured_instructions_are_json():
    q = parse_question("q", {"type": "noul", "instructions": {"ask": "is it"}})
    assert q.instructions == '{"ask": "is it"}'


def test_sequence_layout_and_markers():
    tok = FakeTok()
    q = parse_question("d", {"type": "choice", "instructions": "which one", "criteria": ["a", "b", "c"]})
    e = build_sequence(tok, "hello world", q, max_len=64)
    assert e.ids[0] == tok.cls_token_id and e.ids[-1] == tok.sep_token_id
    assert len(e.markers) == 3 and all(e.ids[m] == tok.mask_token_id for m in e.markers)
    assert e.qtype == 0


def test_state_truncation_keeps_markers():
    tok = FakeTok()
    q = parse_question("d", {"type": "noul", "instructions": "x"})
    e = build_sequence(tok, "w " * 5000, q, max_len=128)
    assert len(e.ids) == 128 and len(e.markers) == 2


def test_left_truncation_keeps_newest_turn():
    tok = FakeTok()
    q = parse_question("d", {"type": "noul", "instructions": "x"})
    state = ["old"] * 400 + ["NEWEST"]
    right = build_sequence(tok, state, q, max_len=64, truncate_left=False)
    left = build_sequence(tok, state, q, max_len=64, truncate_left=True)
    newest = tok.encode('"NEWEST"]')
    assert newest[0] in left.ids and newest[0] not in right.ids


def test_temperature_clamping_refuses_sharpening():
    assert clamp_temperature(0.1) == 0.5
    assert clamp_temperature(99) == 5.0
    assert clamp_temperature("nan") == 1.0
    assert clamp_temperature(float("inf")) == 1.0
    c = Calibration([1.6, 1.2, 2.0], {"choice:11+": 0.1})
    assert c.get(0, 12) == 0.5 and c.get(0, 3) == 1.6
    assert temp_bucket(2, 2) == "noul:2" and temp_bucket(0, 7) == "choice:6-10"


def test_decode_answers_are_jev_shaped():
    calib = Calibration()
    q = parse_question("d", {"type": "choice", "instructions": "x", "criteria": ["a", "b"]})
    a = decode_answer(q, np.array([2.0, 0.0]), calib)
    assert a["type"] == "choice" and a["choice"] == "a" and abs(sum(a["probabilities"].values()) - 1) < 1e-3
    q = parse_question("s", {"type": "score", "instructions": "x", "criteria": ["l0", "l1", "l2"]})
    a = decode_answer(q, np.array([0.0, 0.0, 10.0]), calib)
    assert a["score"] > 1.9 and set(a["legend"]) == {"0", "1", "2"}
    q = parse_question("n", {"type": "noul", "instructions": "x"})
    a = decode_answer(q, np.array([0.0, 0.0]), calib, act_prob=0.9)
    assert a["noul"] == 0.5 and a["confidence"] == 0.5 and a["act_probability"] == 0.9


def test_parse_questions_requires_object():
    with pytest.raises(QuestionError):
        parse_questions([{"type": "noul"}])
