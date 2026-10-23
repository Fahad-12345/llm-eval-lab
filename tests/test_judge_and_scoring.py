import pytest

from src.judge import build_judge_messages, parse_judge
from src.scoring import evaluate_case


def test_parse_clean_json():
    v = parse_judge('{"correct": true, "grounded": false, "reason": "ok"}')
    assert v == {"correct": True, "grounded": False, "reason": "ok"}


def test_parse_json_inside_prose_and_fences():
    text = 'Here you go:\n```json\n{"correct": "false", "grounded": null, "reason": "wrong"}\n```'
    v = parse_judge(text)
    assert v["correct"] is False and v["grounded"] is None


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_judge("I think it is correct.")


def test_judge_messages_contain_all_fields():
    msgs = build_judge_messages("Q?", "A", "E", "C")
    assert msgs[0]["role"] == "system"
    assert "Q?" in msgs[1]["content"] and "Reference answer: E" in msgs[1]["content"]


def test_scoring_uses_deterministic_when_no_judge():
    v = evaluate_case("The answer is 4.", "4", "2 plus 2 is 4.")
    assert v["correct"] is True
    assert v["grounded"] is True
    assert v["passed"] is True


def test_judge_overrides_deterministic_verdict():
    # Deterministic check says correct (contains "4") but the judge says the reasoning is wrong.
    v = evaluate_case("4 because 2 times 2.", "4", "", judge={"correct": False, "grounded": None, "reason": ""})
    assert v["correct"] is False and v["passed"] is False


def test_no_reference_and_no_context_gives_unknown():
    v = evaluate_case("anything", "", "")
    assert v["correct"] is None and v["grounded"] is None and v["passed"] is None


def test_wrong_answer_fails_even_if_grounded():
    v = evaluate_case("Berlin is the capital.", "Paris", "The capital is Paris.")
    assert v["correct"] is False
    assert v["passed"] is False


def test_grounded_is_none_without_context_even_if_judge_says_true():
    v = evaluate_case("Mars is the Red Planet.", "Mars", "", judge={"correct": True, "grounded": True, "reason": ""})
    assert v["grounded"] is None
    assert v["passed"] is True  # correctness still decides when there is no context


def test_judge_prompt_treats_refusals_as_incorrect_unless_reference_says_unavailable():
    from src.judge import JUDGE_SYSTEM
    assert "refusal" in JUDGE_SYSTEM.lower()
    assert "NOT correct" in JUDGE_SYSTEM


def test_judge_user_message_marks_missing_context_explicitly():
    msgs = build_judge_messages("Q?", "A", "E", "   ")
    assert "Context: (none)" in msgs[1]["content"]


def test_judge_cannot_pass_arithmetic_when_correct_number_is_missing():
    # The earlier real failure: working shows 24 + 21 = 72, judge said it matched 45.
    v = evaluate_case("The family pays 72 dollars. (24 + 21 = 72)", "45 dollars",
                      "2 adults at 12 dollars and 3 children at 7 dollars.",
                      judge={"correct": True, "grounded": True, "reason": "matches"})
    assert v["correct"] is False and v["passed"] is False


def test_correct_number_with_judge_still_passes():
    v = evaluate_case("The family pays 45 dollars.", "45 dollars", "",
                      judge={"correct": True, "grounded": None, "reason": ""})
    assert v["passed"] is True


def test_non_numeric_answers_are_not_affected_by_guard():
    v = evaluate_case("Canberra is the capital.", "Canberra", "",
                      judge={"correct": True, "grounded": None, "reason": ""})
    assert v["passed"] is True
