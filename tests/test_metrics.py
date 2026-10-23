import pytest

from src.metrics import (contains_answer, exact_match, groundedness, score_deterministic, token_f1)


def test_exact_ignores_case_and_punctuation():
    assert exact_match("Paris.", "paris")
    assert not exact_match("The answer is Paris", "Paris")


def test_contains_finds_answer_inside_sentence():
    assert contains_answer("The answer is 4.", "4")
    assert contains_answer("It lasts 20 days, per policy.", "20 days")
    assert not contains_answer("It lasts 30 days.", "20 days")


def test_contains_does_not_match_inside_other_words():
    # "4" must not match inside "2024"
    assert not contains_answer("Founded in 2024.", "4")


def test_correct_but_worded_differently_is_not_penalised_as_wrong():
    out = "Paris is the capital of France."
    assert contains_answer(out, "Paris")
    assert token_f1(out, "Paris") > 0


def test_wrong_answer_is_not_correct():
    assert not contains_answer("Berlin is the capital of France.", "Paris")


def test_token_f1_bounds():
    assert token_f1("a b c", "a b c") == pytest.approx(1.0)
    assert token_f1("x y", "a b") == 0.0
    assert token_f1("", "") == 1.0


@pytest.mark.parametrize("output, ctx, expected", [
    ("Paris", "Its capital is Paris.", 1.0),
    ("Berlin", "Its capital is Paris.", 0.0),
])
def test_groundedness_basic(output, ctx, expected):
    assert groundedness(output, ctx) == pytest.approx(expected)


def test_groundedness_none_without_context():
    assert groundedness("anything", "") is None


def test_score_deterministic_without_reference_returns_none_fields():
    d = score_deterministic("hello", "", "")
    assert d["exact"] is None and d["contains"] is None and d["groundedness"] is None


def test_currency_and_thousands_separators_match():
    assert contains_answer("The family pays $45.", "45 dollars")
    assert contains_answer("There are 9,000 seconds.", "9000")
    assert contains_answer("The ball costs $0.05 (5 cents).", "0.05 dollars")


def test_number_words_match_digits():
    assert contains_answer("Four.", "4")
    assert contains_answer("There are twenty days.", "20 days")


def test_wrong_numbers_still_fail_after_normalization():
    assert not contains_answer("The family pays $46.", "45 dollars")
    assert not contains_answer("There are 1,800 seconds.", "9000")
    assert not contains_answer("It takes 56 dollars.", "54 dollars")


def test_numeric_reference_parsing():
    from src.metrics import numeric_reference
    assert numeric_reference("20 days") == 20.0
    assert numeric_reference("45 dollars") == 45.0
    assert numeric_reference("18:05") is None
    assert numeric_reference("Paris") is None
    assert numeric_reference("1969") == 1969.0


def test_expected_number_present_guard():
    from src.metrics import expected_number_present
    assert expected_number_present("The family pays 72 dollars.", "45 dollars") is False
    assert expected_number_present("24 + 21 = 45 dollars in total.", "45 dollars") is True
    assert expected_number_present("Banana", "30") is False
    assert expected_number_present("Paris", "Paris") is None
