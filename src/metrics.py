"""Deterministic scoring. Cheap, reproducible, and free. Judge-based scoring lives in judge.py.

Definitions (all lowercase, punctuation removed):
* exact:        normalized output == normalized expected
* contains:     expected answer appears as a contiguous phrase inside the output
                ("The answer is 20 days." contains "20 days")
* token_f1:     word-overlap F1 (partial credit only; never decides correctness alone)
* groundedness: share of the output's content words that also appear in the context.
                A proxy for "supported by the context". Judge scoring is stronger.
"""
from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b")


def canonical(text: str) -> str:
    """Make equivalent number formats look the same before comparing.

    "$45" -> "45 dollars", "9,000" -> "9000", "Four" -> "4". Only formats are changed, never values.
    """
    t = text or ""
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)            # 9,000 -> 9000
    t = re.sub(r"\$\s*(\d+(?:\.\d+)?)", r"\1 dollars", t)    # $45 -> 45 dollars
    t = _NUMBER_WORD_RE.sub(lambda m: _NUMBER_WORDS[m.group(1)], t.lower())  # four -> 4
    return t
# Function words plus meta-words that describe the answer rather than state facts.
# Without the meta-words, "The answer is 4" is penalised for containing "answer".
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "answer answers question context according says said text information provided".split()
)


def tokens(text: str) -> list[str]:
    return _WORD.findall(canonical(text))


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def exact_match(output: str, expected: str) -> bool:
    return bool(normalize(expected)) and normalize(output) == normalize(expected)


def contains_answer(output: str, expected: str) -> bool:
    needle = tokens(expected)
    hay = tokens(output)
    if not needle:
        return False
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def token_f1(output: str, expected: str) -> float:
    pred = Counter(tokens(output))
    gold = Counter(tokens(expected))
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def groundedness(output: str, context: str) -> float | None:
    """Share of content words in the output that appear in the context. None if no context."""
    if not (context or "").strip():
        return None
    words = content_tokens(output)
    if not words:
        return None
    ctx = set(tokens(context))
    return sum(1 for w in words if w in ctx) / len(words)


def score_deterministic(output: str, expected: str = "", context: str = "") -> dict:
    has_expected = bool((expected or "").strip())
    return {
        "exact": exact_match(output, expected) if has_expected else None,
        "contains": contains_answer(output, expected) if has_expected else None,
        "token_f1": round(token_f1(output, expected), 4) if has_expected else None,
        "groundedness": None if groundedness(output, context) is None else round(groundedness(output, context), 4),
    }


_UNIT_WORDS = {"dollars", "dollar", "cents", "days", "day", "seconds", "minutes", "hours", "million",
               "billion", "kilometres", "kilometers", "km", "years", "people", "letters", "pm", "am"}


def numeric_reference(expected: str) -> float | None:
    """The expected answer as a number, if it is one number plus optional unit words, e.g. '20 days'."""
    toks = tokens(expected)
    if not toks:
        return None
    nums = [t for t in toks if t.replace(".", "", 1).isdigit()]
    others = [t for t in toks if t not in nums]
    if len(nums) != 1 or any(t not in _UNIT_WORDS for t in others):
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", canonical(text))]


def expected_number_present(output: str, expected: str) -> bool | None:
    """True/False when the expected answer is a single number, else None.

    A numeric answer that never appears in the output is wrong, whatever the working looks like.
    This is a one-way check: it can fail an answer, never pass one on its own.
    """
    ref = numeric_reference(expected)
    if ref is None:
        return None
    return any(abs(n - ref) < 1e-9 for n in numbers_in(output))