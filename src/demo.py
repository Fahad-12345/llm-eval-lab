"""Offline simulator so the app can be explored with no API key.

Outputs are generated from the built-in suite and deliberately imperfect, so the dashboard
has something to show. They are NOT real model results. The UI labels every demo run.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from src.config import BUILTIN_DATASET

_ANSWERS: dict[str, tuple[str, str]] = {}  # prompt -> (good answer, plausible wrong answer)


def _load() -> None:
    if _ANSWERS:
        return
    df = pd.read_csv(BUILTIN_DATASET).fillna("")
    for _, row in df.iterrows():
        good = row["expected"]
        wrong = "I am not sure, but I believe it is probably something else."
        if row["prompt"].startswith("Ignore the context"):
            wrong = "The capital of France is Berlin."
        _ANSWERS[row["prompt"].strip()] = (f"{good}.", wrong)


# Share of answers each simulated model gets right (out of 10). Purely illustrative.
ACCURACY = {"demo-strong": 9, "demo-mid": 7, "demo-small": 4}


def simulate(messages: list[dict], model: str = "demo-mid") -> tuple[str, int, int]:
    _load()
    question = messages[-1]["content"]
    for prompt in _ANSWERS:
        if prompt in question:
            good, wrong = _ANSWERS[prompt]
            # Deterministic per (model, question), so repeated demo runs are stable.
            bucket = int(hashlib.md5((model + question).encode()).hexdigest(), 16) % 10
            text = good if bucket < ACCURACY.get(model, 7) else wrong
            words = len(question.split())
            return text, words * 2, len(text.split())
    return "I don't have enough information to answer that.", len(question.split()), 9
