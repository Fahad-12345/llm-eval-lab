"""LLM-as-judge: a second model grades correctness and grounding, returning strict JSON.

The judge is itself a model, so treat its verdicts as a strong signal, not ground truth.
Calibrate it against a few answers you label by hand (see README).
"""
from __future__ import annotations

import json
import re

JUDGE_SYSTEM = (
    "You are a strict evaluator of AI answers. Reply with a single JSON object and nothing else.\n"
    'Schema: {"correct": true|false|null, "grounded": true|false|null, "reason": "max 20 words"}\n'
    "- correct: does the answer match the reference answer in meaning? null if no reference is given.\n"
    "- grounded: is every factual claim in the answer supported by the context? "
    "Always null when the context is (none).\n"
    "- A refusal (for example 'I can't help with that') is NOT correct unless the reference says the "
    "information is unavailable AND the answer says the information is unavailable.\n"
    "- An answer saying the information is not available is correct only if the reference says so too."
)


def build_judge_messages(question: str, output: str, expected: str, context: str) -> list[dict]:
    user = (
        f"Question: {question}\n"
        f"Reference answer: {expected or '(none)'}\n"
        f"Context: {context.strip() if context and context.strip() else '(none)'}\n"
        f"Model answer: {output}\n\nReturn the JSON object now."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def _tri(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def parse_judge(text: str) -> dict:
    """Extract the first JSON object from the judge's reply. Raises ValueError if none is valid."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("Judge reply contained no JSON object.")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Judge JSON is not an object.")
    return {
        "correct": _tri(data.get("correct")),
        "grounded": _tri(data.get("grounded")),
        "reason": str(data.get("reason", ""))[:200],
    }
