"""Combine deterministic metrics and optional judge verdicts into one per-case verdict."""
from __future__ import annotations

from src.metrics import expected_number_present, score_deterministic

GROUNDING_THRESHOLD = 0.6  # used only when no judge verdict is available


def evaluate_case(output: str, expected: str, context: str, judge: dict | None = None) -> dict:
    det = score_deterministic(output, expected, context)

    if judge and judge.get("correct") is not None:
        correct = judge["correct"]
    elif det["contains"] is not None:
        correct = bool(det["exact"] or det["contains"])
    else:
        correct = None  # no reference answer and no judge

    has_context = bool((context or "").strip())
    if not has_context:
        grounded = None  # nothing to ground against, so the judge's grounded value is ignored
    elif judge and judge.get("grounded") is not None:
        grounded = judge["grounded"]
    elif det["groundedness"] is not None:
        grounded = det["groundedness"] >= GROUNDING_THRESHOLD
    else:
        grounded = None

    # Arithmetic guard: if the expected number never appears in the answer, the answer is wrong,
    # even when a judge says otherwise.
    numeric_ok = expected_number_present(output, expected)
    if numeric_ok is False:
        correct = False

    checks = [v for v in (correct, grounded) if v is not None]
    passed = all(checks) if checks else None
    return {**det, "correct": correct, "grounded": grounded, "passed": passed,
            "numeric_ok": numeric_ok}
