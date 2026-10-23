"""Aggregate results into dashboard numbers and detect regressions between two runs."""
from __future__ import annotations

import math

import pandas as pd


def wilson_interval(passed: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a pass rate. Honest about small samples."""
    if n == 0:
        return None, None
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def p50(s: pd.Series) -> float:
    return float(s.quantile(0.50)) if len(s) else float("nan")


def p95(s: pd.Series) -> float:
    return float(s.quantile(0.95)) if len(s) else float("nan")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per prompt and model.

    pass_rate counts every call in the denominator, so a call that errored counts as not passed.
    correct_rate, grounded_rate and latency use only the calls that were answered.
    """
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (prompt_id, label), g in df.groupby(["prompt_id", "label"], sort=True):
        has_error = g["error"].fillna("").ne("")
        answered = g[~has_error]
        calls = len(g)
        passed_count = int(g["passed"].fillna(False).astype(bool).sum())
        fallback = int(g["judge_status"].isin(["error", "parse_error"]).sum()) if "judge_status" in g else 0
        judged = int((answered["judge_status"] == "ok").sum()) if "judge_status" in answered else 0
        ci_low, ci_high = wilson_interval(passed_count, calls)
        rows.append({
            "prompt_id": prompt_id,
            "model": label,
            "calls": calls,
            "answered": len(answered),
            "errors": int(has_error.sum()),
            "judged": judged,
            "judged_pct": round(judged / len(answered), 3) if len(answered) else None,
            "judge_errors": fallback,
            "fallback_flag": (f"{fallback} verdict(s) from deterministic checks only (judge failed)"
                              if fallback else ""),
            "pass_rate": round(passed_count / calls, 3) if calls else None,
            "pass_ci_low": ci_low,
            "pass_ci_high": ci_high,
            "correct_rate": round(float(answered["correct"].dropna().astype(bool).mean()), 3)
            if answered["correct"].notna().any() else None,
            "grounded_rate": round(float(answered["grounded"].dropna().astype(bool).mean()), 3)
            if answered["grounded"].notna().any() else None,
            "latency_p50_ms": round(p50(answered["latency_ms"]), 0) if len(answered) else None,
            "latency_p95_ms": round(p95(answered["latency_ms"]), 0) if len(answered) else None,
            "paid_equiv_usd": round(float(answered["paid_equiv_usd"].sum()), 6),
            "cost_usd": round(float(answered["cost_usd"].sum()), 6),
        })
    return pd.DataFrame(rows)


def regressions(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Cases that passed in the previous run and fail now, or that now error."""
    key = ["case_id", "prompt_id", "label"]
    cur = current[key + ["passed", "error", "output"]]
    prev = previous[key + ["passed", "output"]].rename(columns={"passed": "passed_before", "output": "output_before"})
    merged = cur.merge(prev, on=key, how="inner")
    broke = merged[(merged["passed_before"] == True) & ((merged["passed"] == False) | merged["error"].fillna("").ne(""))]  # noqa: E712
    return broke.reset_index(drop=True)
