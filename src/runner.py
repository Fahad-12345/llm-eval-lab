"""Run the grid: prompts x models x cases, optionally judged. Parallel, with progress and errors kept."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from src.config import JUDGE_MODEL, ModelSpec
from src.judge import build_judge_messages, parse_judge
from src.prompts import build_messages, render
from src.providers import CallResult, run_model
from src.scoring import evaluate_case

COLUMNS = [
    "case_id", "prompt_id", "provider", "model", "label", "question", "expected", "context",
    "output", "latency_ms", "prompt_tokens", "completion_tokens", "attempts",
    "exact", "contains", "token_f1", "groundedness",
    "judge_correct", "judge_grounded", "judge_reason", "judge_status", "judge_attempts",
    "correct", "grounded", "passed", "paid_equiv_usd", "cost_usd", "error",
]

# judge_status values: off | ok | error | parse_error | not_run (main call failed)


@dataclass
class Plan:
    cases: pd.DataFrame
    templates: dict[str, str]
    models: list[ModelSpec]
    keys: dict[str, str]
    use_judge: bool = False
    judge_key: str = ""
    max_workers: int = 4
    judge_spec: ModelSpec = field(default=JUDGE_MODEL)

    def call_count(self) -> int:
        base = len(self.cases) * len(self.templates) * len(self.models)
        return base * 2 if self.use_judge else base


def _cost(spec: ModelSpec, pt: int, ct: int) -> float:
    return (pt * spec.price_in + ct * spec.price_out) / 1_000_000


def _run_one(job, call_fn, judge_fn):
    prompt_id, template, spec, row = job
    question = row["prompt"]
    expected = row.get("expected", "") or ""
    context = row.get("context", "") or ""
    rendered = render(template, question)
    messages = build_messages(rendered, context)

    api_key = judge_fn.keys.get(spec.provider, "") if hasattr(judge_fn, "keys") else ""
    res: CallResult = call_fn(spec, api_key, messages)

    record = {
        "case_id": str(row["id"]), "prompt_id": prompt_id, "provider": spec.provider,
        "model": spec.model, "label": spec.label, "question": question, "expected": expected,
        "context": context, "output": res.output, "latency_ms": res.latency_ms,
        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
        "attempts": res.attempts, "error": res.error,
        "judge_status": "off", "judge_attempts": 0,
        "paid_equiv_usd": round(_cost(spec, res.prompt_tokens, res.completion_tokens), 8),
        "cost_usd": 0.0 if spec.free_tier else round(_cost(spec, res.prompt_tokens, res.completion_tokens), 8),
    }
    if res.error:
        for col in ("exact", "contains", "token_f1", "groundedness", "judge_correct",
                    "judge_grounded", "judge_reason", "correct", "grounded", "passed"):
            record[col] = None
        record["judge_status"] = "not_run"
        return record

    judge = None
    if judge_fn.use_judge:
        jmsgs = build_judge_messages(question, res.output, expected, context)
        jres = call_fn(judge_fn.judge_spec, judge_fn.judge_key, jmsgs)
        record["judge_attempts"] = jres.attempts
        if jres.error:
            record["judge_status"] = "error"
            record["judge_reason"] = f"judge failed, deterministic checks only: {jres.error}"[:300]
        else:
            try:
                judge = parse_judge(jres.output)
                record["judge_status"] = "ok"
            except ValueError as exc:
                record["judge_status"] = "parse_error"
                record["judge_reason"] = f"judge reply unreadable, deterministic checks only: {exc}"[:300]

    verdict = evaluate_case(res.output, expected, context, judge)
    record.update({
        "exact": verdict["exact"], "contains": verdict["contains"], "token_f1": verdict["token_f1"],
        "groundedness": verdict["groundedness"],
        "judge_correct": judge["correct"] if judge else None,
        "judge_grounded": judge["grounded"] if judge else None,
        "judge_reason": (judge or {}).get("reason") or record.get("judge_reason", ""),
        "correct": verdict["correct"], "grounded": verdict["grounded"], "passed": verdict["passed"],
    })
    return record


class _Ctx:
    """Carries the per-run judge settings into the worker without globals."""

    def __init__(self, plan: Plan):
        self.keys = plan.keys
        self.use_judge = plan.use_judge
        self.judge_spec = plan.judge_spec
        self.judge_key = plan.judge_key or plan.keys.get(plan.judge_spec.provider, "")


def run_plan(plan: Plan, call_fn=run_model, progress=None) -> pd.DataFrame:
    ctx = _Ctx(plan)
    jobs = [
        (prompt_id, template, spec, row)
        for _, row in plan.cases.iterrows()
        for prompt_id, template in plan.templates.items()
        for spec in plan.models
    ]
    rows: list[dict] = []
    total = len(jobs) or 1
    with ThreadPoolExecutor(max_workers=max(1, plan.max_workers)) as pool:
        futures = [pool.submit(_run_one, job, call_fn, ctx) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if progress:
                progress(done / total)
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COLUMNS].sort_values(["prompt_id", "label", "case_id"]).reset_index(drop=True)
