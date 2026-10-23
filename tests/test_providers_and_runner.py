import pandas as pd

from src.config import ModelSpec
from src.prompts import build_messages, render
from src.providers import CallResult, run_model, with_retry
from src.runner import Plan, run_plan

SPEC = ModelSpec("groq", "fake-model", "Fake · test", 0.60, 0.80, True)


class FakeHTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def test_retry_succeeds_after_rate_limit():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeHTTPError(429)
        return "ok"

    value, attempts = with_retry(flaky, base_delay=0, sleep=lambda s: None)
    assert value == "ok" and attempts == 3


def test_non_retryable_error_is_raised_immediately():
    def bad():
        raise FakeHTTPError(401)

    try:
        with_retry(bad, sleep=lambda s: None)
    except FakeHTTPError as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("expected 401 to propagate")


def test_run_model_records_missing_key_without_raising():
    res = run_model(SPEC, "", [{"role": "user", "content": "hi"}])
    assert res.error and "No API key" in res.error


def test_render_and_messages():
    assert render("Answer: {input}", " Q? ") == "Answer: Q?"
    msgs = build_messages("Q", "CTX")
    assert msgs[0]["role"] == "system" and "CTX" in msgs[0]["content"]
    assert build_messages("Q")[0]["role"] == "user"


def _fake_call(spec, key, messages):
    # Judge calls have a system message that starts with the evaluator instruction.
    if messages[0]["content"].startswith("You are a strict evaluator"):
        judged_correct = "Model answer: Paris" in messages[1]["content"]
        return CallResult(output='{"correct": %s, "grounded": true, "reason": "fine"}'
                          % ("true" if judged_correct else "false"), latency_ms=5)
    question = messages[-1]["content"]
    answer = "Paris." if "France" in question else "I do not know."
    return CallResult(output=answer, latency_ms=10, prompt_tokens=100, completion_tokens=5)


def test_runner_grid_size_and_pass_logic():
    cases = pd.DataFrame({"id": ["1", "2"], "prompt": ["What is the capital of France?", "Unknown?"],
                          "expected": ["Paris", "Paris"], "context": ["", ""]})
    plan = Plan(cases=cases, templates={"plain": "{input}", "short": "Short: {input}"},
                models=[SPEC], keys={"groq": "k"}, use_judge=True, max_workers=2)
    assert plan.call_count() == 2 * 2 * 1 * 2
    df = run_plan(plan, call_fn=_fake_call)
    assert len(df) == 4
    ok = df[df["case_id"] == "1"]
    assert ok["passed"].all()
    bad = df[df["case_id"] == "2"]
    assert not bad["passed"].any()
    assert (df["cost_usd"] == 0).all()          # free tier bills $0
    assert (df["paid_equiv_usd"] > 0).all()      # but we still show the paid equivalent


def test_runner_keeps_errors_without_crashing():
    def err_call(spec, key, messages):
        return CallResult(error="HTTP 500: boom")

    cases = pd.DataFrame({"id": ["1"], "prompt": ["Q"], "expected": ["A"], "context": [""]})
    plan = Plan(cases=cases, templates={"plain": "{input}"}, models=[SPEC], keys={"groq": "k"}, max_workers=1)
    df = run_plan(plan, call_fn=err_call)
    assert df.loc[0, "error"] == "HTTP 500: boom"
    assert pd.isna(df.loc[0, "passed"]) or df.loc[0, "passed"] is None


def test_judge_failure_is_recorded_not_hidden():
    def judge_fails(spec, key, messages):
        if messages[0]["content"].startswith("You are a strict evaluator"):
            return CallResult(error="RateLimitError: 429", attempts=5)
        return CallResult(output="Paris.", latency_ms=10, prompt_tokens=1, completion_tokens=1)

    cases = pd.DataFrame({"id": ["1"], "prompt": ["What is the capital of France?"],
                          "expected": ["Paris"], "context": [""]})
    plan = Plan(cases=cases, templates={"plain": "{input}"}, models=[SPEC], keys={"groq": "k"},
                use_judge=True, max_workers=1)
    df = run_plan(plan, call_fn=judge_fails)
    row = df.iloc[0]
    assert row["judge_status"] == "error"
    assert row["judge_attempts"] == 5
    assert "deterministic checks only" in row["judge_reason"]
    assert row["passed"] == True  # deterministic fallback still decides, but it is labelled


def test_judge_ok_status_when_judge_runs():
    cases = pd.DataFrame({"id": ["1"], "prompt": ["What is the capital of France?"],
                          "expected": ["Paris"], "context": [""]})
    plan = Plan(cases=cases, templates={"plain": "{input}"}, models=[SPEC], keys={"groq": "k"},
                use_judge=True, max_workers=1)
    df = run_plan(plan, call_fn=_fake_call)
    assert df.iloc[0]["judge_status"] == "ok"


def test_judge_status_off_when_judge_disabled():
    cases = pd.DataFrame({"id": ["1"], "prompt": ["Q"], "expected": ["A"], "context": [""]})
    plan = Plan(cases=cases, templates={"plain": "{input}"}, models=[SPEC], keys={"groq": "k"},
                use_judge=False, max_workers=1)
    df = run_plan(plan, call_fn=_fake_call)
    assert df.iloc[0]["judge_status"] == "off"


def test_careful_template_no_longer_forces_context_wording_without_context():
    from src.prompts import DEFAULT_TEMPLATES, SYSTEM_CONTEXT, build_messages, render
    rendered = render(DEFAULT_TEMPLATES["careful"], "What is 17 multiplied by 3?")
    assert "context does not say" not in rendered.lower()
    # The context rule only appears when context is actually sent.
    assert all("context" not in m["content"].lower() for m in build_messages(rendered, ""))
    assert "say that the context does not say" in SYSTEM_CONTEXT


def test_qwen_output_cap_stays_below_account_limit():
    from src.providers import max_tokens_for
    assert max_tokens_for("qwen/qwen3.8-27b") < 1000
    assert max_tokens_for("openai/gpt-oss-120b") == 1024


def test_request_too_large_is_not_retried():
    calls = {"n": 0}

    class Boom(Exception):
        status_code = 429

    def too_big():
        calls["n"] += 1
        raise Boom("Request too large for model on output tokens per minute (OTPM): Limit 1000")

    try:
        with_retry(too_big, sleep=lambda s: None)
    except Boom:
        pass
    assert calls["n"] == 1


def test_normal_rate_limit_is_still_retried():
    calls = {"n": 0}

    class Limited(Exception):
        status_code = 429

    def limited():
        calls["n"] += 1
        if calls["n"] < 2:
            raise Limited("Rate limit reached, try again in 2s")
        return "ok"

    value, attempts = with_retry(limited, sleep=lambda s: None)
    assert value == "ok" and attempts == 2


def test_empty_output_is_recorded_as_error_not_wrong_answer(monkeypatch):
    import src.providers as providers

    def empty_reply(api_key, model, messages):
        return ("", 10, 1024)

    monkeypatch.setitem(providers.DISPATCH, "groq", empty_reply)
    res = run_model(SPEC, "k", [{"role": "user", "content": "hi"}])
    assert res.error and res.error.startswith("EmptyOutput")
    assert res.completion_tokens == 1024
    assert res.output == ""


def test_whitespace_only_output_counts_as_empty(monkeypatch):
    import src.providers as providers

    monkeypatch.setitem(providers.DISPATCH, "groq", lambda k, m, msgs: ("   \n", 5, 3))
    res = run_model(SPEC, "k", [{"role": "user", "content": "hi"}])
    assert res.error and res.error.startswith("EmptyOutput")


def test_daily_quota_is_not_retried():
    calls = {"n": 0}

    class Quota(Exception):
        status_code = 429

    def daily():
        calls["n"] += 1
        raise Quota("Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199920. "
                    "Please try again in 7m35.76s.")

    try:
        with_retry(daily, sleep=lambda s: None)
    except Quota:
        pass
    assert calls["n"] == 1


def test_retry_follows_server_hint_and_respects_total_wait_cap():
    waits = []

    class Limited(Exception):
        status_code = 429

    def always_limited():
        raise Limited("Rate limit reached on tokens per minute (TPM). Please try again in 2.5s.")

    try:
        with_retry(always_limited, max_attempts=10, sleep=waits.append, max_total_wait=20)
    except Limited:
        pass
    assert waits, "expected at least one retry"
    assert 3.0 <= waits[0] <= 3.25  # server hint 2.5s + 0.5s margin + up to 0.25s jitter
    assert sum(waits) <= 20


def test_hint_longer_than_cap_fails_immediately():
    calls = {"n": 0}

    class Limited(Exception):
        status_code = 429

    def long_wait():
        calls["n"] += 1
        raise Limited("Rate limit reached. Please try again in 3m0s.")

    try:
        with_retry(long_wait, sleep=lambda s: None)
    except Limited:
        pass
    assert calls["n"] == 1


def test_default_parallel_requests_is_one():
    import re
    src = open("app.py").read()
    assert re.search(r'"Parallel requests", 1, 8, 1,', src)
