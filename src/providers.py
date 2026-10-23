"""Model adapters. Each returns a CallResult and never raises: errors are recorded in the result.

Retries use exponential backoff, but only for rate limits (429) and transient server errors.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

from src.config import ModelSpec

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Output-token cap per model. Groq enforces per-minute output-token limits per model, and a
# request that reserves more output tokens than the limit fails with "Request too large".
# Qwen's account limit is 1000 output tokens per minute, so its cap must stay below that.
DEFAULT_MAX_TOKENS = 1024
MAX_TOKENS = {"qwen/qwen3.8-27b": 800}


def max_tokens_for(model: str) -> int:
    return MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS)


@dataclass
class CallResult:
    output: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    attempts: int = 1


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code


# Groq's error text says how long to wait, e.g. "Please try again in 7m35.76s" or "in 2.5s".
_RETRY_HINT = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE)
# Errors that waiting cannot fix. Per-day quotas reset hours later, not in seconds.
_NEVER_RETRY = ("request too large", "tokens per day", "(tpd)", "requests per day", "(rpd)")
MAX_WAIT_PER_CALL = 20.0  # seconds; a visitor should never wait minutes for one answer


def _retry_hint_seconds(message: str):
    match = _RETRY_HINT.search(message)
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    return minutes * 60 + float(match.group(2))


def with_retry(fn, max_attempts: int = 4, base_delay: float = 2.0, sleep=time.sleep,
               max_total_wait: float = MAX_WAIT_PER_CALL):
    """Call fn() and return (value, attempts).

    Retries only transient errors (429 per-minute limits, 5xx). Daily quotas and oversize
    requests fail immediately. Total waiting per call is capped so one slow call cannot stall a run.
    """
    attempt = 0
    waited = 0.0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except Exception as exc:  # noqa: BLE001
            status = _status(exc)
            message = str(exc).lower()
            retryable = status in RETRYABLE_STATUS or (status is None and "rate limit" in message)
            if any(marker in message for marker in _NEVER_RETRY):
                retryable = False
            if not retryable or attempt >= max_attempts:
                raise
            hint = _retry_hint_seconds(message)
            delay = (hint + 0.5) if hint is not None else base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0, 0.25)
            if hint is not None and hint > max_total_wait:
                raise  # the limit resets later than we are willing to wait; fail now
            if waited + delay > max_total_wait:
                raise
            waited += delay
            sleep(delay)


def _call_groq(api_key: str, model: str, messages: list[dict]):
    from groq import Groq

    resp = Groq(api_key=api_key).chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens_for(model)
    )
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def _call_openai(api_key: str, model: str, messages: list[dict]):
    from openai import OpenAI

    resp = OpenAI(api_key=api_key).chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens_for(model)
    )
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def _call_anthropic(api_key: str, model: str, messages: list[dict]):
    from anthropic import Anthropic

    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    resp = Anthropic(api_key=api_key).messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=system or "Answer accurately and concisely.",
        messages=convo,
    )
    text = "".join(block.text for block in resp.content if getattr(block, "text", None))
    usage = resp.usage
    return text, usage.input_tokens if usage else 0, usage.output_tokens if usage else 0


DISPATCH = {"groq": _call_groq, "openai": _call_openai, "anthropic": _call_anthropic}


def run_model(spec: ModelSpec, api_key: str, messages: list[dict]) -> CallResult:
    start = time.perf_counter()
    if spec.provider == "demo":
        from src.demo import simulate

        text, pt, ct = simulate(messages, spec.model)
        return CallResult(text, int((time.perf_counter() - start) * 1000), pt, ct)
    if not api_key:
        return CallResult(error=f"No API key for {spec.provider}.")
    try:
        (text, pt, ct), attempts = with_retry(lambda: DISPATCH[spec.provider](api_key, spec.model, messages))
        latency = int((time.perf_counter() - start) * 1000)
        if not (text or "").strip():
            # Reasoning models can spend the whole output budget on reasoning and return no text.
            # That is a truncation, not a wrong answer, so record it as an error.
            return CallResult("", latency, pt, ct,
                              f"EmptyOutput: no text returned after {ct} completion tokens "
                              f"(output limit {max_tokens_for(spec.model)} likely reached)", attempts)
        return CallResult(text, latency, pt, ct, None, attempts)
    except Exception as exc:  # noqa: BLE001 - surface provider errors in the results table
        return CallResult(error=f"{type(exc).__name__}: {exc}"[:300],
                          latency_ms=int((time.perf_counter() - start) * 1000))
