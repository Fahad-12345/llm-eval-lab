"""Self-contained HTML report that can be attached to a LinkedIn post or shared as a file.

The report states its own limitations. Notes that depend on the run (fallback verdicts,
self-judging, rate-limit errors) appear only when they apply.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd


def _esc(v) -> str:
    return html.escape("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))


def limitations_for(results: pd.DataFrame, judge_model: str = "") -> list[str]:
    """Plain-language notes on how to read this run. Used by both the app and the HTML report."""
    notes: list[str] = []
    n = len(results)
    judge_on = "judge_status" in results and (results["judge_status"].fillna("off") != "off").any()

    if judge_on:
        notes.append("How scores are made: each answer is checked against the expected answer by exact and "
                     "partial-match rules, and an LLM judge decides pass or fail. Where the judge failed, the "
                     "deterministic rules decide instead, and that is marked.")
    else:
        notes.append("How scores are made: each answer is checked against the expected answer by exact and "
                     "partial-match rules only. No judge was used in this run.")

    notes.append(f"This is a single run of {n} results. Pass rates can move between runs; "
                 "no repeat runs are included, so treat the numbers as a snapshot.")

    if judge_on:
        notes.append("Judge accuracy is not measured in this report. The LLM judge's verdicts have not been "
                     "checked against human labels here, so judged results are unverified until that check is done.")
    else:
        notes.append("The LLM judge was off for this run. Verdicts come from deterministic checks only, which "
                     "mark correct answers wrong when they differ in units or wording (for example '$45' vs "
                     "'45 dollars') and use word overlap as a proxy for grounding.")

    models = set(results["model"].dropna()) if "model" in results else set()
    if judge_on and judge_model and judge_model in models:
        notes.append(f"Self-judging: the judge model ({judge_model}) is also one of the models under test. "
                     "Its verdicts on its own answers may be more lenient than a neutral judge's.")

    if "judge_status" in results:
        fallback = int(results["judge_status"].isin(["error", "parse_error"]).sum())
        if fallback:
            notes.append(f"{fallback} result(s) were scored by deterministic checks only because the judge call "
                         "failed. Those verdicts are marked in the summary and can be wrong for unit or wording differences.")

    errors = int(results["error"].fillna("").ne("").sum()) if "error" in results else 0
    if errors:
        rate_limited = int(results["error"].fillna("").str.contains("429|rate limit", case=False, regex=True).sum())
        notes.append(f"{errors} call(s) returned an error and count as not passed in the pass rate. "
                     f"{rate_limited} of them were rate-limit errors from the provider's free-tier budget, "
                     "not model answers. The 'answered' column shows the rate over answered calls only.")

    notes.append("The suite is small (28 cases in the built-in hard set), so differences of a few cases are "
                 "within the 95% intervals shown in the summary. Read the intervals before comparing models.")
    return notes


# Backwards-compatible name used by earlier code and tests.
_limitations = limitations_for


def build_html(title: str, summary: pd.DataFrame, results: pd.DataFrame, judge_model: str = "") -> str:
    failures = results[results["passed"] == False]  # noqa: E712
    limitations = limitations_for(results, judge_model)
    judge_failed = results[results["judge_status"].isin(["error", "parse_error"])] \
        if "judge_status" in results else results.iloc[0:0]
    warning = (f"<div class='warn'>{len(judge_failed)} result(s) were scored WITHOUT the judge because "
               f"the judge call failed. Their pass/fail uses deterministic checks only.</div>") if len(judge_failed) else ""
    limitations_html = "".join(f"<li>{_esc(note)}</li>" for note in limitations)
    summary_rows = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in r) + "</tr>"
        for r in summary.itertuples(index=False)
    )
    summary_head = "".join(f"<th>{_esc(c)}</th>" for c in summary.columns)
    fail_rows = "".join(
        f"<tr><td>{_esc(r.case_id)}</td><td>{_esc(r.prompt_id)}</td><td>{_esc(r.label)}</td>"
        f"<td>{_esc(r.question)}</td><td>{_esc(r.expected)}</td><td>{_esc(r.output)}</td>"
        f"<td>{_esc(r.judge_reason)}</td></tr>"
        for r in failures.itertuples(index=False)
    ) or "<tr><td colspan='7'>No failing cases.</td></tr>"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    judge_line = f"Judge: {_esc(judge_model)}" if judge_model else "Judge: not recorded"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{_esc(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#1b1f24;background:#fafbfc}}
h1{{margin-bottom:.2rem}} .meta{{color:#666;margin-bottom:1.5rem}}
table{{border-collapse:collapse;width:100%;margin-bottom:2rem;background:#fff}}
th,td{{border:1px solid #d9dee4;padding:.45rem .6rem;text-align:left;vertical-align:top;font-size:.9rem}}
th{{background:#eef2f6}}
.limits{{background:#f3f6fa;border:1px solid #c9d4e3;padding:.8rem 1.2rem;margin-bottom:2rem}}
.limits li{{margin-bottom:.4rem}}
.warn{{background:#fff4d6;border:1px solid #e0b84a;padding:.6rem .8rem;margin-bottom:1.5rem}}
</style></head><body>
<h1>{_esc(title)}</h1><div class="meta">Generated {stamp} · {len(results)} results · {judge_line}</div>
{warning}
<div class="limits"><h2 style="margin-top:0">Limitations of this report</h2><ul>{limitations_html}</ul></div>
<h2>Summary by model</h2><table><tr>{summary_head}</tr>{summary_rows}</table>
<h2>Failing cases</h2><table><tr><th>Case</th><th>Prompt</th><th>Model</th><th>Question</th><th>Expected</th><th>Output</th><th>Judge note</th></tr>{fail_rows}</table>
</body></html>"""
