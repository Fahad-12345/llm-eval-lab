import pandas as pd

from src.analysis import regressions, summarize
from src.history import list_runs, load_run, previous_run_id, save_run
from src.report import build_html


def _results(passed_flags, label="M · test"):
    rows = []
    for i, p in enumerate(passed_flags):
        rows.append({
            "case_id": str(i + 1), "prompt_id": "plain", "provider": "groq", "model": "m", "label": label,
            "question": f"Q{i}", "expected": "A", "context": "", "output": "A" if p else "B",
            "latency_ms": 100 + i, "prompt_tokens": 10, "completion_tokens": 2, "attempts": 1,
            "exact": p, "contains": p, "token_f1": 1.0 if p else 0.0, "groundedness": None,
            "judge_correct": None, "judge_grounded": None, "judge_reason": "",
            "correct": p, "grounded": None, "passed": p, "paid_equiv_usd": 0.001,
            "cost_usd": 0.0, "error": None,
        })
    return pd.DataFrame(rows)


def test_summary_pass_rate_and_latency(tmp_path):
    df = _results([True, True, False, True])
    s = summarize(df)
    assert s.loc[0, "pass_rate"] == 0.75
    assert s.loc[0, "latency_p50_ms"] == 102.0  # median of 100..103
    assert s.loc[0, "cost_usd"] == 0.0 and s.loc[0, "paid_equiv_usd"] > 0


def test_history_roundtrip_and_previous_run(tmp_path):
    db = tmp_path / "h.db"
    first = save_run(db, "run A", "suite", _results([True, True]))
    second = save_run(db, "run B", "suite", _results([True, False]))
    assert previous_run_id(db, "suite", second) == first
    assert previous_run_id(db, "other", second) is None
    runs = list_runs(db)
    assert list(runs["id"]) == [second, first]
    loaded = load_run(db, second)
    assert len(loaded) == 2 and loaded["passed"].tolist() == [True, False]


def test_regression_detects_pass_to_fail(tmp_path):
    before = _results([True, True, False])
    after = _results([True, False, True])
    broke = regressions(after, before)
    assert list(broke["case_id"]) == ["2"]


def test_regression_detects_new_error():
    before = _results([True])
    after = _results([True]).assign(error=["timeout"])
    broke = regressions(after, before)
    assert len(broke) == 1


def test_report_escapes_html_and_lists_failures():
    df = _results([True, False])
    df.loc[1, "question"] = "<script>alert(1)</script>"
    html = build_html("Report <b>", summarize(df), df)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "Failing cases" in html


def test_old_database_is_migrated_with_new_columns(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)  # schema as shipped in the first release (no judge_status/judge_attempts)
    conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, dataset TEXT, "
                 "created_at TEXT, n_results INTEGER, pass_rate REAL, demo INTEGER)")
    conn.execute("CREATE TABLE results (run_id INTEGER, case_id TEXT, prompt_id TEXT, label TEXT, "
                 "provider TEXT, model TEXT, question TEXT, expected TEXT, context TEXT, output TEXT, "
                 "latency_ms INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, attempts INTEGER, "
                 "exact INTEGER, contains INTEGER, token_f1 REAL, groundedness REAL, judge_correct INTEGER, "
                 "judge_grounded INTEGER, judge_reason TEXT, correct INTEGER, grounded INTEGER, passed INTEGER, "
                 "paid_equiv_usd REAL, cost_usd REAL, error TEXT)")
    conn.commit()
    conn.close()
    df = _results([True, False]).assign(judge_status="ok", judge_attempts=1)
    run_id = save_run(db, "after upgrade", "suite", df)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(results)")}
    assert {"judge_status", "judge_attempts"} <= cols
    assert len(load_run(db, run_id)) == 2


def test_summary_counts_judge_errors():
    df = _results([True, True]).assign(judge_status=["ok", "error"])
    s = summarize(df)
    assert s.loc[0, "judge_errors"] == 1


def test_report_warns_when_judge_failed_and_shows_prompt():
    df = _results([True, False]).assign(judge_status=["error", "ok"], prompt_id="careful")
    html = build_html("t", summarize(df), df)
    assert "scored WITHOUT the judge" in html
    assert "<th>Prompt</th>" in html


def test_errors_count_against_pass_rate():
    df = _results([True, True, True, True]).copy()
    df["passed"] = df["passed"].astype(object)  # the runner stores passed as object so it can hold None
    df.loc[2:3, "error"] = "RateLimitError: 429"
    df.loc[2:3, "passed"] = None
    s = summarize(df)
    assert s.loc[0, "calls"] == 4
    assert s.loc[0, "answered"] == 2
    assert s.loc[0, "errors"] == 2
    assert s.loc[0, "pass_rate"] == 0.5  # 2 passed out of 4 calls, not 2 out of 2 answered


def test_wilson_interval_is_wide_for_small_samples_and_contains_rate():
    from src.analysis import wilson_interval
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi
    assert hi - lo > 0.3          # 8 of 10 is a very uncertain rate
    lo2, hi2 = wilson_interval(80, 100)
    assert hi2 - lo2 < 0.2        # 80 of 100 is much tighter
    assert wilson_interval(0, 0) == (None, None)


def test_summary_reports_judged_share_and_fallback_flag():
    df = _results([True, True, True, True]).assign(
        judge_status=["ok", "ok", "error", "ok"])
    s = summarize(df)
    assert s.loc[0, "judged"] == 3
    assert s.loc[0, "judged_pct"] == 0.75
    assert s.loc[0, "judge_errors"] == 1
    assert "deterministic checks only" in s.loc[0, "fallback_flag"]
    assert s.loc[0, "pass_ci_low"] <= s.loc[0, "pass_rate"] <= s.loc[0, "pass_ci_high"]


def test_no_fallback_means_empty_flag():
    df = _results([True, False]).assign(judge_status=["ok", "ok"])
    s = summarize(df)
    assert s.loc[0, "fallback_flag"] == ""


def test_report_has_limitations_and_self_judge_warning_when_judge_is_under_test():
    df = _results([True, False]).assign(judge_status=["ok", "ok"], model="judge-model")
    html = build_html("t", summarize(df), df, judge_model="judge-model")
    assert "Limitations of this report" in html
    assert "Self-judging" in html


def test_report_no_self_judge_note_for_independent_judge():
    df = _results([True, False]).assign(judge_status=["ok", "ok"], model="other-model")
    html = build_html("t", summarize(df), df, judge_model="judge-model")
    assert "Self-judging" not in html
    assert "Judge accuracy is not measured" in html


def test_report_notes_judge_off_and_rate_limit_errors():
    df = _results([True, True]).assign(judge_status=["off", "off"],
                                       error=[None, "RateLimitError: Error code: 429"])
    html = build_html("t", summarize(df), df)
    assert "The LLM judge was off" in html
    assert "rate-limit" in html


def test_limitations_explain_how_scores_are_made_with_and_without_judge():
    from src.report import limitations_for
    with_judge = _results([True]).assign(judge_status=["ok"])
    without = _results([True]).assign(judge_status=["off"])
    assert any("LLM judge decides" in n for n in limitations_for(with_judge, "x"))
    assert any("No judge was used" in n for n in limitations_for(without, "x"))


def test_limitations_helper_matches_report_content():
    from src.report import limitations_for, build_html
    df = _results([True, False]).assign(judge_status=["ok", "ok"], model="judge-model")
    notes = limitations_for(df, "judge-model")
    html = build_html("t", summarize(df), df, judge_model="judge-model")
    for note in notes:
        assert note.replace("'", "&#x27;")[:40] in html or note[:40] in html
