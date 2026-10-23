"""Run history in SQLite (standard library, no server, works on Streamlit Cloud)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Columns added after the first release. Older databases get them via ALTER TABLE.
_ADDED_COLUMNS = {"judge_status": "TEXT", "judge_attempts": "INTEGER"}


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, "
        "dataset TEXT, created_at TEXT, n_results INTEGER, pass_rate REAL, demo INTEGER)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS results (run_id INTEGER, case_id TEXT, prompt_id TEXT, "
        "label TEXT, provider TEXT, model TEXT, question TEXT, expected TEXT, context TEXT, output TEXT, "
        "latency_ms INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, attempts INTEGER, "
        "exact INTEGER, contains INTEGER, token_f1 REAL, groundedness REAL, judge_correct INTEGER, "
        "judge_grounded INTEGER, judge_reason TEXT, correct INTEGER, grounded INTEGER, passed INTEGER, "
        "paid_equiv_usd REAL, cost_usd REAL, error TEXT)"
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    for name, sql_type in _ADDED_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE results ADD COLUMN {name} {sql_type}")
    conn.commit()
    return conn


def save_run(path: Path, name: str, dataset: str, df: pd.DataFrame, demo: bool = False) -> int:
    conn = _connect(path)
    try:
        passed = df["passed"].dropna()
        rate = float(passed.astype(bool).mean()) if len(passed) else None
        cur = conn.execute(
            "INSERT INTO runs (name, dataset, created_at, n_results, pass_rate, demo) VALUES (?,?,?,?,?,?)",
            (name, dataset, datetime.now(timezone.utc).isoformat(timespec="seconds"), len(df), rate, int(demo)),
        )
        run_id = int(cur.lastrowid)
        out = df.copy()
        out.insert(0, "run_id", run_id)
        out.to_sql("results", conn, if_exists="append", index=False)
        conn.commit()
        return run_id
    finally:
        conn.close()


def list_runs(path: Path) -> pd.DataFrame:
    conn = _connect(path)
    try:
        return pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC", conn)
    finally:
        conn.close()


def load_run(path: Path, run_id: int) -> pd.DataFrame:
    conn = _connect(path)
    try:
        df = pd.read_sql_query("SELECT * FROM results WHERE run_id = ?", conn, params=(int(run_id),))
    finally:
        conn.close()
    df = df.drop(columns=["run_id"])
    for col in ("exact", "contains", "judge_correct", "judge_grounded", "correct", "grounded", "passed"):
        df[col] = df[col].map(lambda v: None if pd.isna(v) else bool(v))
    return df


def previous_run_id(path: Path, dataset: str, before_id: int) -> int | None:
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT id FROM runs WHERE dataset = ? AND id < ? ORDER BY id DESC LIMIT 1", (dataset, int(before_id))
        ).fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()
