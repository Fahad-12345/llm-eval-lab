"""Load golden datasets from CSV, JSON or XLSX and normalize their columns."""
from __future__ import annotations

import json
from io import BytesIO

import pandas as pd

REQUIRED = "prompt"
OPTIONAL = ("expected", "context", "id")

ALIASES = {
    "question": "prompt",
    "input": "prompt",
    "query": "prompt",
    "answer": "expected",
    "gold": "expected",
    "reference": "expected",
    "expected_answer": "expected",
    "ground_truth": "expected",
    "ctx": "context",
    "source": "context",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})
    if REQUIRED not in df.columns:
        raise ValueError("Dataset must include a 'prompt' column (aliases: question, input, query).")
    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = ""
    for col in ("prompt", "expected", "context"):
        df[col] = df[col].fillna("").astype(str)
    df["id"] = df["id"].fillna("").astype(str)
    df = df[df["prompt"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("No non-empty prompts found in the dataset.")
    blank = df["id"].str.strip() == ""
    df.loc[blank, "id"] = [str(i + 1) for i in df.index[blank]]
    return df[["id", "prompt", "expected", "context"]]


def load_bytes(name: str, raw: bytes) -> pd.DataFrame:
    lower = name.lower()
    if lower.endswith(".json"):
        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, dict) and "examples" in payload:
            payload = payload["examples"]
        df = pd.DataFrame(payload)
    elif lower.endswith(".csv"):
        df = pd.read_csv(BytesIO(raw))
    elif lower.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(raw))
    else:
        raise ValueError("Upload a CSV, JSON or XLSX file.")
    return normalize_columns(df)


def load_file(path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        return load_bytes(str(path), fh.read())
