"""Model catalog, secrets lookup and paths.

Groq's free tier is the default. Prices below are *reference* list prices used only to
show what the same usage would cost on a paid plan. Free-tier usage is billed at $0.
Verify model IDs at https://console.groq.com/docs/models before a demo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "golden_datasets"
HISTORY_DB = DATA_DIR / "eval_history.db"
BUILTIN_DATASET = DATASET_DIR / "eval_suite.csv"


@dataclass(frozen=True)
class ModelSpec:
    provider: str          # "groq" | "openai" | "anthropic" | "demo"
    model: str
    label: str
    price_in: float        # reference USD per 1M input tokens
    price_out: float       # reference USD per 1M output tokens
    free_tier: bool = False

MODELS: list[ModelSpec] = [
    ModelSpec("groq", "allam-2-7b", "ALLaM 2 7B · Groq", 0.20, 0.20, True),
    ModelSpec("groq", "openai/gpt-oss-20b", "GPT-OSS 20B · Groq", 0.20, 0.30, True),
    ModelSpec("groq", "openai/gpt-oss-120b", "GPT-OSS 120B · Groq", 0.60, 0.80, True),
    ModelSpec("openai", "gpt-4o-mini", "GPT-4o mini · OpenAI (paid)", 0.15, 0.60),
    ModelSpec("anthropic", "claude-3-haiku-20240307", "Claude 3 Haiku · Anthropic (paid)", 0.25, 1.25),
]

DEMO_MODELS: list[ModelSpec] = [
    ModelSpec("demo", "demo-strong", "Simulated strong · demo", 0.0, 0.0, True),
    ModelSpec("demo", "demo-mid", "Simulated mid · demo", 0.0, 0.0, True),
    ModelSpec("demo", "demo-small", "Simulated small · demo", 0.0, 0.0, True),
]

# The judge runs on the free Groq tier by default. Change here to use a different model.
JUDGE_MODEL = ModelSpec("groq", "openai/gpt-oss-20b", "Judge · Groq", 0.20, 0.30, True)
# JUDGE_MODEL = ModelSpec("groq", "openai/gpt-oss-120b", "Judge · Groq", 0.60, 0.80, True)
# JUDGE_MODEL = ModelSpec("groq", "allam-2-7b", "Judge · Groq", 0.20, 0.20, True)

PROVIDER_ENV = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def get_secret(name: str) -> str:
    """Read a key from Streamlit secrets, then environment variables. Never from a file in the repo."""
    try:
        import streamlit as st  # imported lazily so pure modules and tests don't need Streamlit

        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:  # noqa: BLE001 - no secrets.toml present locally
        pass
    return os.getenv(name, "")
