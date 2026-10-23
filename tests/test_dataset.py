import pytest

from src.dataset import load_bytes, normalize_columns
import pandas as pd


def test_aliases_are_mapped():
    df = pd.DataFrame({"question": ["q1"], "ground_truth": ["a1"], "ctx": ["c1"]})
    out = normalize_columns(df)
    assert list(out.columns) == ["id", "prompt", "expected", "context"]
    assert out.loc[0, "prompt"] == "q1" and out.loc[0, "expected"] == "a1"


def test_missing_prompt_column_raises():
    with pytest.raises(ValueError, match="prompt"):
        normalize_columns(pd.DataFrame({"foo": [1]}))


def test_blank_prompts_are_dropped_and_ids_filled():
    df = pd.DataFrame({"prompt": ["a", "  ", "c"]})
    out = normalize_columns(df)
    assert len(out) == 2
    assert list(out["id"]) == ["1", "2"]  # IDs are assigned after blank rows are dropped


def test_all_blank_prompts_raise():
    with pytest.raises(ValueError):
        normalize_columns(pd.DataFrame({"prompt": ["", " "]}))


def test_json_with_examples_wrapper():
    raw = b'{"examples": [{"prompt": "p", "expected": "e"}]}'
    out = load_bytes("set.json", raw)
    assert out.loc[0, "expected"] == "e"


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        load_bytes("notes.txt", b"hi")


def test_builtin_suite_loads_and_has_12_cases():
    from src.config import BUILTIN_DATASET
    from src.dataset import load_file

    df = load_file(BUILTIN_DATASET)
    assert len(df) == 12
    assert df["expected"].str.strip().ne("").sum() == 12
