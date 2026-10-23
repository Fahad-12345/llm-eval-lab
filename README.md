# LLM Eval Lab

A free Streamlit app for comparing prompts and models on your own questions. Each answer is scored by
rule-based checks and, optionally, an LLM judge. The app shows its scoring method and its limits next to
every result.

**Live demo:** https://llm-eval-lab-sw6tqzo5cyyalzdsjmtpfy.streamlit.app/
**Demo mode** needs no API key. Its outputs are simulated and labelled as demo everywhere.

<img width="1920" height="968" alt="llm-eval-lab" src="https://github.com/user-attachments/assets/49f904b6-457d-41eb-b118-7f6071574b53" />


## What it does

| Tab | What you get |
|---|---|
| Run | Built-in 12-case suite, or your own CSV, JSON or XLSX. Edit prompt variants in a table (`{input}` marks where the question goes). |
| Dashboard | "How to read these numbers" box, pass rates with 95% intervals, quality-vs-latency chart, latency distribution, summary table. |
| Failures & regressions | Failing cases with the judge's reason, and cases that passed in the last run but fail now. |
| History | Every run is saved to SQLite. Reload any past run. |
| Export | CSV, JSON, and a self-contained HTML report with the same limitations notes as the dashboard. |

## How answers are scored

**Rule-based checks** (always on, free, reproducible):
- Exact match and phrase containment after normalization, so "The answer is 20 days." matches "20 days".
- Number formats are treated as equal: `$45` and "45 dollars", `9,000` and `9000`, `Four` and `4`.
- Arithmetic guard: if the expected answer is a single number and that number does not appear in the
  answer, the answer fails, whatever the judge says.
- Word-overlap F1 as partial credit, and a word-overlap proxy for grounding in the context.

**LLM judge** (optional): a second model returns strict JSON with `correct` and `grounded` verdicts and a
short reason. The judge model is set by `JUDGE_MODEL` in `src/config.py`.

When the judge call fails (for example, a rate limit), the answer is scored by the rule-based checks alone.
That row is labelled as a fallback, and the summary and report count it.

A case passes when every check that applies to it passes. A case with no reference answer and no context
is unknown, not passed.

## Limitations

These are also shown in the app and the report.

- **One run per evaluation.** Language model outputs vary between runs, so scores can move. Repeat runs are
  not built in yet.
- **The judge is not validated on every kind of question.** It agreed with hand labels on 20 of 20 sample
  answers after two corrections. The sample is small and includes several easy prompt-injection cases.
  Arithmetic, paraphrases and "not stated" answers are not covered by that sample.
- **Rule-based checks can be wrong.** For example, "5 cents" does not match "0.05 dollars".
- **Groundedness is a word-overlap proxy.** It can miss a false claim that reuses words from the context.
- **Free-tier limits.** Groq's free tier caps requests and tokens per minute and per day. Runs can be cut
  short. Failed calls are recorded as errors and counted as not passed.
- **Small suite.** 28 cases in the hard set. Differences of a few cases fall within the intervals.

Judge-based and rule-based numbers should not be compared as if they were the same measurement.

## Run locally

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
# open .streamlit\secrets.toml and paste your Groq key
streamlit run app.py
```

macOS / Linux:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

You can skip the secrets file and paste a key into the sidebar instead. The key stays in the session and
is not written to disk.

Run the tests (no API keys needed):

```bash
pytest -q
```

## Get a free Groq key

Create a key at [console.groq.com](https://console.groq.com). Model IDs change over time, so check
[console.groq.com/docs/models](https://console.groq.com/docs/models) and update `MODELS` in `src/config.py`
if one is retired.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub. `.gitignore` excludes `.streamlit/secrets.toml` and private label files.
2. On [share.streamlit.io](https://share.streamlit.io), choose **Create app** → **Deploy a public app from GitHub**.
3. Set the main file path to `app.py` and the Python version to `3.11`.
4. Leave Secrets empty for a public app. Visitors paste their own free key, or use demo mode.

## Bring your own dataset

CSV, JSON or XLSX with a `prompt` column. Optional: `expected`, `context`, `id`.
Aliases: `question`, `input`, `query` for prompt; `answer`, `gold`, `reference`, `ground_truth` for expected;
`ctx`, `source` for context. JSON can be a list of objects or `{"examples": [...]}`.

## Hand-label check

`score_labels.py` compares your own PASS/FAIL labels with the judge's verdicts and reports agreement and
Cohen's kappa. Label the items first, without looking at the judge's verdicts, then run:

```bash
python score_labels.py LABELS.csv KEY.csv
```

`LABELS.csv` needs `sample_id` and `your_verdict (PASS or FAIL)`. `KEY.csv` needs `sample_id`, `judge_passed`
and `judge_reason`.

## Project layout

```
app.py                  Streamlit UI
src/config.py           model catalog, judge model, secrets lookup
src/prompts.py          prompt templates and the shared message format
src/providers.py        Groq, OpenAI and Anthropic adapters, retries, output caps
src/demo.py             offline simulator for demo mode
src/metrics.py          rule-based scoring and number normalization
src/judge.py            LLM-judge prompt and strict JSON parsing
src/scoring.py          combines rule-based and judge verdicts; arithmetic guard
src/runner.py           prompts x models x cases, parallel execution, error capture
src/analysis.py         summaries, intervals, fallback counts, regression detection
src/history.py          SQLite run history
src/report.py           HTML report and the limitations notes
src/dataset.py          CSV, JSON and XLSX loading and column normalization
data/golden_datasets/   built-in eval suite
score_labels.py         hand-label agreement check
tests/                  72 unit tests, no API keys needed
.github/workflows/      CI runs the tests on every push
```

