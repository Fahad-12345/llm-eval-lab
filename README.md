# LLM Eval Lab

Compare prompts and models on a golden dataset. Every answer is scored by deterministic checks and an
LLM judge, results are saved across runs, and regressions are flagged automatically.

**Free to run.** The default path uses Groq's free tier. Demo mode needs no key at all.

> Screenshot: add one of the Dashboard tab here before posting.

## What it does

| Tab | What you get |
|---|---|
| Run | Built-in 12-case suite or your own CSV/JSON/XLSX. Edit prompt variants in a table (`{input}` placeholder). |
| Dashboard | Pass rate per model and prompt, quality-vs-latency chart, latency distribution, summary table. |
| Failures & regressions | Failing cases with the judge's reason, plus cases that passed last run and fail now. |
| History | Every run is saved to SQLite. Reload any past run. |
| Export | CSV, JSON, and a self-contained HTML report. |

## Scoring

Each answer gets two kinds of checks:

- **Deterministic (free, reproducible):** exact match after normalization, containment of the expected
  answer as a phrase (so "The answer is 20 days." matches "20 days"), word-overlap F1 as partial credit,
  and a groundedness proxy (share of content words found in the context).
- **LLM judge (optional):** a second model returns strict JSON with `correct` and `grounded` verdicts and a
  short reason. When the judge is on, its verdicts take precedence over the deterministic ones.

A case **passes** when every check that applies to it passes. Cases with no reference answer and no context
are marked unknown, not passed.

### Known limitations

- The groundedness proxy is lexical. It can miss a claim that uses the context's words to say something false.
  The judge is stronger, but it is also a model.
- The judge can be wrong. Before you quote judge-based numbers, label 20 or so answers by hand and check
  how often the judge agrees with you. Report that agreement next to the results.
- Groq's free tier is rate-limited. Keep parallel requests at 2–4. The app retries rate-limit errors with
  backoff, but very large runs will be slow.
- Reference prices in `src/config.py` are approximate and for comparison only. Free-tier calls cost $0.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your Groq key
streamlit run app.py
```

No key yet? Tick **Demo mode** in the sidebar. Demo outputs are simulated, so they are labeled as demo
everywhere and should not be quoted as real results.

Run the tests:

```bash
pytest -q
```

## Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub. `.gitignore` already excludes secrets.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and choose **Create app**.
3. Set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, paste:

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

## Get a free Groq key

Sign in at [console.groq.com](https://console.groq.com), create an API key, and paste it into the sidebar
or `secrets.toml`. Check [console.groq.com/docs/models](https://console.groq.com/docs/models) for the
current model IDs, since they change over time. Update `MODELS` in `src/config.py` if one is retired.

## Bring your own dataset

CSV, JSON, or XLSX with a `prompt` column. Optional: `expected`, `context`, `id`.
Aliases work: `question`, `input`, `query` for prompt; `answer`, `gold`, `reference`, `ground_truth` for expected;
`ctx`, `source` for context. JSON can be a list of objects or `{"examples": [...]}`.

## Project layout

```
app.py                 Streamlit UI
src/config.py          model catalog, secrets lookup
src/prompts.py         prompt templates and the shared message format
src/providers.py       Groq, OpenAI, Anthropic adapters, retries, demo routing
src/demo.py            offline simulator for demo mode
src/metrics.py         deterministic scoring
src/judge.py           LLM-as-judge prompt and strict JSON parsing
src/scoring.py         combines deterministic and judge verdicts into pass/fail
src/runner.py          prompts x models x cases grid, parallel execution
src/analysis.py        summaries, latency percentiles, regression detection
src/history.py         SQLite run history
src/report.py          self-contained HTML report
src/dataset.py         CSV/JSON/XLSX loading and column normalization
data/golden_datasets/  built-in eval suite
tests/                 36 unit tests, no API keys needed
```

## Tests and CI

The tests use fake providers, so they run without keys. GitHub Actions runs them on every push
(`.github/workflows/ci.yml`).

## License

MIT (add a LICENSE file before publishing).
