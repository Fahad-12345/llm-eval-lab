"""LLM Eval Lab: compare prompts and models on a golden dataset, judged and tracked over time.

Free by default: Groq's free tier, plus a demo mode that needs no key at all.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import regressions, summarize
from src.config import (
    BUILTIN_DATASET, DEMO_MODELS, HISTORY_DB, JUDGE_MODEL, MODELS, PROVIDER_ENV, get_secret,
)
from src.dataset import load_bytes, load_file
from src.history import list_runs, load_run, previous_run_id, save_run
from src.prompts import DEFAULT_TEMPLATES
from src.report import build_html, limitations_for
from src.runner import Plan, run_plan

st.set_page_config(page_title="LLM Eval Lab", page_icon="🧪", layout="wide")

PROVIDER_LABEL = {"groq": "Groq (free tier)", "openai": "OpenAI (paid)", "anthropic": "Anthropic (paid)"}


# ---------------------------------------------------------------- sidebar
def sidebar() -> dict:
    st.sidebar.markdown("## Setup")
    demo = st.sidebar.toggle("Demo mode (no API calls)", value=False,
                             help="Simulated outputs so you can explore the dashboard. Not real model results.")

    keys: dict[str, str] = {}
    st.sidebar.markdown("### API keys")
    # st.sidebar.caption("Keys come from `.streamlit/secrets.toml` or environment variables when set. "
    #                    "They are never written to disk by this app.")
    for provider, env in PROVIDER_ENV.items():
        preset = get_secret(env)
        if preset:
            keys[provider] = preset
            st.sidebar.success(f"{PROVIDER_LABEL[provider]}: key loaded", icon="🔑")
        else:
            typed = st.sidebar.text_input(f"{PROVIDER_LABEL[provider]} key", type="password", key=f"key_{provider}")
            keys[provider] = typed.strip()

    st.sidebar.markdown("### Models")
    catalog = DEMO_MODELS if demo else MODELS
    labels = [m.label for m in catalog]
    chosen_labels = st.sidebar.multiselect("Compare", labels, default=labels[:2] if demo else labels[:1])
    chosen = [m for m in catalog if m.label in chosen_labels]
    missing = sorted({m.provider for m in chosen if m.provider not in ("demo",) and not keys.get(m.provider)})
    if missing:
        st.sidebar.warning("Add a key for: " + ", ".join(PROVIDER_LABEL[p] for p in missing))

    st.sidebar.markdown("### Judge")
    use_judge = st.sidebar.toggle("LLM-as-judge scoring", value=not demo,
                                  help="A second model grades correctness and grounding. Uses one extra call per answer.")
    if demo:
        use_judge = False
        st.sidebar.caption("Judge is off in demo mode.")
    elif use_judge:
        st.sidebar.caption(f"Judge model: {JUDGE_MODEL.label}")

    st.sidebar.markdown("### Run settings")
    workers = st.sidebar.slider("Parallel requests", 1, 8, 1,
                                help="Groq's free tier has tight per-minute limits. "
                                     "Keep this at 1 when the judge is on; raise it only without the judge.")
    save = st.sidebar.toggle("Save runs to history", value=True)
    return {"demo": demo, "keys": keys, "models": chosen, "use_judge": use_judge,
            "workers": workers, "save": save}


# ---------------------------------------------------------------- run tab
def run_tab(cfg: dict) -> None:
    st.subheader("1 · Dataset")
    source = st.radio("Source", ["Built-in eval suite (12 cases)", "Upload my own"], horizontal=True)
    cases = None
    dataset_name = "eval_suite"
    if source.startswith("Built-in"):
        cases = load_file(BUILTIN_DATASET)
    else:
        upload = st.file_uploader("CSV, JSON or XLSX with a prompt column (optional: expected, context, id)",
                                  type=["csv", "json", "xlsx"])
        if upload is not None:
            try:
                cases = load_bytes(upload.name, upload.getvalue())
                dataset_name = upload.name
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read the file: {exc}")
    if cases is None:
        st.info("Upload a dataset to continue.")
        return
    st.dataframe(cases, hide_index=True, use_container_width=True, height=220)
    limit = st.slider("Cases to run", 1, len(cases), len(cases))
    cases = cases.head(limit)

    st.subheader("2 · Prompt variants")
    st.caption("Each row is one prompt template. Use {input} where the question goes.")
    templates_df = st.data_editor(
        pd.DataFrame({"name": list(DEFAULT_TEMPLATES), "template": list(DEFAULT_TEMPLATES.values())}),
        num_rows="dynamic", hide_index=True, use_container_width=True, key="templates",
    )
    templates = {
        str(r["name"]).strip(): str(r["template"])
        for _, r in templates_df.iterrows()
        if str(r.get("name", "")).strip() and "{input}" in str(r.get("template", ""))
    }

    st.subheader("3 · Run")
    if not cfg["models"]:
        st.warning("Choose at least one model in the sidebar.")
        return
    if not templates:
        st.warning("Add at least one prompt template that contains {input}.")
        return

    plan = Plan(cases=cases, templates=templates, models=cfg["models"], keys=cfg["keys"],
                use_judge=cfg["use_judge"], judge_key=cfg["keys"].get(JUDGE_MODEL.provider, ""),
                max_workers=cfg["workers"])
    st.caption(f"This run makes **{plan.call_count()}** model calls "
               f"({len(cases)} cases × {len(templates)} prompts × {len(cfg['models'])} models"
               f"{' × 2 with the judge' if cfg['use_judge'] else ''}).")
    if cfg["demo"]:
        st.info("DEMO MODE: outputs are simulated, not produced by real models.")

    if st.button("Run evaluation", type="primary", disabled=missing_keys(cfg)):
        bar = st.progress(0.0, text="Calling models…")
        df = run_plan(plan, progress=lambda f: bar.progress(min(f, 1.0), text=f"Scored {int(f * 100)}%"))
        bar.empty()
        run_name = f"{dataset_name} · {len(cases)} cases"
        run_id = None
        if cfg["save"]:
            run_id = save_run(HISTORY_DB, run_name, dataset_name, df, demo=cfg["demo"])
        st.session_state.current = {"df": df, "dataset": dataset_name, "name": run_name,
                                    "run_id": run_id, "demo": cfg["demo"]}
        st.success(f"Done. {len(df)} results" + (f" saved as run #{run_id}." if run_id else "."))

    if st.session_state.get("current") is not None:
        st.dataframe(st.session_state.current["df"][["case_id", "label", "prompt_id", "output",
                                                    "expected", "passed", "latency_ms", "error"]],
                     hide_index=True, use_container_width=True)


def missing_keys(cfg: dict) -> bool:
    return any(m.provider not in ("demo",) and not cfg["keys"].get(m.provider) for m in cfg["models"])


# ---------------------------------------------------------------- dashboard tab
def dashboard_tab() -> None:
    cur = st.session_state.get("current")
    if cur is None:
        st.info("Run an evaluation to populate the dashboard.")
        return
    df = cur["df"]
    summary = summarize(df)
    if cur["demo"]:
        st.warning("DEMO MODE: these numbers come from simulated models. Do not quote them as real results.")
    with st.expander("How to read these numbers", expanded=True):
        for note in limitations_for(df, JUDGE_MODEL.model):
            st.markdown(f"- {note}")

    scored = df[df["passed"].notna()]
    overall = scored["passed"].astype(bool).mean() if len(scored) else None
    best = summary.sort_values("pass_rate", ascending=False).iloc[0] if summary["pass_rate"].notna().any() else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pass rate (all)", f"{overall:.0%}" if overall is not None else "n/a")
    c2.metric("Model calls", len(df))
    c3.metric("Errors", int(df["error"].fillna("").ne("").sum()))
    c4.metric("Best model", best["model"] if best is not None else "n/a")

    st.markdown("#### Quality vs speed")
    ok = summary.dropna(subset=["pass_rate", "latency_p50_ms"])
    if not ok.empty:
        fig = px.scatter(ok, x="latency_p50_ms", y="pass_rate", color="model", symbol="prompt_id",
                         size="calls", hover_data=["latency_p95_ms", "errors"],
                         labels={"latency_p50_ms": "Median latency (ms)", "pass_rate": "Pass rate"},
                         title="Upper-left is best: high pass rate, low latency")
        fig.update_yaxes(range=[-0.05, 1.05])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if summary["pass_rate"].notna().any():
            fig = px.bar(summary, x="model", y="pass_rate", color="prompt_id", barmode="group",
                         title="Pass rate by model and prompt", labels={"pass_rate": "Pass rate"})
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(df.dropna(subset=["latency_ms"]), x="label", y="latency_ms", points="all",
                     title="Latency distribution (ms)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Summary by model and prompt")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.caption("paid_equiv_usd = what the same tokens would cost on the listed paid price. "
               "Free-tier calls are billed at $0 (cost_usd).")


# ---------------------------------------------------------------- failures and regressions
def failures_tab() -> None:
    cur = st.session_state.get("current")
    if cur is None:
        st.info("Run an evaluation first.")
        return
    df = cur["df"]
    failed = df[(df["passed"] == False) | df["error"].fillna("").ne("")]  # noqa: E712
    st.markdown(f"#### Failing cases · {len(failed)}")
    if failed.empty:
        st.success("No failing cases in this run.")
    else:
        st.dataframe(failed[["case_id", "label", "prompt_id", "question", "expected", "output",
                             "judge_reason", "error"]], hide_index=True, use_container_width=True)

    st.markdown("#### Regressions against the previous run")
    if cur.get("run_id") is None:
        st.info("Turn on 'Save runs to history' to compare against earlier runs.")
        return
    prev_id = previous_run_id(HISTORY_DB, cur["dataset"], cur["run_id"])
    if prev_id is None:
        st.info("No earlier run on this dataset yet. Run again later to see regressions.")
        return
    prev = load_run(HISTORY_DB, prev_id)
    broke = regressions(df, prev)
    st.caption(f"Compared with run #{prev_id}.")
    if broke.empty:
        st.success("No regressions: every case that passed before still passes.")
    else:
        st.error(f"{len(broke)} case(s) that passed before now fail or error.")
        st.dataframe(broke, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- history and export
def history_tab() -> None:
    runs = list_runs(HISTORY_DB)
    if runs.empty:
        st.info("No saved runs yet.")
        return
    st.dataframe(runs, hide_index=True, use_container_width=True)
    run_id = st.selectbox("Load a saved run into the dashboard", runs["id"].tolist(),
                          format_func=lambda i: f"#{i} · {runs.set_index('id').loc[i, 'name']}")
    if st.button("Load run"):
        row = runs.set_index("id").loc[run_id]
        st.session_state.current = {"df": load_run(HISTORY_DB, run_id), "dataset": row["dataset"],
                                    "name": row["name"], "run_id": int(run_id), "demo": bool(row["demo"])}
        st.success(f"Loaded run #{run_id}. Open the Dashboard tab.")


def export_tab() -> None:
    cur = st.session_state.get("current")
    if cur is None:
        st.info("Run an evaluation first.")
        return
    df = cur["df"]
    summary = summarize(df)
    st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="eval_results.csv", mime="text/csv")
    st.download_button("Download results (JSON)", df.to_json(orient="records", indent=2).encode("utf-8"),
                       file_name="eval_results.json", mime="application/json")
    st.download_button("Download HTML report",
                       build_html(f"LLM Eval Lab · {cur['name']}", summary, df,
                                  judge_model=JUDGE_MODEL.model).encode("utf-8"),
                       file_name="eval_report.html", mime="text/html")
    st.caption("The HTML report is self-contained and opens in any browser.")


# ---------------------------------------------------------------- main
def main() -> None:
    st.title("LLM Eval Lab")
    st.caption("Compare prompts and models on a golden dataset. Scored by deterministic checks and an "
               "LLM judge, tracked across runs, and free to run on Groq's free tier.")
    cfg = sidebar()
    tabs = st.tabs(["Run", "Dashboard", "Failures & regressions", "History", "Export"])
    with tabs[0]:
        run_tab(cfg)
    with tabs[1]:
        dashboard_tab()
    with tabs[2]:
        failures_tab()
    with tabs[3]:
        history_tab()
    with tabs[4]:
        export_tab()


if __name__ == "__main__":
    main()
