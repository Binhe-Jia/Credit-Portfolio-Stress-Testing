from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

from credit_engine import AnalysisConfig, SAMPLE_PORTFOLIO, analyze_portfolio, executive_summary_markdown


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Merton Credit Portfolio Analyzer",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    [data-testid="stMetric"] {
        border: 1px solid #d7dee8;
        border-radius: 6px;
        padding: 0.7rem 0.8rem;
        background: #fbfcfe;
    }
    .small-note {
        color: #52616f;
        font-size: 0.88rem;
        margin-top: -0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


def make_download_zip(results: dict) -> bytes:
    csv_outputs = {
        "executive_summary.csv": results["executive_summary"],
        "firm_level_merton_and_portfolio_inputs.csv": results["firm_level"],
        "portfolio_risk_metrics_equal_ead.csv": results["risk_metrics"],
        "portfolio_risk_metrics_concentrated_ead.csv": results["risk_metrics_concentrated"],
        "sector_tail_risk_attribution_equal_ead.csv": results["tail_attribution"],
        "macro_stress_comparison.csv": results["macro_stress"],
        "pd_floor_sensitivity.csv": results["pd_floor_sensitivity"],
        "top_pd_names.csv": results["top_pd"],
        "top_expected_loss_names.csv": results["top_expected_loss"],
    }
    mem = BytesIO()
    with ZipFile(mem, "w", ZIP_DEFLATED) as zf:
        zf.writestr("executive_summary.md", executive_summary_markdown(results))
        for name, df in csv_outputs.items():
            zf.writestr(name, df.to_csv(index=False))
    return mem.getvalue()


def save_outputs_to_disk(results: dict) -> Path:
    run_dir = OUTPUT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "executive_summary.csv": results["executive_summary"],
        "firm_level_merton_and_portfolio_inputs.csv": results["firm_level"],
        "portfolio_risk_metrics_equal_ead.csv": results["risk_metrics"],
        "portfolio_risk_metrics_concentrated_ead.csv": results["risk_metrics_concentrated"],
        "sector_tail_risk_attribution_equal_ead.csv": results["tail_attribution"],
        "macro_stress_comparison.csv": results["macro_stress"],
        "pd_floor_sensitivity.csv": results["pd_floor_sensitivity"],
        "top_pd_names.csv": results["top_pd"],
        "top_expected_loss_names.csv": results["top_expected_loss"],
    }
    for filename, df in tables.items():
        df.to_csv(run_dir / filename, index=False)
    (run_dir / "executive_summary.md").write_text(executive_summary_markdown(results), encoding="utf-8")
    return run_dir


def fmt_money_mn(value: float) -> str:
    return f"${value:,.2f} mn"


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


st.title("Merton Credit Portfolio Analyzer")
st.caption("Structural distance-to-default, correlated default simulation, stress testing, and executive summary generation.")

with st.sidebar:
    st.header("Portfolio")
    input_mode = st.radio("Input method", ["Use editable table", "Upload CSV"], index=0)

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
        portfolio_df = read_uploaded_csv(uploaded)
        if portfolio_df is None:
            portfolio_df = SAMPLE_PORTFOLIO.copy()
            st.info("Upload a CSV or use the sample shown in the table.")
    else:
        portfolio_df = SAMPLE_PORTFOLIO.copy()

    st.header("Model Settings")
    start_date = st.date_input("Price start date", pd.Timestamp("2019-01-01"))
    end_date = st.date_input("Price end date", pd.Timestamp("2024-12-31"))
    n_sims = st.slider("Monte Carlo simulations", 10_000, 300_000, 50_000, step=10_000)
    pd_floor_bp = st.number_input("PD floor, basis points", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
    baseline_ead_mn = st.number_input("Fallback EAD, $mn", min_value=1.0, max_value=5_000.0, value=100.0, step=10.0)
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
    asset_drift = st.number_input("Asset drift", min_value=-0.20, max_value=0.30, value=0.05, step=0.005, format="%.3f")

st.subheader("Portfolio Input")
st.markdown(
    '<div class="small-note">Required: ticker or symbol. Optional: sector_group/sector, sector_etf, ead/exposure, lgd/loss_given_default.</div>',
    unsafe_allow_html=True,
)

if input_mode == "Use editable table":
    edited_portfolio = st.data_editor(
        portfolio_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )
else:
    edited_portfolio = portfolio_df
    st.dataframe(edited_portfolio, use_container_width=True, hide_index=True)

run = st.button("Run Complete Analysis", type="primary", use_container_width=True)

if run:
    config = AnalysisConfig(
        start_date=str(start_date),
        end_date=str(end_date),
        n_sims=int(n_sims),
        pd_floor=float(pd_floor_bp) / 10_000,
        baseline_ead=float(baseline_ead_mn) * 1e6,
        risk_free_rate=float(risk_free_rate),
        asset_drift=float(asset_drift),
    )

    with st.spinner("Downloading market data, calibrating Merton inputs, and running simulations..."):
        try:
            results = analyze_portfolio(edited_portfolio, config)
            save_path = save_outputs_to_disk(results)
            st.session_state["results"] = results
            st.session_state["save_path"] = save_path
        except Exception as exc:
            st.error(str(exc))
            st.stop()

if "results" not in st.session_state:
    st.info("Add or upload a portfolio, then run the analysis.")
    st.stop()

results = st.session_state["results"]
summary = results["executive_summary"]

summary_lookup = dict(zip(summary["item"], summary["value"]))
metric_cols = st.columns(5)
metric_cols[0].metric("Borrowers", f"{int(summary_lookup['Borrowers analyzed'])}")
metric_cols[1].metric("Mean PD", fmt_pct(summary_lookup["Mean model PD"]))
metric_cols[2].metric("Expected Loss", fmt_money_mn(summary_lookup["Expected loss, mn"]))
metric_cols[3].metric("VaR 99.9%", fmt_money_mn(summary_lookup["VaR 99.9%, mn"]))
metric_cols[4].metric("ES 99.9%", fmt_money_mn(summary_lookup["Expected shortfall 99.9%, mn"]))

tab_summary, tab_firms, tab_portfolio, tab_stress, tab_downloads, tab_method = st.tabs(
    ["Executive Summary", "Firm Risk", "Portfolio Risk", "Stress Testing", "Downloads", "Method"]
)

with tab_summary:
    st.markdown(executive_summary_markdown(results))
    st.dataframe(summary, use_container_width=True, hide_index=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top PD Names")
        st.dataframe(
            results["top_pd"][
                ["ticker", "sector_group", "distance_to_default", "pd_model", "merton_pd_risk_neutral", "debt_to_equity", "lgd"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with col_b:
        st.subheader("Top Expected Loss Names")
        st.dataframe(
            results["top_expected_loss"][
                ["ticker", "sector_group", "pd_model", "ead_equal", "lgd", "expected_loss_equal_ead"]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tab_firms:
    st.subheader("Firm-Level Merton Inputs")
    firm_cols = [
        "ticker",
        "sector_group",
        "equity_value",
        "default_barrier",
        "equity_vol",
        "asset_vol",
        "distance_to_default",
        "merton_pd",
        "pd_model",
        "lgd",
        "ead_equal",
    ]
    st.dataframe(results["firm_level"][firm_cols], use_container_width=True, hide_index=True)
    chart_df = results["firm_level"].sort_values("pd_model", ascending=False).head(20).set_index("ticker")
    st.bar_chart(chart_df["pd_model"])

with tab_portfolio:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Equal-EAD Risk Metrics")
        st.dataframe(results["risk_metrics"], use_container_width=True, hide_index=True)
    with col_b:
        st.subheader("Concentrated-EAD Risk Metrics")
        st.dataframe(results["risk_metrics_concentrated"], use_container_width=True, hide_index=True)
    st.subheader("Sector Tail Attribution")
    st.dataframe(results["tail_attribution"], use_container_width=True, hide_index=True)
    if not results["tail_attribution"].empty:
        st.bar_chart(results["tail_attribution"].set_index("sector_group")["tail_loss_share"])

with tab_stress:
    st.subheader("Macro Stress Comparison")
    st.dataframe(results["macro_stress"], use_container_width=True, hide_index=True)
    st.bar_chart(results["macro_stress"].set_index("scenario")[["expected_loss_mn", "var_999_mn", "es_999_mn"]])
    st.subheader("PD Floor Sensitivity")
    st.dataframe(results["pd_floor_sensitivity"], use_container_width=True, hide_index=True)
    st.line_chart(results["pd_floor_sensitivity"].set_index("pd_floor")[["expected_loss_mn", "var_999_mn", "es_999_mn"]])

with tab_downloads:
    zip_bytes = make_download_zip(results)
    st.download_button(
        "Download complete analysis ZIP",
        data=zip_bytes,
        file_name="merton_credit_portfolio_analysis.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.success(f"Latest outputs saved to: {st.session_state['save_path']}")

with tab_method:
    st.markdown(
        """
        The app estimates market-implied credit risk using a Merton structural model.

        1. It standardizes the uploaded portfolio and fills missing sector factor, EAD, and LGD assumptions.
        2. It downloads equity prices and public-company fundamentals from Yahoo Finance.
        3. It estimates annualized equity volatility from daily returns.
        4. It builds a KMV-style default barrier using short-term debt plus half of long-term debt when available.
        5. It calibrates asset value and asset volatility with the Merton equity-pricing equations.
        6. It converts distance-to-default into physical and risk-neutral PDs.
        7. It estimates market and sector factor loadings from equity returns.
        8. It simulates correlated defaults and calculates EL, VaR, ES, concentration, tail attribution, and stress results.

        This is a research and prototyping tool. Market data, debt fields, LGD assumptions, and model calibration choices
        should be independently validated before use in formal credit decisions.
        """
    )

