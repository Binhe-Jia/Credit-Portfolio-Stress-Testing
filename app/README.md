# Merton Distance-to-Default Credit Risk Project

This repository contains a research notebook and a Streamlit app for estimating firm-level Merton distance-to-default, simulating correlated portfolio defaults, running stress scenarios, and generating an executive summary.

## Project Structure

```text
merton-credit-risk/
  README.md
  requirements.txt
  .gitignore

  notebooks/
    wholesale_credit_portfolio.ipynb

  app/
    app.py
    credit_engine.py
    sample_portfolio.csv
    stress_test_portfolio_25.csv
    high_pd_small_company_stress_portfolio_35.csv
    run_app.bat

  docs/
    methodology.md
```

The notebook is the research and methodology version. The app is the user-facing version for uploading a portfolio and generating the complete analysis.

## Main Features

- Flexible portfolio input with a minimum of one company
- Merton distance-to-default and probability of default estimation
- KMV-style default barrier using short-term debt plus half of long-term debt when available
- Iterative asset value and asset volatility calibration
- Physical PD and risk-neutral PD comparison
- Correlated default simulation using market and sector factors
- Equal-EAD and concentrated-EAD portfolio views
- Expected loss, VaR, expected shortfall, and zero-loss path diagnostics
- Sector tail-loss attribution
- PD-floor sensitivity
- Macro stress scenarios with higher PD, LGD, and correlation assumptions
- Downloadable executive summary and CSV outputs

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

From the repository root:

```bash
cd app
streamlit run app.py
```

On Windows, you can also double-click:

```text
app/run_app.bat
```

The app lets users upload a CSV, edit a portfolio table, adjust model settings, run the complete analysis, and download a ZIP file containing an executive summary and output tables. It also includes a preflight market-data check that can drop unavailable tickers and continue instead of stopping the full analysis.

## Run the Notebook

Open:

```text
notebooks/wholesale_credit_portfolio.ipynb
```

The notebook contains the full research workflow, formulas, diagnostics, and export steps. It is useful for understanding the methodology and modifying model assumptions.

## Portfolio Input Format

The only required field is a ticker-like column:

```csv
ticker
AAPL
JPM
XOM
```

Recommended input:

```csv
ticker,sector_group,ead,lgd
AAPL,Technology,50000000,0.45
JPM,Financials,80000000,0.50
XOM,Energy,60000000,0.42
```

Supported aliases include:

- `symbol` for `ticker`
- `sector` for `sector_group`
- `exposure` for `ead`
- `loss_given_default` for `lgd`

If sector is missing, the app assigns `User Portfolio` and uses `SPY` as the common factor. If EAD or LGD is missing, the app applies fallback assumptions.

## Key Model Settings

- **PD floor, basis points:** minimum PD used in the default simulation. A 1 bp floor equals `0.0001`, or `0.01%`.
- **Fallback EAD, $mn:** exposure used when a portfolio row does not provide EAD.
- **Risk-free rate:** rate used in the Merton equity-as-call-option calibration.
- **Asset drift:** expected asset return used for the physical default probability.
- **Monte Carlo simulations:** number of simulated one-year portfolio loss scenarios.

## Interpreting VaR and Expected Shortfall

For small or high-quality portfolios, many simulated paths may have no defaults. In that case, VaR can be zero even when expected shortfall is positive.

Example: if 99.98% of simulated paths have zero loss, then 95%, 99%, and 99.9% VaR may all equal zero. Expected shortfall can still be positive because it averages the worst tail paths, where rare default losses occur.

## Interpreting PD Floors

The app reports both raw Merton PD and the model PD used in simulation. If many names show the same PD, they may be hitting the PD floor:

```text
pd_model = max(raw_merton_pd, pd_floor)
```

This is common for large public companies with high distance-to-default. Use the PD-floor sensitivity table to see how results change under different floor assumptions.

## Outputs

The app and notebook can generate:

- executive summary
- firm-level Merton inputs
- equal-EAD portfolio risk metrics
- concentrated-EAD portfolio risk metrics
- sector tail-risk attribution
- macro stress comparison
- PD-floor sensitivity
- top PD names
- top expected-loss names

Generated app outputs are written to `app/analysis_outputs/`, which is ignored by Git.

## Limitations

This project is for research and prototyping. It relies on public equity data and Yahoo Finance fundamentals, which should be validated before production or credit-decision use. The model does not replace internal ratings, private financials, covenant information, collateral analysis, or expert credit review.
