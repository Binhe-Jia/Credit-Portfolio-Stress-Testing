# Merton Credit Portfolio Analyzer App

This folder contains the Streamlit app for the Merton distance-to-default credit risk project.

## Run

From this folder:

```powershell
python -m streamlit run app.py
```

On Windows, you can also double-click:

```text
run_app.bat
```

## Portfolio Input

The app accepts a CSV upload or editable table. The only required field is a ticker-like column:

```csv
ticker
AAPL
JPM
XOM
```

Recommended fields:

```csv
ticker,sector_group,ead,lgd
AAPL,Technology,50000000,0.45
JPM,Financials,80000000,0.50
XOM,Energy,60000000,0.42
```

Supported aliases include `symbol`, `sector`, `exposure`, and `loss_given_default`.

## Outputs

After each run, the app displays:

- executive summary
- firm-level Merton PDs and distance-to-default
- portfolio EL, VaR, and expected shortfall
- sector tail attribution
- macro stress scenarios
- PD-floor sensitivity
- downloadable ZIP with CSV outputs and an executive-summary Markdown file

The latest outputs are also written to `analysis_outputs/`.
