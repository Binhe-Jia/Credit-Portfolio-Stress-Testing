# Methodology

This project estimates credit risk using a Merton structural model and a correlated default portfolio simulation.

## Firm-Level Merton Model

The model treats equity as a call option on firm assets. Default occurs when asset value falls below a default barrier at the risk horizon.

The app estimates:

- market capitalization as observed equity value
- equity volatility from daily equity returns
- default barrier using a KMV-style approximation when debt components are available
- asset value and asset volatility through iterative Merton calibration
- distance-to-default
- physical and risk-neutral default probabilities

## Portfolio Simulation

The portfolio model simulates correlated defaults using:

- one broad market factor
- sector factors
- idiosyncratic borrower shocks
- borrower-level PD, EAD, and LGD

Loss is calculated as:

```text
loss = default_indicator * EAD * LGD
```

The app reports expected loss, unexpected loss, VaR, expected shortfall, zero-loss path rate, concentration diagnostics, sector tail attribution, and stress results.

## Stress Testing

Stress scenarios change PD, LGD, and common-factor dependence. This gives a more informative view of downside portfolio behavior than baseline Merton PDs alone, especially for high-quality public-company portfolios.

