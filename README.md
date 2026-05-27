# Wholesale Credit Portfolio Stress Testing using a Multi-Factor Merton Framework

## Overview

This project builds an end-to-end **wholesale credit portfolio stress-testing framework** using real public-company market data and a structural credit risk model.

The core idea is to use publicly observable equity market information to estimate firm-level credit risk, then embed those firm-level default probabilities into a portfolio simulation framework that captures correlated defaults, concentration risk, and tail losses.

The project is designed to be closer to a **credit risk management / model development** exercise than a standard machine learning prediction project.

## Project Motivation

Banks and credit risk teams do not only care about whether a single borrower may default. They need to understand how a portfolio of corporate borrowers behaves under normal and stressed conditions.

This project asks:

> How can firm-level structural credit risk estimates be translated into portfolio-level credit risk metrics such as Expected Loss, Credit VaR, Expected Shortfall, and economic capital?

The project combines:

- firm-level Merton distance-to-default modeling
- market and sector factor calibration
- correlated default simulation
- EAD/LGD-based loss modeling
- portfolio loss distribution analysis
- tail-risk attribution by sector
- equal-exposure and concentrated-exposure portfolio scenarios

## Methodology

### 1. Public Companies as Corporate Borrower Proxies

The portfolio is constructed using a diversified set of publicly traded companies across sectors such as Technology, Energy, Industrials, Consumer, Healthcare, Financials, and Real Estate.

Each company is treated as a hypothetical wholesale borrower. The project does **not** model an equity investment portfolio. Public equity data is used only as an observable proxy for estimating firm-level credit risk inputs.

### 2. Firm-Level Merton Model Inputs

For each company, the project collects:

| Input | Proxy Used | Purpose |
|---|---|---|
| Equity value | Market capitalization | Observable market value of equity |
| Default barrier | Total debt | Proxy for debt threshold |
| Equity volatility | Annualized stock-return volatility | Market-based risk input |
| Sector | Manually assigned sector group | Portfolio grouping |
| Sector factor | Sector ETF return | Sector shock proxy |

The simplified Merton approximation is:

```text
V ≈ E + K
```

where:

- `V` = firm asset value
- `E` = equity value
- `K` = default barrier

Asset volatility is approximated as:

```text
sigma_V ≈ sigma_E × E / (E + K)
```

Distance to default is calculated as:

```text
DD = [ln(V/K) + (mu - 0.5 × sigma_V^2)T] / [sigma_V × sqrt(T)]
```

The one-year default probability is then:

```text
PD = Phi(-DD)
```

Distance to Default is conceptually closer to a physical-measure version of the Black-Scholes `d2`, not `d1`, because it is directly linked to the probability that asset value falls below the default barrier.

### 3. Market and Sector Factor Calibration

For each firm, daily stock returns are regressed on:

```text
r_i = alpha_i + b_i × r_SPY + g_i × r_sector + e_i
```

where:

- `r_i` = firm return
- `r_SPY` = broad market return
- `r_sector` = sector ETF return
- `b_i` = market sensitivity
- `g_i` = sector sensitivity

The regression coefficients are then rescaled into latent Gaussian factor loadings used in the default simulation.

### 4. Multi-Factor Correlated Default Simulation

The latent asset return for firm `i` is modeled as:

```text
Y_i = beta_i × M + gamma_i × I_sector(i) + sqrt(1 - beta_i^2 - gamma_i^2) × epsilon_i
```

where:

- `M` = global market factor
- `I_sector(i)` = sector factor for firm `i`
- `epsilon_i` = idiosyncratic firm-specific shock
- `beta_i` = global market loading
- `gamma_i` = sector loading

Firm `i` defaults if:

```text
Y_i < Phi^-1(PD_i)
```

This allows defaults to cluster when firms share market or sector risk exposure.

### 5. Portfolio Loss Modeling

For each borrower, credit loss is defined as:

```text
Loss_i = EAD_i × LGD_i × default_indicator_i
```

where:

- `EAD_i` = exposure at default
- `LGD_i` = loss given default
- `default_indicator_i` = 1 if borrower defaults, otherwise 0

The total portfolio loss is:

```text
L = sum(Loss_i)
```

The notebook evaluates two exposure designs:

1. **Equal-EAD baseline portfolio**  
   Each borrower receives the same exposure. This reduces mechanical concentration effects.

2. **Concentrated-EAD portfolio**  
   Exposures follow a capped lognormal distribution. This illustrates concentration risk in wholesale lending.

## Risk Metrics

The simulation produces a portfolio loss distribution and calculates:

| Metric | Interpretation |
|---|---|
| Expected Loss | Average simulated credit loss |
| Unexpected Loss | Standard deviation of portfolio loss |
| VaR 95% | Loss threshold exceeded in 5% of scenarios |
| VaR 99% | Loss threshold exceeded in 1% of scenarios |
| VaR 99.9% | Extreme tail loss threshold |
| Expected Shortfall | Average loss beyond VaR |
| Economic Capital | VaR 99.9% minus Expected Loss |

Economic capital is calculated as:

```text
Economic Capital = VaR_99.9% - Expected Loss
```

## Tail-Risk Attribution

The project also attributes 99.9% tail losses by sector.

This helps answer:

> Which sectors contribute most to extreme portfolio losses?

A sector can dominate tail losses because of high EAD, high LGD, high PD, high market/sector factor exposure, correlated defaults, or single-name concentration.

## Repository Structure

```text
.
├── wholesale_credit_portfolio_full_notebook_more_explained.ipynb
├── README.md
├── data/
│   ├── firm_level_merton_and_portfolio_inputs.csv
│   ├── portfolio_loss_simulations_equal_ead.csv
│   ├── portfolio_loss_simulations_concentrated_ead.csv
│   ├── portfolio_risk_metrics_equal_ead.csv
│   ├── portfolio_risk_metrics_concentrated_ead.csv
│   └── sector_tail_risk_attribution_equal_ead.csv
├── figures/
│   ├── portfolio_loss_distribution.png
│   ├── sector_tail_loss_attribution.png
│   └── ead_by_sector.png
└── requirements.txt
```

The CSV and figure files are generated by the notebook.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` should include:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
yfinance
jupyter
```

### 3. Run the notebook

Open:

```text
wholesale_credit_portfolio_full_notebook_more_explained.ipynb
```

and run all cells from top to bottom.

The notebook downloads public market data using `yfinance`, constructs the firm-level dataset, estimates Merton PDs, simulates correlated defaults, and saves output files.

## Key Assumptions

This project is a simplified academic / portfolio modeling exercise. Major assumptions include:

1. **Public companies proxy wholesale borrowers**  
   The firms are real public companies, but the loan exposures are hypothetical.

2. **Total debt is used as the default barrier**  
   A more refined KMV-style approach would use short-term debt plus half of long-term debt.

3. **Simplified Merton calibration**  
   Asset value and asset volatility are approximated rather than solved from the full nonlinear option-pricing equations.

4. **PD floor is applied**  
   A minimum PD is used to make finite-sample portfolio simulation more informative.

5. **EAD and LGD are assumed**  
   The project does not use actual bank loan exposures or realized loss severities.

6. **Gaussian factor structure**  
   Defaults are simulated using a Gaussian latent-variable framework, which may understate tail dependence compared with heavier-tailed models.

## Limitations

The model should not be interpreted as a production-grade credit model. Important limitations include:

- firm asset values are not directly observed
- total debt is a simplified default barrier
- market-implied structural PDs may differ from historical or rating-agency PDs
- public companies are generally lower-risk than many private wholesale borrowers
- the portfolio is small relative to real bank credit portfolios
- EAD and LGD are hypothetical
- Gaussian dependence may not fully capture crisis-period default clustering

These limitations are useful from a model governance perspective and can be discussed in a final model development report.

## Possible Extensions

### 1. Full Merton Calibration

Solve the nonlinear Merton equations:

```text
E = V × N(d1) - K × exp(-rT) × N(d2)
sigma_E = (V / E) × N(d1) × sigma_V
```

to infer `V` and `sigma_V` more rigorously.

### 2. KMV-Style Default Point

Replace total debt with:

```text
K = short-term debt + 0.5 × long-term debt
```

### 3. Stress Testing

Add scenarios such as:

- global recession
- sector shock
- correlation stress
- LGD stress
- downgrade wave
- PD multiplier stress

### 4. PCA Factor Model

Use PCA on the firm return covariance matrix to estimate empirical systematic factors.

### 5. Heavier-Tailed Dependence

Replace the Gaussian factor model with a t-copula or other heavy-tailed dependence structure.


### 6. Note
This project is still WIP, and open to further update and modification

