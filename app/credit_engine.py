from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


DEFAULT_SECTOR_ETFS = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer": "XLY",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Utilities": "XLU",
    "Materials": "XLB",
}


SAMPLE_PORTFOLIO = pd.DataFrame(
    {
        "ticker": ["AAPL", "JPM", "XOM", "UNH", "PLD"],
        "sector_group": ["Technology", "Financials", "Energy", "Healthcare", "Real Estate"],
        "ead": [50_000_000, 80_000_000, 60_000_000, 70_000_000, 45_000_000],
        "lgd": [0.45, 0.50, 0.42, 0.40, 0.55],
    }
)


@dataclass
class AnalysisConfig:
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    n_sims: int = 50_000
    pd_floor: float = 0.0001
    baseline_ead: float = 100_000_000
    risk_free_rate: float = 0.04
    asset_drift: float = 0.05
    trading_days: int = 252
    random_seed: int = 42
    market_data_timeout: int = 15
    drop_unavailable_tickers: bool = True


def standardize_portfolio_input(
    portfolio_input: pd.DataFrame | str | Path,
    sector_etfs: dict[str, str] | None = None,
) -> pd.DataFrame:
    sector_etfs = sector_etfs or DEFAULT_SECTOR_ETFS
    portfolio_df = pd.read_csv(portfolio_input) if isinstance(portfolio_input, (str, Path)) else pd.DataFrame(portfolio_input).copy()

    if portfolio_df.empty:
        raise ValueError("Portfolio must contain at least one company.")

    normalized = {str(col).lower().strip(): col for col in portfolio_df.columns}
    aliases = {
        "ticker": ["ticker", "symbol", "borrower_ticker", "company_ticker", "name"],
        "sector_group": ["sector_group", "sector", "industry_group", "gics_sector"],
        "sector_etf": ["sector_etf", "factor_etf", "sector_factor", "etf"],
        "ead": ["ead", "exposure", "exposure_at_default", "loan_amount", "commitment"],
        "lgd": ["lgd", "loss_given_default"],
    }

    rename_map: dict[str, str] = {}
    for standard_name, choices in aliases.items():
        for alias in choices:
            if alias in normalized:
                rename_map[normalized[alias]] = standard_name
                break
    portfolio_df = portfolio_df.rename(columns=rename_map)

    if "ticker" not in portfolio_df.columns:
        raise ValueError("Portfolio input must include a ticker or symbol column.")

    portfolio_df["ticker"] = portfolio_df["ticker"].astype(str).str.upper().str.strip()
    portfolio_df = portfolio_df.loc[portfolio_df["ticker"].ne("")].drop_duplicates("ticker")
    if portfolio_df.empty:
        raise ValueError("Portfolio must contain at least one valid ticker.")

    if "sector_group" not in portfolio_df.columns:
        portfolio_df["sector_group"] = "User Portfolio"
    portfolio_df["sector_group"] = portfolio_df["sector_group"].fillna("User Portfolio").astype(str).str.strip()
    portfolio_df.loc[portfolio_df["sector_group"].eq(""), "sector_group"] = "User Portfolio"

    if "sector_etf" not in portfolio_df.columns:
        portfolio_df["sector_etf"] = portfolio_df["sector_group"].map(sector_etfs)
    else:
        portfolio_df["sector_etf"] = portfolio_df["sector_etf"].fillna(portfolio_df["sector_group"].map(sector_etfs))
    portfolio_df["sector_etf"] = portfolio_df["sector_etf"].fillna("SPY").astype(str).str.upper().str.strip()
    portfolio_df.loc[portfolio_df["sector_etf"].eq(""), "sector_etf"] = "SPY"

    for col in ["ead", "lgd"]:
        if col in portfolio_df.columns:
            portfolio_df[col] = pd.to_numeric(portfolio_df[col], errors="coerce")
    if "lgd" in portfolio_df.columns and portfolio_df["lgd"].dropna().gt(1).any():
        portfolio_df["lgd"] = np.where(portfolio_df["lgd"] > 1, portfolio_df["lgd"] / 100, portfolio_df["lgd"])

    ordered = ["ticker", "sector_group", "sector_etf"]
    optional = [col for col in ["ead", "lgd"] if col in portfolio_df.columns]
    remaining = [col for col in portfolio_df.columns if col not in ordered + optional]
    return portfolio_df[ordered + optional + remaining].reset_index(drop=True)


def _close_prices(downloaded: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(downloaded, pd.Series):
        return downloaded.to_frame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(0):
            return downloaded["Close"]
        if "Close" in downloaded.columns.get_level_values(-1):
            return downloaded.xs("Close", axis=1, level=-1)
    if "Close" in downloaded.columns:
        return downloaded[["Close"]]
    return downloaded


def download_price_data(tickers: list[str], factor_tickers: list[str], config: AnalysisConfig) -> pd.DataFrame:
    all_tickers = sorted(set(tickers + factor_tickers))
    raw = yf.download(
        all_tickers,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
        timeout=config.market_data_timeout,
    )
    prices = _close_prices(raw)
    if len(all_tickers) == 1:
        prices.columns = all_tickers
    prices = prices.dropna(axis=1, how="all")
    if "SPY" not in prices.columns:
        raise ValueError("SPY price history is required as the broad market factor.")
    return prices


def validate_market_data(
    portfolio_df: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    tickers = portfolio_df["ticker"].tolist()
    factor_tickers = sorted(set(["SPY"] + portfolio_df["sector_etf"].tolist()))
    prices = download_price_data(tickers, factor_tickers, config)
    available_tickers = sorted(set(tickers).intersection(prices.columns))
    missing_tickers = sorted(set(tickers) - set(available_tickers))

    if missing_tickers and not config.drop_unavailable_tickers:
        raise ValueError(
            "No price history returned for: "
            + ", ".join(missing_tickers)
            + ". Remove these tickers or enable 'Drop unavailable tickers and continue'."
        )

    filtered_portfolio = portfolio_df.loc[portfolio_df["ticker"].isin(available_tickers)].copy()
    if filtered_portfolio.empty:
        raise ValueError("No portfolio tickers returned usable price history. Check ticker symbols and date range.")
    return filtered_portfolio, prices, available_tickers, missing_tickers, factor_tickers


def fetch_company_metadata(tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).get_info(timeout=10)
        except TypeError:
            try:
                info = yf.Ticker(ticker).info
            except Exception:
                info = {}
        except Exception:
            info = {}
        rows.append(
            {
                "ticker": ticker,
                "market_cap": info.get("marketCap"),
                "total_debt": info.get("totalDebt"),
                "short_term_debt": info.get("currentDebt") or info.get("shortTermDebt") or info.get("shortLongTermDebt"),
                "long_term_debt": info.get("longTermDebt"),
                "total_cash": info.get("totalCash"),
                "enterprise_value": info.get("enterpriseValue"),
                "industry": info.get("industry"),
                "sector_yfinance": info.get("sector"),
            }
        )
    return pd.DataFrame(rows)


def calibrate_merton_asset_value_and_vol(
    equity_value: float,
    equity_vol: float,
    default_barrier: float,
    risk_free_rate: float,
    horizon: float = 1.0,
) -> pd.Series:
    if pd.isna(equity_value) or pd.isna(equity_vol) or pd.isna(default_barrier) or equity_value <= 0 or equity_vol <= 0 or default_barrier <= 0:
        return pd.Series({"asset_value": np.nan, "asset_vol": np.nan, "calibration_error": np.nan})

    initial_asset_value = equity_value + default_barrier
    initial_asset_vol = np.clip(equity_vol * equity_value / initial_asset_value, 0.01, 2.0)

    def residuals(log_params: np.ndarray) -> np.ndarray:
        asset_value = np.exp(log_params[0])
        asset_vol = np.exp(log_params[1])
        d1 = (np.log(asset_value / default_barrier) + (risk_free_rate + 0.5 * asset_vol**2) * horizon) / (asset_vol * np.sqrt(horizon))
        d2 = d1 - asset_vol * np.sqrt(horizon)
        equity_model = asset_value * norm.cdf(d1) - default_barrier * np.exp(-risk_free_rate * horizon) * norm.cdf(d2)
        equity_vol_model = norm.cdf(d1) * asset_value * asset_vol / equity_value
        return np.array([(equity_model - equity_value) / equity_value, (equity_vol_model - equity_vol) / max(equity_vol, 1e-6)])

    result = least_squares(
        residuals,
        x0=np.log([initial_asset_value, initial_asset_vol]),
        bounds=(np.log([max(default_barrier * 1.0001, 1.0), 0.001]), np.log([1e16, 3.0])),
        max_nfev=1_000,
    )
    asset_value, asset_vol = np.exp(result.x)
    return pd.Series({"asset_value": asset_value, "asset_vol": asset_vol, "calibration_error": np.sqrt(np.mean(result.fun**2))})


def positive_quantile_scale(series: pd.Series, q: float = 0.95) -> float:
    positive_values = series.clip(lower=0)
    scale = positive_values.quantile(q)
    if pd.isna(scale) or scale <= 0:
        scale = positive_values.max()
    return 1.0 if pd.isna(scale) or scale <= 0 else float(scale)


def estimate_factor_loadings(merton_inputs: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in merton_inputs.iterrows():
        ticker = row["ticker"]
        sector_etf = row["sector_etf"]
        data_cols = [ticker, "SPY"] if sector_etf == "SPY" else [ticker, "SPY", sector_etf]
        data_cols = [col for col in dict.fromkeys(data_cols) if col in returns.columns]
        data = returns[data_cols].dropna().copy()
        if len(data) < 30:
            market_beta = 0.0
            sector_beta = 0.0
            sector_spy_beta = 0.0
            r_squared = 0.0
        else:
            if sector_etf == "SPY" or sector_etf not in data.columns:
                data["sector_resid"] = 0.0
                sector_spy_beta = 0.0
            else:
                sector_model = LinearRegression()
                sector_model.fit(data[["SPY"]], data[sector_etf])
                data["sector_resid"] = data[sector_etf] - sector_model.predict(data[["SPY"]])
                sector_spy_beta = float(sector_model.coef_[0])

            model = LinearRegression()
            model.fit(data[["SPY", "sector_resid"]], data[ticker])
            market_beta = float(model.coef_[0])
            sector_beta = float(model.coef_[1])
            r_squared = float(model.score(data[["SPY", "sector_resid"]], data[ticker]))

        rows.append(
            {
                "ticker": ticker,
                "market_beta_raw": market_beta,
                "sector_beta_resid_raw": sector_beta,
                "sector_spy_beta": sector_spy_beta,
                "r_squared": r_squared,
            }
        )

    out = pd.DataFrame(rows)
    market_scale = positive_quantile_scale(out["market_beta_raw"])
    sector_scale = positive_quantile_scale(out["sector_beta_resid_raw"])
    out["beta_global"] = (0.30 * out["market_beta_raw"].clip(lower=0) / market_scale).clip(0.0, 0.45)
    out["gamma_sector"] = (0.25 * out["sector_beta_resid_raw"].clip(lower=0) / sector_scale).clip(0.0, 0.35)
    common_var = out["beta_global"] ** 2 + out["gamma_sector"] ** 2
    out["idio_weight"] = np.sqrt(np.maximum(1 - common_var, 0))
    return out


def expected_shortfall_from_worst_tail(portfolio_losses: np.ndarray, alpha: float) -> float:
    losses_sorted = np.sort(np.asarray(portfolio_losses))
    n_tail = max(1, int(np.ceil((1 - alpha) * len(losses_sorted))))
    return float(losses_sorted[-n_tail:].mean())


def run_credit_portfolio_simulation(
    merton_df: pd.DataFrame,
    ead_col: str,
    n_sims: int,
    seed: int,
    store_matrices: bool = True,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_firms = len(merton_df)
    pd_values = np.clip(merton_df["pd_model"].to_numpy(float), 1e-12, 0.999999)
    default_thresholds = norm.ppf(pd_values)
    beta = merton_df["beta_global"].to_numpy(float)
    gamma = merton_df["gamma_sector"].to_numpy(float)
    idio = merton_df["idio_weight"].to_numpy(float)
    ead_lgd = (merton_df[ead_col].to_numpy(float) * merton_df["lgd"].to_numpy(float)).astype(float)
    sector_names = merton_df["sector_group"].unique()
    sector_index = {sector: idx for idx, sector in enumerate(sector_names)}
    firm_sector_idx = merton_df["sector_group"].map(sector_index).to_numpy()

    portfolio_losses = np.empty(n_sims, dtype=float)
    losses = np.empty((n_sims, n_firms), dtype=np.float32) if store_matrices else None
    defaults = np.empty((n_sims, n_firms), dtype=bool) if store_matrices else None

    for start in range(0, n_sims, chunk_size):
        end = min(start + chunk_size, n_sims)
        size = end - start
        market_factor = rng.normal(size=size)
        sector_factors = rng.normal(size=(size, len(sector_names)))
        eps = rng.normal(size=(size, n_firms))
        latent = market_factor[:, None] * beta[None, :] + sector_factors[:, firm_sector_idx] * gamma[None, :] + eps * idio[None, :]
        defaults_chunk = latent < default_thresholds[None, :]
        losses_chunk = defaults_chunk * ead_lgd[None, :]
        portfolio_losses[start:end] = losses_chunk.sum(axis=1)
        if store_matrices:
            defaults[start:end, :] = defaults_chunk
            losses[start:end, :] = losses_chunk.astype(np.float32)

    return {"portfolio_losses": portfolio_losses, "losses": losses, "defaults": defaults, "sector_names": sector_names}


def calculate_risk_metrics(portfolio_losses: np.ndarray) -> pd.DataFrame:
    values = {
        "Expected Loss": float(portfolio_losses.mean()),
        "Unexpected Loss": float(portfolio_losses.std()),
        "VaR 95%": float(np.quantile(portfolio_losses, 0.95)),
        "VaR 99%": float(np.quantile(portfolio_losses, 0.99)),
        "VaR 99.9%": float(np.quantile(portfolio_losses, 0.999)),
        "Expected Shortfall 95%": expected_shortfall_from_worst_tail(portfolio_losses, 0.95),
        "Expected Shortfall 99%": expected_shortfall_from_worst_tail(portfolio_losses, 0.99),
        "Expected Shortfall 99.9%": expected_shortfall_from_worst_tail(portfolio_losses, 0.999),
        "Zero-loss Path Rate": float((portfolio_losses == 0).mean()),
    }
    out = pd.DataFrame({"metric": list(values), "value": list(values.values())})
    out["value_mn"] = np.where(out["metric"].eq("Zero-loss Path Rate"), np.nan, out["value"] / 1e6)
    return out


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    return float(metrics.loc[metrics["metric"].eq(name), "value"].iloc[0])


def apply_credit_stress(base_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    stressed = base_df.copy()
    pd_multiplier = kwargs.get("pd_multiplier", 1.0)
    pd_addition = kwargs.get("pd_addition", 0.0)
    pd_cap = kwargs.get("pd_cap", 0.25)
    lgd_uplift = kwargs.get("lgd_uplift", 0.0)
    beta_multiplier = kwargs.get("beta_multiplier", 1.0)
    gamma_multiplier = kwargs.get("gamma_multiplier", 1.0)
    stressed_sectors = set(kwargs.get("stressed_sectors") or [])
    sector_pd_multiplier = kwargs.get("sector_pd_multiplier", 1.0)
    sector_lgd_uplift = kwargs.get("sector_lgd_uplift", 0.0)

    stressed["pd_model"] = (stressed["pd_model"] * pd_multiplier + pd_addition).clip(upper=pd_cap)
    stressed["lgd"] = (stressed["lgd"] + lgd_uplift).clip(upper=0.90)
    if stressed_sectors:
        mask = stressed["sector_group"].isin(stressed_sectors)
        stressed.loc[mask, "pd_model"] = (stressed.loc[mask, "pd_model"] * sector_pd_multiplier).clip(upper=pd_cap)
        stressed.loc[mask, "lgd"] = (stressed.loc[mask, "lgd"] + sector_lgd_uplift).clip(upper=0.90)

    stressed["beta_global"] *= beta_multiplier
    stressed["gamma_sector"] *= gamma_multiplier
    common_var = stressed["beta_global"] ** 2 + stressed["gamma_sector"] ** 2
    scale = np.where(common_var > 0.90, np.sqrt(0.90 / common_var), 1.0)
    stressed["beta_global"] *= scale
    stressed["gamma_sector"] *= scale
    stressed["idio_weight"] = np.sqrt(np.maximum(1 - stressed["beta_global"] ** 2 - stressed["gamma_sector"] ** 2, 0))
    return stressed


def analyze_portfolio(portfolio_input: pd.DataFrame | str | Path, config: AnalysisConfig) -> dict[str, Any]:
    portfolio_df = standardize_portfolio_input(portfolio_input)
    portfolio_df, prices, available_tickers, missing_price_tickers, factor_tickers = validate_market_data(portfolio_df, config)
    tickers = portfolio_df["ticker"].tolist()
    returns = prices.pct_change().dropna()
    metadata = fetch_company_metadata(tickers)
    df = portfolio_df.merge(metadata, on="ticker", how="left")
    df["equity_vol"] = df["ticker"].map(returns[tickers].std() * np.sqrt(config.trading_days))
    df["equity_value"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["default_barrier_total_debt"] = pd.to_numeric(df["total_debt"], errors="coerce")
    df["short_term_debt"] = pd.to_numeric(df["short_term_debt"], errors="coerce")
    df["long_term_debt"] = pd.to_numeric(df["long_term_debt"], errors="coerce")

    short_debt = df["short_term_debt"].fillna(0)
    long_debt = df["long_term_debt"].fillna((df["default_barrier_total_debt"] - short_debt).clip(lower=0))
    kmv_barrier = short_debt + 0.5 * long_debt
    df["default_barrier"] = np.where(kmv_barrier > 0, kmv_barrier, df["default_barrier_total_debt"])
    df["debt_to_equity"] = df["default_barrier"] / df["equity_value"]

    merton = df.dropna(subset=["equity_value", "default_barrier", "equity_vol"]).copy()
    merton = merton.loc[(merton["equity_value"] > 0) & (merton["default_barrier"] > 0) & (merton["equity_vol"] > 0)].copy()
    if merton.empty:
        raise ValueError("No borrowers have enough market cap, debt, and return data for Merton estimation.")
    excluded_after_fundamentals = sorted(set(tickers) - set(merton["ticker"]))

    calibrated = merton.apply(
        lambda row: calibrate_merton_asset_value_and_vol(
            row["equity_value"],
            row["equity_vol"],
            row["default_barrier"],
            risk_free_rate=config.risk_free_rate,
        ),
        axis=1,
    )
    merton = pd.concat([merton, calibrated], axis=1)
    merton["distance_to_default"] = (
        np.log(merton["asset_value"] / merton["default_barrier"]) + (config.asset_drift - 0.5 * merton["asset_vol"] ** 2)
    ) / merton["asset_vol"]
    merton["distance_to_default_risk_neutral"] = (
        np.log(merton["asset_value"] / merton["default_barrier"]) + (config.risk_free_rate - 0.5 * merton["asset_vol"] ** 2)
    ) / merton["asset_vol"]
    merton["merton_pd"] = norm.cdf(-merton["distance_to_default"])
    merton["merton_pd_risk_neutral"] = norm.cdf(-merton["distance_to_default_risk_neutral"])
    merton["pd_model"] = merton["merton_pd"].clip(lower=config.pd_floor)

    factor_loadings = estimate_factor_loadings(merton, returns)
    merton = merton.merge(factor_loadings, on="ticker", how="left")

    rng = np.random.default_rng(config.random_seed)
    supplied_ead = pd.to_numeric(merton["ead"], errors="coerce") if "ead" in merton else pd.Series(np.nan, index=merton.index)
    supplied_lgd = pd.to_numeric(merton["lgd"], errors="coerce") if "lgd" in merton else pd.Series(np.nan, index=merton.index)
    merton["ead_equal"] = supplied_ead.fillna(config.baseline_ead)
    raw_concentrated = rng.lognormal(mean=np.log(config.baseline_ead), sigma=0.7, size=len(merton))
    cap = min(1.0, max(0.05, 1.0 / len(merton))) * raw_concentrated.sum()
    merton["ead_concentrated"] = np.clip(raw_concentrated, None, cap)
    merton["lgd"] = supplied_lgd.fillna(pd.Series(rng.uniform(0.35, 0.60, len(merton)), index=merton.index)).clip(0.0, 1.0)
    merton["ead_mn"] = merton["ead_equal"] / 1e6
    merton["ead_weight"] = merton["ead_equal"] / merton["ead_equal"].sum()
    merton["expected_loss_equal_ead"] = merton["pd_model"] * merton["ead_equal"] * merton["lgd"]
    merton["expected_loss_concentrated_ead"] = merton["pd_model"] * merton["ead_concentrated"] * merton["lgd"]

    baseline = run_credit_portfolio_simulation(merton, "ead_equal", config.n_sims, config.random_seed, store_matrices=True)
    conc = run_credit_portfolio_simulation(merton, "ead_concentrated", config.n_sims, config.random_seed, store_matrices=False)
    risk_metrics = calculate_risk_metrics(baseline["portfolio_losses"])
    risk_metrics_conc = calculate_risk_metrics(conc["portfolio_losses"])

    losses = baseline["losses"]
    losses_df = pd.DataFrame(losses, columns=merton["ticker"]) if losses is not None else pd.DataFrame()
    tail_count = max(1, int(np.ceil(0.001 * len(baseline["portfolio_losses"]))))
    tail_idx = np.argsort(baseline["portfolio_losses"])[-tail_count:]
    sector_rows = []
    for sector in baseline["sector_names"]:
        sector_tickers = merton.loc[merton["sector_group"].eq(sector), "ticker"].tolist()
        sector_loss = losses_df[sector_tickers].sum(axis=1) if not losses_df.empty else pd.Series(dtype=float)
        sector_rows.append(
            {
                "sector_group": sector,
                "expected_loss_mn": float(sector_loss.mean() / 1e6) if len(sector_loss) else 0.0,
                "tail_loss_999_mn": float(sector_loss.iloc[tail_idx].mean() / 1e6) if len(sector_loss) else 0.0,
            }
        )
    tail_attribution = pd.DataFrame(sector_rows)
    total_tail = tail_attribution["tail_loss_999_mn"].sum()
    tail_attribution["tail_loss_share"] = np.where(total_tail > 0, tail_attribution["tail_loss_999_mn"] / total_tail, 0.0)

    stress_definitions = {
        "Baseline": {},
        "Mild recession": {"pd_multiplier": 2.0, "lgd_uplift": 0.05, "beta_multiplier": 1.10, "gamma_multiplier": 1.10},
        "Severe recession": {"pd_multiplier": 5.0, "pd_addition": 0.0005, "lgd_uplift": 0.15, "beta_multiplier": 1.35, "gamma_multiplier": 1.35, "pd_cap": 0.10},
        "Real estate and financial shock": {
            "pd_multiplier": 1.5,
            "lgd_uplift": 0.05,
            "beta_multiplier": 1.20,
            "gamma_multiplier": 1.40,
            "stressed_sectors": ["Real Estate", "Financials"],
            "sector_pd_multiplier": 4.0,
            "sector_lgd_uplift": 0.10,
            "pd_cap": 0.12,
        },
    }
    stress_rows = []
    stress_sims = min(config.n_sims, 100_000)
    for scenario, kwargs in stress_definitions.items():
        stressed = apply_credit_stress(merton, **kwargs)
        stress_losses = run_credit_portfolio_simulation(stressed, "ead_equal", stress_sims, config.random_seed, store_matrices=False)["portfolio_losses"]
        metrics = calculate_risk_metrics(stress_losses)
        stress_rows.append(
            {
                "scenario": scenario,
                "mean_pd": stressed["pd_model"].mean(),
                "mean_lgd": stressed["lgd"].mean(),
                "expected_loss_mn": metric_value(metrics, "Expected Loss") / 1e6,
                "var_99_mn": metric_value(metrics, "VaR 99%") / 1e6,
                "var_999_mn": metric_value(metrics, "VaR 99.9%") / 1e6,
                "es_999_mn": metric_value(metrics, "Expected Shortfall 99.9%") / 1e6,
                "zero_loss_rate": metric_value(metrics, "Zero-loss Path Rate"),
            }
        )
    macro_stress = pd.DataFrame(stress_rows)

    pd_floor_rows = []
    for floor in [0.0, 0.00001, 0.00005, 0.0001, 0.00025, 0.001]:
        floor_df = merton.copy()
        floor_df["pd_model"] = floor_df["merton_pd"].clip(lower=max(floor, 1e-12))
        floor_losses = run_credit_portfolio_simulation(
            floor_df,
            "ead_equal",
            min(config.n_sims, 50_000),
            config.random_seed + 1,
            store_matrices=False,
        )["portfolio_losses"]
        metrics = calculate_risk_metrics(floor_losses)
        pd_floor_rows.append(
            {
                "pd_floor": floor,
                "mean_model_pd": floor_df["pd_model"].mean(),
                "share_floored": float((floor_df["merton_pd"] < floor).mean()) if floor > 0 else 0.0,
                "expected_loss_mn": metric_value(metrics, "Expected Loss") / 1e6,
                "var_999_mn": metric_value(metrics, "VaR 99.9%") / 1e6,
                "es_999_mn": metric_value(metrics, "Expected Shortfall 99.9%") / 1e6,
                "zero_loss_rate": metric_value(metrics, "Zero-loss Path Rate"),
            }
        )

    top_pd = merton.sort_values("pd_model", ascending=False).head(10)
    top_el = merton.sort_values("expected_loss_equal_ead", ascending=False).head(10)
    executive_summary = pd.DataFrame(
        {
            "item": [
                "Borrowers analyzed",
                "Mean model PD",
                "Mean LGD",
                "Expected loss, mn",
                "VaR 99.9%, mn",
                "Expected shortfall 99.9%, mn",
                "Concentrated EAD VaR 99.9%, mn",
                "Severe recession ES 99.9%, mn",
                "Zero-loss path rate",
            ],
            "value": [
                len(merton),
                merton["pd_model"].mean(),
                merton["lgd"].mean(),
                metric_value(risk_metrics, "Expected Loss") / 1e6,
                metric_value(risk_metrics, "VaR 99.9%") / 1e6,
                metric_value(risk_metrics, "Expected Shortfall 99.9%") / 1e6,
                metric_value(risk_metrics_conc, "VaR 99.9%") / 1e6,
                macro_stress.loc[macro_stress["scenario"].eq("Severe recession"), "es_999_mn"].iloc[0],
                metric_value(risk_metrics, "Zero-loss Path Rate"),
            ],
        }
    )

    return {
        "portfolio_input": portfolio_df,
        "available_tickers": available_tickers,
        "missing_price_tickers": missing_price_tickers,
        "excluded_after_fundamentals": excluded_after_fundamentals,
        "prices": prices,
        "firm_level": merton,
        "risk_metrics": risk_metrics,
        "risk_metrics_concentrated": risk_metrics_conc,
        "tail_attribution": tail_attribution.sort_values("tail_loss_999_mn", ascending=False),
        "macro_stress": macro_stress,
        "pd_floor_sensitivity": pd.DataFrame(pd_floor_rows),
        "top_pd": top_pd,
        "top_expected_loss": top_el,
        "executive_summary": executive_summary,
        "portfolio_losses": baseline["portfolio_losses"],
        "portfolio_losses_concentrated": conc["portfolio_losses"],
    }


def executive_summary_markdown(results: dict[str, Any]) -> str:
    summary = results["executive_summary"]
    firm_level = results["firm_level"]
    top_pd = results["top_pd"].iloc[0]
    top_el = results["top_expected_loss"].iloc[0]
    risk = results["risk_metrics"]

    def item(name: str) -> float:
        return float(summary.loc[summary["item"].eq(name), "value"].iloc[0])

    return (
        "# Executive Summary\n\n"
        f"The analysis covered **{int(item('Borrowers analyzed'))} borrower(s)**. "
        f"Average model PD was **{item('Mean model PD'):.4%}** and average LGD was **{item('Mean LGD'):.1%}**.\n\n"
        f"Baseline expected loss was **USD {item('Expected loss, mn'):.2f} million**. "
        f"The 99.9% VaR was **USD {item('VaR 99.9%, mn'):.2f} million**, and 99.9% expected shortfall was "
        f"**USD {item('Expected shortfall 99.9%, mn'):.2f} million**.\n\n"
        f"The highest model-PD borrower was **{top_pd['ticker']}** with PD **{top_pd['pd_model']:.4%}**. "
        f"The largest expected-loss contributor was **{top_el['ticker']}** with expected loss "
        f"**USD {top_el['expected_loss_equal_ead'] / 1e6:.2f} million**.\n\n"
        f"The zero-loss path rate was **{metric_value(risk, 'Zero-loss Path Rate'):.1%}**, which should be interpreted carefully "
        "for small or high-quality portfolios because discrete default simulations can have a large mass at zero loss.\n\n"
        "## Main Limitations\n\n"
        "- Market data and fundamentals are pulled from Yahoo Finance and should be validated before production use.\n"
        "- Public equity volatility is a proxy for borrower asset risk.\n"
        "- The default barrier uses a KMV-style approximation when debt components are available.\n"
        "- Tail estimates depend on the number of simulations, PD floor, LGD assumptions, and factor-loading calibration.\n"
        f"- {len(firm_level)} borrower(s) passed the data filters; excluded names, if any, lacked required market data.\n"
    )
