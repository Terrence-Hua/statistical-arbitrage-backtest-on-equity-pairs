"""Synthetic price data generator.

Produces a panel of equity price series with embedded cointegration structure.
Prices are driven by latent common factors plus an OU-process idiosyncratic
component so that a subset of pairs are cointegrated by construction.

Usage
-----
    from statarb.data import generate_prices

    prices = generate_prices(n_assets=30, n_days=504, seed=42)
    # returns pd.DataFrame, shape (n_days, n_assets), columns = ticker labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _simulate_ou(
    n: int,
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Euler-Maruyama discretisation of dx = theta*(mu-x)dt + sigma*dW."""
    dt = 1.0  # daily
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * rng.standard_normal()
    return x


def generate_prices(
    n_assets: int = 40,
    n_days: int = 756,  # ~3 years
    n_factors: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily close prices for *n_assets* tickers.

    Architecture
    ------------
    Each asset i has log-return:

        r_i(t) = B_i @ f(t) + e_i(t)

    where f(t) are common factor returns (random walk innovations) and
    e_i(t) is an OU-process idiosyncratic component.  Assets that share
    the same factor loadings differ only in their OU noise, so their
    spread is a stationary OU process — i.e. they are cointegrated.

    Parameters
    ----------
    n_assets : int
        Number of synthetic tickers to generate.
    n_days : int
        Number of trading days in the panel.
    n_factors : int
        Number of latent common factors.
    seed : int
        RNG seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape (n_days, n_assets).  Index is a DatetimeIndex of business
        days starting 2020-01-02.  Columns are ticker labels ``S00``…``SNN``.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Latent factor returns  (random walk)                             #
    # ------------------------------------------------------------------ #
    factor_vols = rng.uniform(0.005, 0.012, size=n_factors)
    factor_returns = rng.normal(0, 1, size=(n_days, n_factors)) * factor_vols

    # ------------------------------------------------------------------ #
    # 2. Factor loadings — assets grouped into n_factors clusters         #
    # ------------------------------------------------------------------ #
    # Within each cluster, assets share roughly the same loadings,
    # making intra-cluster pairs cointegrated.
    cluster_size = max(1, n_assets // n_factors)
    loadings = np.zeros((n_assets, n_factors))
    for i in range(n_assets):
        cluster = min(i // cluster_size, n_factors - 1)
        base = np.zeros(n_factors)
        base[cluster] = 1.0
        # small cross-loading noise
        noise = rng.normal(0, 0.05, size=n_factors)
        loadings[i] = base + noise

    # ------------------------------------------------------------------ #
    # 3. OU idiosyncratic components                                      #
    # ------------------------------------------------------------------ #
    ou_thetas = rng.uniform(0.02, 0.15, size=n_assets)
    ou_sigmas = rng.uniform(0.005, 0.015, size=n_assets)
    ou_x0 = rng.uniform(-0.02, 0.02, size=n_assets)

    idio = np.column_stack(
        [
            _simulate_ou(n_days, ou_thetas[i], 0.0, ou_sigmas[i], ou_x0[i], rng)
            for i in range(n_assets)
        ]
    )

    # ------------------------------------------------------------------ #
    # 4. Assemble log-returns and cumulate to price                       #
    # ------------------------------------------------------------------ #
    common_returns = factor_returns @ loadings.T  # (n_days, n_assets)
    total_returns = common_returns + idio  # (n_days, n_assets)

    # Drift: small positive expected return (~8 % annually)
    drift = 0.0003
    total_returns = total_returns + drift

    # Start all assets at price 100
    log_prices = np.cumsum(total_returns, axis=0)
    prices = 100.0 * np.exp(log_prices)

    # ------------------------------------------------------------------ #
    # 5. Pack into DataFrame                                              #
    # ------------------------------------------------------------------ #
    tickers = [f"S{i:02d}" for i in range(n_assets)]
    dates = pd.bdate_range(start="2020-01-02", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def load_prices(path: str) -> pd.DataFrame:
    """Load a CSV of daily close prices.

    Expected format: first column is date, remaining columns are tickers.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Shape (n_days, n_assets) with DatetimeIndex.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.astype(float)
