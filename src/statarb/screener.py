"""Cointegration screener for equity pairs.

Ranks all pairs in a universe by the Engle-Granger cointegration test and
returns the top-N by significance.  Each pair record includes OLS hedge ratio,
EG test statistic, p-value, ADF statistic on the spread, and Johansen trace
statistic.

Usage
-----
    from statarb.screener import screen_pairs

    results = screen_pairs(prices, max_pairs=20, pvalue_cutoff=0.05)
    # returns list[PairResult] sorted ascending by eg_pvalue
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class PairResult:
    """Holds screening statistics for a single pair (asset_a, asset_b).

    Attributes
    ----------
    ticker_a, ticker_b : str
        Ticker labels from the price DataFrame.
    hedge_ratio : float
        OLS beta of regressing log(price_a) on log(price_b).
    eg_stat : float
        Engle-Granger test statistic (ADF on the OLS residual).
    eg_pvalue : float
        MacKinnon approximate p-value for the EG test.
    adf_stat : float
        ADF t-statistic on the spread (with optimal lag selection).
    adf_pvalue : float
        ADF p-value.
    joh_trace : float
        Johansen trace statistic for r=0 (at most 0 cointegrating vectors).
    spread : pd.Series
        Spread series: log(price_a) - hedge_ratio * log(price_b).
    """

    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    eg_stat: float
    eg_pvalue: float
    adf_stat: float
    adf_pvalue: float
    joh_trace: float
    spread: pd.Series = field(repr=False)


def _compute_spread(log_a: pd.Series, log_b: pd.Series) -> tuple[float, pd.Series]:
    """OLS regression of log_a on log_b; return (beta, residual spread)."""
    x = np.column_stack([np.ones(len(log_b)), log_b.values])
    res = OLS(log_a.values, x).fit()
    beta = res.params[1]
    spread = log_a - beta * log_b
    return beta, spread


def _johansen_trace_r0(series_a: pd.Series, series_b: pd.Series) -> float:
    """Johansen trace statistic for the null of r=0 (no cointegration).

    Uses the Johansen VAR procedure via statsmodels.  Returns the trace
    statistic; larger values provide more evidence against the null.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    data = np.column_stack([series_a.values, series_b.values])
    try:
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        # trace statistic for first (r=0) hypothesis
        return float(result.lr1[0])
    except Exception:
        return np.nan


def screen_pairs(
    prices: pd.DataFrame,
    max_pairs: int = 20,
    pvalue_cutoff: float = 0.10,
    min_obs: int = 252,
) -> list[PairResult]:
    """Screen all pairs for cointegration and return top results.

    For each unique pair (i < j) the function:
    1. Runs the Engle-Granger coint() test on log prices.
    2. Computes the OLS hedge ratio and spread.
    3. Runs ADF on the spread with automatic lag selection.
    4. Computes the Johansen trace statistic.

    Pairs with eg_pvalue > pvalue_cutoff are discarded.  The remaining
    pairs are sorted ascending by eg_pvalue and the top max_pairs returned.

    Parameters
    ----------
    prices : pd.DataFrame
        Panel of close prices, shape (n_days, n_assets).
    max_pairs : int
        Maximum number of pairs to return.
    pvalue_cutoff : float
        Engle-Granger p-value threshold; pairs above this are discarded.
    min_obs : int
        Minimum number of non-NaN observations required per asset.

    Returns
    -------
    list[PairResult]
        Sorted ascending by eg_pvalue.
    """
    log_prices = np.log(prices)
    tickers = list(log_prices.columns)
    n = len(tickers)

    results: list[PairResult] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = tickers[i], tickers[j]
            la = log_prices[a].dropna()
            lb = log_prices[b].dropna()
            # align on common dates
            common = la.index.intersection(lb.index)
            if len(common) < min_obs:
                continue
            la, lb = la.loc[common], lb.loc[common]

            # ---- Engle-Granger ----------------------------------------
            try:
                eg_stat, eg_pval, _ = coint(la.values, lb.values, trend="c")
            except Exception:
                continue

            if eg_pval > pvalue_cutoff:
                continue

            # ---- OLS spread -------------------------------------------
            beta, spread = _compute_spread(la, lb)

            # ---- ADF on spread ----------------------------------------
            try:
                adf_out = adfuller(spread.values, autolag="AIC")
                adf_stat, adf_pval = adf_out[0], adf_out[1]
            except Exception:
                adf_stat, adf_pval = np.nan, np.nan

            # ---- Johansen trace ---------------------------------------
            joh_trace = _johansen_trace_r0(la, lb)

            results.append(
                PairResult(
                    ticker_a=a,
                    ticker_b=b,
                    hedge_ratio=beta,
                    eg_stat=eg_stat,
                    eg_pvalue=eg_pval,
                    adf_stat=adf_stat,
                    adf_pvalue=adf_pval,
                    joh_trace=joh_trace,
                    spread=spread,
                )
            )

    results.sort(key=lambda r: r.eg_pvalue)
    return results[:max_pairs]
