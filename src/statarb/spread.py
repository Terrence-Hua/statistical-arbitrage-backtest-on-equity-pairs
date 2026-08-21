"""Ornstein-Uhlenbeck spread modelling.

Fits OU parameters (theta, mu, sigma) to a spread series using the exact
discrete-time maximum-likelihood estimator, then derives the half-life and
provides a rolling z-score signal.

Usage
-----
    from statarb.spread import fit_ou, rolling_zscore

    params = fit_ou(spread)
    z = rolling_zscore(spread, window=params.half_life_days)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OUParams:
    """Fitted Ornstein-Uhlenbeck parameters.

    Attributes
    ----------
    theta : float
        Mean-reversion speed (per day).  Higher values → faster reversion.
    mu : float
        Long-run mean of the process.
    sigma : float
        Daily diffusion coefficient (standard deviation of OU innovations).
    half_life_days : float
        Expected time in days for the spread to halve its deviation from mu.
        Computed as ln(2) / theta.
    sigma_eq : float
        Equilibrium (stationary) standard deviation sqrt(sigma^2 / (2*theta)).
    """

    theta: float
    mu: float
    sigma: float
    half_life_days: float
    sigma_eq: float


def fit_ou(spread: pd.Series) -> OUParams:
    """Fit OU parameters to *spread* via discrete-time MLE.

    The discrete OU transition is:

        x_{t+1} = x_t * exp(-theta) + mu * (1 - exp(-theta)) + eps

    where eps ~ N(0, sigma_d^2) and sigma_d^2 = sigma^2 / (2*theta) * (1 - exp(-2*theta)).

    This is equivalent to OLS regression:

        x_{t+1} = a + b * x_t + eps

    with a = mu*(1-b), b = exp(-theta), sigma_d = std(residuals).

    Parameters
    ----------
    spread : pd.Series
        The spread (or log-spread) time series.  Must have >= 30 observations.

    Returns
    -------
    OUParams
        Fitted parameters.  If theta ≤ 0 (non-stationary), raises ValueError.
    """
    x = spread.dropna().values
    if len(x) < 30:
        raise ValueError(f"Need at least 30 observations; got {len(x)}")

    x_t = x[:-1]
    x_t1 = x[1:]

    # OLS: x_{t+1} = a + b * x_t
    n = len(x_t)
    sx = x_t.sum()
    sy = x_t1.sum()
    sxx = (x_t ** 2).sum()
    sxy = (x_t * x_t1).sum()

    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12:
        raise ValueError("Degenerate spread — cannot fit OU.")

    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n

    resid = x_t1 - (a + b * x_t)
    sigma_d = resid.std(ddof=1)

    # Recover OU parameters
    b = np.clip(b, 1e-6, 1.0 - 1e-9)  # force mean-reverting
    theta = -np.log(b)
    mu = a / (1.0 - b)

    if theta <= 0:
        raise ValueError(f"Non-mean-reverting spread; theta={theta:.4f}")

    # sigma from discrete variance
    sigma_sq = sigma_d ** 2 * 2 * theta / (1.0 - np.exp(-2.0 * theta))
    sigma = np.sqrt(max(sigma_sq, 1e-12))

    half_life = np.log(2.0) / theta
    sigma_eq = sigma / np.sqrt(2.0 * theta)

    return OUParams(
        theta=theta,
        mu=mu,
        sigma=sigma,
        half_life_days=half_life,
        sigma_eq=sigma_eq,
    )


def rolling_zscore(
    spread: pd.Series,
    window: int | None = None,
    ou_params: OUParams | None = None,
) -> pd.Series:
    """Compute a rolling z-score of *spread*.

    If *ou_params* is provided, the score is normalised by the equilibrium
    standard deviation (sigma_eq) and centred on the OU long-run mean (mu).
    Otherwise a standard rolling mean/std normalisation is used with the
    given *window*.

    Parameters
    ----------
    spread : pd.Series
        Spread time series.
    window : int or None
        Rolling window in days.  Used when ou_params is None.  Defaults to
        half the series length.
    ou_params : OUParams or None
        If provided, uses equilibrium mean and std for normalisation.

    Returns
    -------
    pd.Series
        Z-score series, same index as *spread*.
    """
    if ou_params is not None:
        z = (spread - ou_params.mu) / ou_params.sigma_eq
        return z

    if window is None:
        window = max(20, len(spread) // 2)

    roll_mean = spread.rolling(window, min_periods=window // 2).mean()
    roll_std = spread.rolling(window, min_periods=window // 2).std()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = (spread - roll_mean) / roll_std.replace(0, np.nan)
    return z
