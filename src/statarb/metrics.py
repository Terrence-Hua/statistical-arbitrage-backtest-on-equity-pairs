"""Performance metrics for backtest results.

Computes annualised Sharpe ratio (net of costs), maximum drawdown,
hit rate, CAGR, Calmar ratio, and trade-level statistics from a
BacktestResult.

Usage
-----
    from statarb.metrics import compute_metrics, PerfMetrics

    m = compute_metrics(result, trading_days_per_year=252)
    print(m.sharpe, m.max_drawdown_pct, m.hit_rate)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.backtest import BacktestResult


@dataclass
class PerfMetrics:
    """Aggregated performance metrics for a single backtest.

    Attributes
    ----------
    sharpe : float
        Annualised Sharpe ratio of net daily P&L.
    sortino : float
        Annualised Sortino ratio (downside deviation denominator).
    max_drawdown_pct : float
        Maximum peak-to-trough drawdown as a percentage of peak equity.
    cagr : float
        Compound annual growth rate as a fraction (e.g. 0.12 = 12 %).
    calmar : float
        CAGR / abs(max_drawdown_pct).  inf if drawdown is zero.
    hit_rate : float
        Fraction of completed trades with positive net P&L.
    avg_win : float
        Average net P&L of winning trades.
    avg_loss : float
        Average net P&L of losing trades (negative value).
    profit_factor : float
        Sum of wins / abs(sum of losses).  inf if no losses.
    n_trades : int
        Total number of completed round-trip trades.
    costs_total : float
        Total transaction costs paid over the backtest in dollars.
    avg_holding_days : float
        Mean trade duration in calendar days.
    """

    sharpe: float
    sortino: float
    max_drawdown_pct: float
    cagr: float
    calmar: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    costs_total: float
    avg_holding_days: float


def _annualised_sharpe(daily_pnl: pd.Series, ann_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio from a daily P&L series.

    Uses the standard sqrt(T) scaling.  Returns 0.0 if std is zero.

    Parameters
    ----------
    daily_pnl : pd.Series
        Daily net P&L in dollars.
    ann_factor : float
        Trading days per year for annualisation.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    if daily_pnl.std(ddof=1) == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(ann_factor))


def _annualised_sortino(daily_pnl: pd.Series, ann_factor: float = 252.0) -> float:
    """Annualised Sortino ratio from a daily P&L series.

    Uses downside deviation (negative returns only) in the denominator.
    Returns 0.0 if downside deviation is zero.

    Parameters
    ----------
    daily_pnl : pd.Series
        Daily net P&L in dollars.
    ann_factor : float
        Trading days per year.

    Returns
    -------
    float
        Annualised Sortino ratio.
    """
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(daily_pnl.mean() / downside.std(ddof=1) * np.sqrt(ann_factor))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a percentage.

    Parameters
    ----------
    equity : pd.Series
        Cumulative P&L series (dollars).  Can start at any level.

    Returns
    -------
    float
        Maximum drawdown in percent (e.g. -15.3).  Zero or negative.
    """
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    if roll_max.max() <= 0:
        return 0.0
    return float((drawdown / roll_max.abs().replace(0, np.nan)).min() * 100)


def _cagr(equity: pd.Series, ann_factor: float = 252.0) -> float:
    """Compound annual growth rate from a cumulative P&L series.

    Defined as (final_equity / initial_notional)^(1/years) - 1, where
    initial_notional defaults to 1 if the series starts at or below 0.

    Parameters
    ----------
    equity : pd.Series
        Cumulative dollar P&L.
    ann_factor : float
        Trading days per year.

    Returns
    -------
    float
        CAGR as a fraction.  Returns 0.0 if the series is too short.
    """
    n = len(equity)
    if n < 2:
        return 0.0
    years = n / ann_factor
    final = equity.iloc[-1]
    # For dollar P&L: growth relative to a reference base of 1 notional unit
    # We use absolute equity; if negative overall, CAGR is negative.
    base = max(abs(equity.iloc[0]) + 1e-9, 1.0)
    total_return = final / base
    if total_return <= 0:
        return float(-((abs(total_return)) ** (1.0 / years) - 1))
    return float(total_return ** (1.0 / years) - 1)


def compute_metrics(
    result: BacktestResult,
    ann_factor: float = 252.0,
) -> PerfMetrics:
    """Compute all performance metrics for a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Output from statarb.backtest.run_backtest().
    ann_factor : float
        Trading days per year (default 252).

    Returns
    -------
    PerfMetrics
        Populated metrics dataclass.
    """
    pnl = result.daily_pnl
    equity = result.equity
    trades = result.trades

    sharpe = _annualised_sharpe(pnl, ann_factor)
    sortino = _annualised_sortino(pnl, ann_factor)
    mdd = max_drawdown(equity)
    cagr = _cagr(equity, ann_factor)
    calmar = cagr / abs(mdd / 100) if mdd != 0 else float("inf")

    # Trade-level stats
    n_trades = len(trades)
    if n_trades > 0:
        net_pnls = [t.pnl_net for t in trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]
        hit_rate = len(wins) / n_trades
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        sum_wins = sum(wins)
        sum_losses = abs(sum(losses))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else float("inf")

        durations = [
            (t.exit_date - t.entry_date).days for t in trades
            if hasattr(t.exit_date, "days") is False  # pd.Timestamp subtraction
        ]
        # Recalculate properly for pd.Timestamp
        durations = []
        for t in trades:
            try:
                d = (t.exit_date - t.entry_date).days
                durations.append(d)
            except Exception:
                durations.append(0)
        avg_holding = float(np.mean(durations)) if durations else 0.0
    else:
        hit_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        avg_holding = 0.0

    return PerfMetrics(
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=mdd,
        cagr=cagr,
        calmar=calmar,
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        n_trades=n_trades,
        costs_total=result.costs_total,
        avg_holding_days=avg_holding,
    )


def metrics_to_dict(m: PerfMetrics) -> dict:
    """Serialise PerfMetrics to a plain dict with formatted values.

    Parameters
    ----------
    m : PerfMetrics

    Returns
    -------
    dict
        Human-readable string values for display.
    """
    return {
        "Sharpe (net)": f"{m.sharpe:.3f}",
        "Sortino": f"{m.sortino:.3f}",
        "Max drawdown": f"{m.max_drawdown_pct:.2f}%",
        "CAGR": f"{m.cagr:.2%}",
        "Calmar": f"{m.calmar:.2f}" if m.calmar != float("inf") else "inf",
        "Hit rate": f"{m.hit_rate:.1%}",
        "Avg win ($)": f"{m.avg_win:,.0f}",
        "Avg loss ($)": f"{m.avg_loss:,.0f}",
        "Profit factor": f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "inf",
        "Trades": str(m.n_trades),
        "Total costs ($)": f"{m.costs_total:,.0f}",
        "Avg holding (days)": f"{m.avg_holding_days:.1f}",
    }
