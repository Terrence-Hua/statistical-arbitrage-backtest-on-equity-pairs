"""Performance metrics for backtest results.

Computes Sharpe ratio, maximum drawdown, hit rate, and other
standard statistics from a BacktestResult object.

Usage
-----
    from statarb.metrics import compute_metrics, MetricsResult

    m = compute_metrics(result, trading_days=252)
    print(f"Sharpe: {m.sharpe:.2f}, Max DD: {m.max_drawdown_pct:.1f}%")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.backtest import BacktestResult, TradeRecord


@dataclass
class MetricsResult:
    """Aggregate performance statistics.

    Attributes
    ----------
    sharpe : float
        Annualised Sharpe ratio on net daily P&L.  Risk-free rate assumed 0.
    sortino : float
        Annualised Sortino ratio (downside deviation denominator).
    max_drawdown : float
        Maximum drawdown in dollars from peak equity.
    max_drawdown_pct : float
        Maximum drawdown as a percentage of peak equity.
    drawdown_series : pd.Series
        Rolling drawdown in dollars throughout the backtest.
    total_return : float
        Total net P&L in dollars.
    total_return_pct : float
        Total net P&L as a percentage of initial notional.
    n_trades : int
        Total number of completed round-trip trades.
    hit_rate : float
        Fraction of trades with positive net P&L.
    avg_trade_pnl : float
        Average net P&L per trade.
    profit_factor : float
        Gross profit / gross loss across all trades.
    costs_total : float
        Total transaction costs paid.
    annual_vol : float
        Annualised daily P&L standard deviation.
    """

    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_pct: float
    drawdown_series: pd.Series
    total_return: float
    total_return_pct: float
    n_trades: int
    hit_rate: float
    avg_trade_pnl: float
    profit_factor: float
    costs_total: float
    annual_vol: float


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute rolling dollar drawdown from peak equity."""
    peak = equity.cummax()
    return equity - peak


def _hit_rate(trades: list[TradeRecord]) -> float:
    """Fraction of trades with net P&L > 0."""
    if not trades:
        return float("nan")
    wins = sum(1 for t in trades if t.pnl_net > 0)
    return wins / len(trades)


def _profit_factor(trades: list[TradeRecord]) -> float:
    """Ratio of total gross profit to total gross loss."""
    gross_profit = sum(t.pnl_net for t in trades if t.pnl_net > 0)
    gross_loss = sum(-t.pnl_net for t in trades if t.pnl_net < 0)
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def compute_metrics(
    result: BacktestResult,
    notional: float = 100_000.0,
    trading_days: int = 252,
) -> MetricsResult:
    """Compute performance metrics from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Output of run_backtest().
    notional : float
        Initial dollar notional (for percentage return calculation).
    trading_days : int
        Number of trading days per year (default 252).

    Returns
    -------
    MetricsResult
    """
    pnl = result.daily_pnl
    eq = result.equity

    # Annualised Sharpe (risk-free rate = 0)
    mean_daily = pnl.mean()
    std_daily = pnl.std(ddof=1)
    sharpe = (mean_daily / std_daily * np.sqrt(trading_days)) if std_daily > 0 else 0.0

    # Sortino
    downside = pnl[pnl < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else std_daily
    sortino = (mean_daily / downside_std * np.sqrt(trading_days)) if downside_std > 0 else 0.0

    # Drawdown
    dd_series = _drawdown_series(eq)
    max_dd = dd_series.min()  # most negative value
    peak_at_max_dd = eq.cummax().loc[dd_series.idxmin()] if len(dd_series) > 0 else notional
    max_dd_pct = (max_dd / peak_at_max_dd * 100.0) if peak_at_max_dd != 0 else 0.0

    # Returns
    total_return = eq.iloc[-1] if len(eq) > 0 else 0.0
    total_return_pct = total_return / notional * 100.0

    # Trade stats
    trades = result.trades
    n_trades = len(trades)
    hit_rate = _hit_rate(trades)
    avg_pnl = np.mean([t.pnl_net for t in trades]) if trades else 0.0
    pf = _profit_factor(trades)

    annual_vol = std_daily * np.sqrt(trading_days)

    return MetricsResult(
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        drawdown_series=dd_series,
        total_return=total_return,
        total_return_pct=total_return_pct,
        n_trades=n_trades,
        hit_rate=hit_rate,
        avg_trade_pnl=avg_pnl,
        profit_factor=pf,
        costs_total=result.costs_total,
        annual_vol=annual_vol,
    )


def print_metrics(m: MetricsResult, label: str = "") -> None:
    """Print a formatted metrics summary to stdout.

    Parameters
    ----------
    m : MetricsResult
        Metrics to display.
    label : str
        Optional label (e.g. pair ticker) printed as header.
    """
    if label:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

    print(f"  Sharpe (annualised, after costs): {m.sharpe:>8.3f}")
    print(f"  Sortino:                          {m.sortino:>8.3f}")
    print(f"  Max drawdown:                     {m.max_drawdown:>10.2f}  ({m.max_drawdown_pct:.1f}%)")
    print(f"  Total return:                     {m.total_return:>10.2f}  ({m.total_return_pct:.1f}%)")
    print(f"  Annual vol:                       {m.annual_vol:>10.2f}")
    print(f"  Trades:                           {m.n_trades:>8d}")
    print(f"  Hit rate:                         {m.hit_rate:>8.1%}")
    print(f"  Avg trade P&L:                    {m.avg_trade_pnl:>10.2f}")
    print(f"  Profit factor:                    {m.profit_factor:>8.3f}")
    print(f"  Total costs:                      {m.costs_total:>10.2f}")
