"""Vectorised pairs backtest engine.

Simulates a dollar-neutral long/short position in a cointegrated pair driven
by z-score signals.  Returns daily P&L, position sizes, and trade log.

Signal rules
------------
- **Enter long spread** (long A, short B): z < -entry_z
- **Enter short spread** (short A, long B): z > +entry_z
- **Exit**: z crosses back through ±exit_z toward zero  (default exit_z = 0)
- **Stop-loss**: position is closed if z exceeds stop_z in the adverse
  direction (default stop_z = 3.0)

Usage
-----
    from statarb.backtest import run_backtest, BacktestResult

    result = run_backtest(
        prices_a=prices["S23"],
        prices_b=prices["S24"],
        hedge_ratio=0.22,
        z_score=z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        notional=100_000,
    )
    print(result.sharpe)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """Single round-trip trade.

    Attributes
    ----------
    entry_date, exit_date : pd.Timestamp
        Open and close dates.
    direction : int
        +1 = long spread (long A / short B), -1 = short spread.
    entry_z, exit_z_val : float
        Z-score at entry and exit.
    pnl_gross : float
        Gross P&L in dollars before costs.
    pnl_net : float
        Net P&L after transaction costs.
    exit_reason : str
        One of "signal", "stop_loss", or "end_of_data".
    """

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int
    entry_z: float
    exit_z_val: float
    pnl_gross: float
    pnl_net: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Full backtest output for a single pair.

    Attributes
    ----------
    equity : pd.Series
        Cumulative net P&L series (daily, dollars).
    daily_pnl : pd.Series
        Daily net P&L.
    positions : pd.Series
        Position direction (+1, -1, 0) on each day.
    trades : list[TradeRecord]
        All completed round-trip trades.
    costs_total : float
        Total transaction costs paid over the backtest.
    """

    equity: pd.Series
    daily_pnl: pd.Series
    positions: pd.Series
    trades: list[TradeRecord] = field(default_factory=list)
    costs_total: float = 0.0


def _compute_fill_cost(
    price_a: float,
    price_b: float,
    hedge_ratio: float,
    notional: float,
    commission_bps: float,
    slippage_bps: float,
) -> float:
    """Compute one-way transaction cost for entering/exiting a spread position.

    Each leg pays commission_bps + slippage_bps (half-spread) on the notional
    traded.  The hedge leg notional scales by hedge_ratio.

    Parameters
    ----------
    price_a, price_b : float
        Current prices of asset A and B.
    hedge_ratio : float
        OLS beta (shares of B per share of A).
    notional : float
        Dollar notional allocated to the A leg.
    commission_bps : float
        Commission in basis points (e.g. 5 = 0.05 %).
    slippage_bps : float
        One-way slippage in basis points.

    Returns
    -------
    float
        Total one-way cost in dollars.
    """
    total_bps = (commission_bps + slippage_bps) / 10_000.0
    cost_a = notional * total_bps
    notional_b = notional * abs(hedge_ratio) * price_b / price_a
    cost_b = notional_b * total_bps
    return cost_a + cost_b


def run_backtest(
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: float,
    z_score: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    notional: float = 100_000.0,
    commission_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> BacktestResult:
    """Run a vectorised backtest on a single pair.

    The engine iterates day-by-day maintaining position state and recording
    daily mark-to-market P&L.  Fills occur at the next day's open (here
    approximated as next day's close, conservative).

    Parameters
    ----------
    prices_a, prices_b : pd.Series
        Daily close prices for each leg, aligned on the same DatetimeIndex.
    hedge_ratio : float
        Shares of B per unit of A (OLS beta).
    z_score : pd.Series
        Pre-computed spread z-score on the same index.
    entry_z : float
        |z| threshold to open a position.
    exit_z : float
        |z| threshold (toward zero) to close a position.
    stop_z : float
        |z| threshold (away from zero) to cut a position (stop-loss).
    notional : float
        Dollar notional for the A leg.
    commission_bps : float
        Commission per leg, in basis points.
    slippage_bps : float
        One-way slippage per leg, in basis points.

    Returns
    -------
    BacktestResult
    """
    # Align all series
    idx = prices_a.index.intersection(prices_b.index).intersection(z_score.index)
    pa = prices_a.loc[idx].values
    pb = prices_b.loc[idx].values
    z = z_score.loc[idx].values
    dates = idx

    n = len(dates)
    daily_pnl = np.zeros(n)
    positions = np.zeros(n, dtype=int)

    trades: list[TradeRecord] = []
    total_cost = 0.0

    # Position state
    pos = 0          # current direction: +1, -1, or 0
    shares_a = 0.0   # shares of A held (signed)
    shares_b = 0.0   # shares of B held (signed)
    entry_date = None
    entry_z_val = 0.0

    prev_pa = pa[0]
    prev_pb = pb[0]

    for t in range(1, n):
        cur_z = z[t]
        cur_pa = pa[t]
        cur_pb = pb[t]
        date = dates[t]

        # Mark-to-market on existing position (use today's close)
        if pos != 0:
            pnl_a = shares_a * (cur_pa - prev_pa)
            pnl_b = shares_b * (cur_pb - prev_pb)
            daily_pnl[t] = pnl_a + pnl_b
        else:
            daily_pnl[t] = 0.0

        # ----------------------------------------------------------------
        # Exit logic (checked before entry for same-bar exit)
        # ----------------------------------------------------------------
        exit_triggered = False
        exit_reason = ""

        if pos == 1:  # long spread: close when z >= -exit_z (crosses back)
            if cur_z >= -exit_z:
                exit_triggered = True
                exit_reason = "signal"
            elif cur_z <= -stop_z:  # spread widened further against us
                exit_triggered = True
                exit_reason = "stop_loss"
        elif pos == -1:  # short spread: close when z <= +exit_z
            if cur_z <= exit_z:
                exit_triggered = True
                exit_reason = "signal"
            elif cur_z >= stop_z:
                exit_triggered = True
                exit_reason = "stop_loss"

        if exit_triggered and pos != 0:
            cost = _compute_fill_cost(cur_pa, cur_pb, hedge_ratio, notional,
                                      commission_bps, slippage_bps)
            total_cost += cost
            # deduct closing cost from today's pnl
            daily_pnl[t] -= cost

            # record trade
            trade_pnl_net = daily_pnl[entry_idx:t + 1].sum()  # noqa: F821
            gross = trade_pnl_net + cost  # add back closing cost for gross
            trades.append(TradeRecord(
                entry_date=entry_date,
                exit_date=date,
                direction=pos,
                entry_z=entry_z_val,
                exit_z_val=cur_z,
                pnl_gross=gross,
                pnl_net=trade_pnl_net,
                exit_reason=exit_reason,
            ))
            pos = 0
            shares_a = 0.0
            shares_b = 0.0

        # ----------------------------------------------------------------
        # Entry logic
        # ----------------------------------------------------------------
        if pos == 0:
            if not np.isnan(cur_z):
                if cur_z < -entry_z:
                    # long spread: long A, short B
                    pos = 1
                    shares_a = notional / cur_pa
                    shares_b = -(hedge_ratio * notional / cur_pa) / (cur_pb / cur_pa) if cur_pb != 0 else 0.0
                    # simpler: dollar-neutral on B leg
                    shares_b = -(hedge_ratio * notional) / cur_pb if cur_pb != 0 else 0.0
                    cost = _compute_fill_cost(cur_pa, cur_pb, hedge_ratio, notional,
                                              commission_bps, slippage_bps)
                    total_cost += cost
                    daily_pnl[t] -= cost
                    entry_date = date
                    entry_idx = t  # noqa: F841
                    entry_z_val = cur_z

                elif cur_z > entry_z:
                    # short spread: short A, long B
                    pos = -1
                    shares_a = -notional / cur_pa
                    shares_b = (hedge_ratio * notional) / cur_pb if cur_pb != 0 else 0.0
                    cost = _compute_fill_cost(cur_pa, cur_pb, hedge_ratio, notional,
                                              commission_bps, slippage_bps)
                    total_cost += cost
                    daily_pnl[t] -= cost
                    entry_date = date
                    entry_idx = t  # noqa: F841
                    entry_z_val = cur_z

        positions[t] = pos
        prev_pa = cur_pa
        prev_pb = cur_pb

    # Close any open position at end of data
    if pos != 0:
        cost = _compute_fill_cost(pa[-1], pb[-1], hedge_ratio, notional,
                                  commission_bps, slippage_bps)
        total_cost += cost
        daily_pnl[-1] -= cost
        trade_pnl_net = daily_pnl[entry_idx:-1].sum()
        trades.append(TradeRecord(
            entry_date=entry_date,
            exit_date=dates[-1],
            direction=pos,
            entry_z=entry_z_val,
            exit_z_val=z[-1],
            pnl_gross=trade_pnl_net + cost,
            pnl_net=trade_pnl_net,
            exit_reason="end_of_data",
        ))

    equity = pd.Series(daily_pnl, index=dates).cumsum()
    return BacktestResult(
        equity=equity,
        daily_pnl=pd.Series(daily_pnl, index=dates),
        positions=pd.Series(positions, index=dates),
        trades=trades,
        costs_total=total_cost,
    )
