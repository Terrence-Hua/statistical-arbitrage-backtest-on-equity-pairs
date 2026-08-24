"""Chart generation for backtest reports.

Produces equity curve, drawdown, z-score signal, and spread charts
and saves them to the reports/ directory.

Usage
-----
    from statarb.charts import plot_equity_curve, plot_zscore_signals

    plot_equity_curve(result, metrics, pair_label="S23/S24",
                      out_path="reports/equity_S23_S24.png")
    plot_zscore_signals(z_score, result, pair_label="S23/S24",
                        entry_z=2.0, stop_z=3.5,
                        out_path="reports/zscore_S23_S24.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend

_BLUE = "#1a73e8"
_RED = "#ea4335"
_GREEN = "#34a853"
_GREY = "#5f6368"
_LIGHT_GREY = "#dadce0"


def _save(fig: plt.Figure, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curve(
    equity: pd.Series,
    drawdown: pd.Series,
    pair_label: str = "",
    out_path: str = "reports/equity.png",
    sharpe: float | None = None,
    max_dd_pct: float | None = None,
) -> None:
    """Plot equity curve and drawdown on a two-panel figure.

    Parameters
    ----------
    equity : pd.Series
        Cumulative net P&L series.
    drawdown : pd.Series
        Rolling drawdown series (negative values).
    pair_label : str
        Chart title suffix.
    out_path : str
        File path for the output PNG.
    sharpe : float or None
        Sharpe ratio annotation.
    max_dd_pct : float or None
        Max drawdown % annotation.
    """
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("white")

    # Equity curve
    ax_eq.plot(equity.index, equity.values, color=_BLUE, linewidth=1.4, label="Net equity")
    ax_eq.axhline(0, color=_GREY, linewidth=0.6, linestyle="--")
    ax_eq.fill_between(equity.index, 0, equity.values,
                       where=(equity.values >= 0), alpha=0.08, color=_GREEN)
    ax_eq.fill_between(equity.index, 0, equity.values,
                       where=(equity.values < 0), alpha=0.08, color=_RED)
    ax_eq.set_ylabel("Cumulative P&L ($)", fontsize=9)
    ax_eq.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    title = f"Equity curve — {pair_label}" if pair_label else "Equity curve"
    annot = []
    if sharpe is not None:
        annot.append(f"Sharpe {sharpe:.2f}")
    if max_dd_pct is not None:
        annot.append(f"Max DD {max_dd_pct:.1f}%")
    if annot:
        title += f"  |  {', '.join(annot)}"
    ax_eq.set_title(title, fontsize=10)
    ax_eq.grid(axis="y", color=_LIGHT_GREY, linewidth=0.5)

    # Drawdown
    ax_dd.fill_between(drawdown.index, 0, drawdown.values, color=_RED, alpha=0.6)
    ax_dd.set_ylabel("Drawdown ($)", fontsize=9)
    ax_dd.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax_dd.grid(axis="y", color=_LIGHT_GREY, linewidth=0.5)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout(pad=1.2)
    _save(fig, out_path)


def plot_zscore_signals(
    z_score: pd.Series,
    positions: pd.Series,
    pair_label: str = "",
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    out_path: str = "reports/zscore.png",
) -> None:
    """Plot z-score time series with signal thresholds and position shading.

    Parameters
    ----------
    z_score : pd.Series
        Spread z-score series.
    positions : pd.Series
        Position direction series (+1, -1, 0).
    pair_label : str
        Chart title suffix.
    entry_z : float
        Entry z-score threshold.
    exit_z : float
        Exit z-score threshold.
    stop_z : float
        Stop-loss z-score threshold.
    out_path : str
        File path for the output PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")

    ax.plot(z_score.index, z_score.values, color=_BLUE, linewidth=0.9, label="Z-score")
    ax.axhline(0, color=_GREY, linewidth=0.6)

    for val, ls, label in [
        (entry_z, "--", f"±{entry_z} entry"),
        (-entry_z, "--", None),
        (exit_z, ":", f"±{exit_z} exit"),
        (-exit_z, ":", None),
        (stop_z, "-.", f"±{stop_z} stop"),
        (-stop_z, "-.", None),
    ]:
        ax.axhline(val, color=_GREY, linewidth=0.7, linestyle=ls,
                   label=label if label else "_nolegend_")

    # shade position periods
    common = z_score.index.intersection(positions.index)
    pos = positions.loc[common]
    z_common = z_score.loc[common]

    long_mask = pos == 1
    short_mask = pos == -1
    ax.fill_between(common, z_common.values, where=long_mask.values,
                    alpha=0.15, color=_GREEN, label="Long spread")
    ax.fill_between(common, z_common.values, where=short_mask.values,
                    alpha=0.15, color=_RED, label="Short spread")

    title = f"Z-score signals — {pair_label}" if pair_label else "Z-score signals"
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Z-score", fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", color=_LIGHT_GREY, linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout(pad=1.2)
    _save(fig, out_path)


def plot_spread(
    spread: pd.Series,
    pair_label: str = "",
    out_path: str = "reports/spread.png",
) -> None:
    """Plot the raw spread (log price difference) over time.

    Parameters
    ----------
    spread : pd.Series
        Spread series (log_a - beta * log_b).
    pair_label : str
        Chart title suffix.
    out_path : str
        File path for the output PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("white")

    ax.plot(spread.index, spread.values, color=_BLUE, linewidth=0.9)
    roll_mean = spread.rolling(60).mean()
    ax.plot(roll_mean.index, roll_mean.values, color=_RED, linewidth=1.1,
            linestyle="--", label="60d rolling mean")
    ax.set_title(f"Spread — {pair_label}" if pair_label else "Spread", fontsize=10)
    ax.set_ylabel("log(A) - β·log(B)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", color=_LIGHT_GREY, linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout(pad=1.2)
    _save(fig, out_path)
