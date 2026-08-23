"""Equity curve and drawdown chart generation.

Produces two-panel figures showing the cumulative P&L equity curve and
the corresponding underwater drawdown series.  Figures are saved as PNG
files in the specified output directory.

Usage
-----
    from statarb.charts import plot_equity_drawdown, plot_zscore_signals

    path = plot_equity_drawdown(result, pair_label="S05/S09", out_dir="reports")
    path2 = plot_zscore_signals(z_score, positions, pair_label="S05/S09", out_dir="reports")
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI use

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from statarb.backtest import BacktestResult


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute percentage drawdown from peak at each point.

    Parameters
    ----------
    equity : pd.Series
        Cumulative P&L series.

    Returns
    -------
    pd.Series
        Drawdown in percent (0 to -100).
    """
    roll_max = equity.cummax()
    # Guard against division by zero when equity starts at 0
    base = roll_max.abs().clip(lower=1e-9)
    dd = (equity - roll_max) / base * 100
    return dd


def plot_equity_drawdown(
    result: BacktestResult,
    pair_label: str = "",
    out_dir: str = "reports",
    filename: str | None = None,
    notional: float = 100_000.0,
) -> str:
    """Plot equity curve and drawdown in a two-panel figure.

    Parameters
    ----------
    result : BacktestResult
        Output from statarb.backtest.run_backtest().
    pair_label : str
        Pair identifier shown in the chart title (e.g. "S05/S09").
    out_dir : str
        Directory where the PNG is saved.  Created if missing.
    filename : str or None
        Output filename.  Defaults to ``equity_{pair_label}.png``.
    notional : float
        Reference notional for the title annotation.

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    equity = result.equity
    dd = _drawdown_series(equity)

    label_safe = pair_label.replace("/", "_")
    if filename is None:
        filename = f"equity_{label_safe}.png" if label_safe else "equity_curve.png"
    out_path = os.path.join(out_dir, filename)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#f8f9fa")

    # — Equity curve —
    ax1.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.5)
    ax1.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax1.fill_between(equity.index, equity.values, 0,
                     where=(equity.values >= 0), alpha=0.15, color="#1f77b4")
    ax1.fill_between(equity.index, equity.values, 0,
                     where=(equity.values < 0), alpha=0.15, color="#d62728")
    ax1.set_ylabel("Cumulative P&L ($)", fontsize=11)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    title = f"Equity curve — {pair_label}" if pair_label else "Equity curve"
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    # Annotate final P&L
    final_pnl = equity.iloc[-1]
    ax1.annotate(
        f"Final: ${final_pnl:,.0f}",
        xy=(equity.index[-1], final_pnl),
        xytext=(-80, 15 if final_pnl >= 0 else -25),
        textcoords="offset points",
        fontsize=9,
        color="#1f77b4" if final_pnl >= 0 else "#d62728",
    )

    # — Drawdown —
    ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.6)
    ax2.plot(dd.index, dd.values, color="#d62728", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f8f9fa")

    max_dd = dd.min()
    ax2.axhline(max_dd, color="#888888", linewidth=0.8, linestyle=":")
    ax2.annotate(
        f"Max DD: {max_dd:.1f}%",
        xy=(dd.idxmin(), max_dd),
        xytext=(10, -15),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def plot_zscore_signals(
    z_score: pd.Series,
    positions: pd.Series,
    pair_label: str = "",
    out_dir: str = "reports",
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    filename: str | None = None,
) -> str:
    """Plot z-score with entry/exit thresholds and position shading.

    Parameters
    ----------
    z_score : pd.Series
        Spread z-score time series.
    positions : pd.Series
        Position direction (+1, 0, -1) on the same index.
    pair_label : str
        Pair identifier for the title.
    out_dir : str
        Output directory.
    entry_z : float
        Entry z-score threshold lines drawn at ±entry_z.
    exit_z : float
        Exit z-score threshold lines drawn at ±exit_z.
    filename : str or None
        Output filename.  Defaults to ``zscore_{pair_label}.png``.

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    label_safe = pair_label.replace("/", "_")
    if filename is None:
        filename = f"zscore_{label_safe}.png" if label_safe else "zscore.png"
    out_path = os.path.join(out_dir, filename)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Shade long/short regions
    long_mask = positions > 0
    short_mask = positions < 0
    ax.fill_between(z_score.index, z_score, where=long_mask,
                    alpha=0.25, color="#2ca02c", label="Long spread")
    ax.fill_between(z_score.index, z_score, where=short_mask,
                    alpha=0.25, color="#d62728", label="Short spread")

    ax.plot(z_score.index, z_score.values, color="#1f77b4", linewidth=1.0)

    # Threshold lines
    for level, style, color in [
        (entry_z, "--", "#d62728"),
        (-entry_z, "--", "#2ca02c"),
        (exit_z, ":", "#888888"),
        (-exit_z, ":", "#888888"),
        (0, "-", "#333333"),
    ]:
        ax.axhline(level, linestyle=style, color=color, linewidth=0.9, alpha=0.8)

    title = f"Spread z-score — {pair_label}" if pair_label else "Spread z-score"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Z-score", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


def plot_pair_selection_table(
    stats_df: pd.DataFrame,
    out_dir: str = "reports",
    filename: str = "pair_selection.png",
) -> str:
    """Render the pair selection statistics table as a PNG.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of statarb.stats.pair_stats_table().
    out_dir : str
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    display = stats_df.copy()
    for col in display.select_dtypes(include=bool).columns:
        display[col] = display[col].map({True: "yes", False: "no"})

    # Truncate to max 15 rows for readability
    display = display.head(15)

    if display.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No pairs selected.", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(out_path)

    n_rows, n_cols = display.shape
    fig_h = max(2.0, 0.45 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.4), fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(n_cols)))

    # Header styling
    for j in range(n_cols):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, n_rows + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i, j].set_facecolor(color)

    ax.set_title("Pair selection — cointegration statistics", fontsize=12,
                 fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)
