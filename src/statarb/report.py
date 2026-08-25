"""End-to-end report generator.

Runs the full pipeline (screen → fit → backtest → metrics → charts) for
a set of top pairs and writes a Markdown summary to reports/report.md.

Usage
-----
    from statarb.report import generate_report

    generate_report(prices, n_pairs=5, seed=42, out_dir="reports")
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.screener import PairResult, screen_pairs
from statarb.spread import OUParams, fit_ou, rolling_zscore
from statarb.backtest import BacktestResult, run_backtest
from statarb.metrics import MetricsResult, compute_metrics
from statarb.charts import plot_equity_curve, plot_zscore_signals, plot_spread
from statarb.stats import format_markdown_table, pair_stats_table


def _pair_key(r: PairResult) -> str:
    return f"{r.ticker_a}/{r.ticker_b}"


def _safe_fit_ou(spread: pd.Series) -> OUParams | None:
    try:
        return fit_ou(spread)
    except Exception:
        return None


def generate_report(
    prices: pd.DataFrame,
    n_pairs: int = 5,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    notional: float = 100_000.0,
    commission_bps: float = 5.0,
    slippage_bps: float = 3.0,
    pvalue_cutoff: float = 0.10,
    out_dir: str = "reports",
    seed: int = 42,
) -> dict[str, tuple[BacktestResult, MetricsResult]]:
    """Run the full pipeline and write reports.

    Screens pairs, fits OU parameters, runs backtests, produces charts,
    and writes a Markdown summary with a pair selection table.

    Parameters
    ----------
    prices : pd.DataFrame
        Panel of close prices, shape (n_days, n_assets).
    n_pairs : int
        Number of top pairs to backtest.
    entry_z : float
        Z-score entry threshold.
    exit_z : float
        Z-score exit threshold.
    stop_z : float
        Z-score stop-loss threshold.
    notional : float
        Dollar notional per pair.
    commission_bps : float
        Commission per leg in basis points.
    slippage_bps : float
        One-way slippage per leg in basis points.
    pvalue_cutoff : float
        EG test p-value cutoff for pair screening.
    out_dir : str
        Directory for output files.
    seed : int
        Kept for reproducibility note in the report.

    Returns
    -------
    dict mapping pair key -> (BacktestResult, MetricsResult)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Screening pairs (EG p-value < {pvalue_cutoff})...")
    pairs = screen_pairs(prices, max_pairs=n_pairs, pvalue_cutoff=pvalue_cutoff)
    if not pairs:
        print("No cointegrated pairs found at the given cutoff.")
        return {}

    print(f"Found {len(pairs)} pairs. Fitting OU models...")

    ou_map: dict[str, OUParams] = {}
    results: dict[str, tuple[BacktestResult, MetricsResult]] = {}

    for r in pairs:
        key = _pair_key(r)
        ou = _safe_fit_ou(r.spread)
        if ou is None:
            print(f"  {key}: OU fit failed, skipping.")
            continue
        ou_map[key] = ou

        z = rolling_zscore(r.spread, ou_params=ou)
        bt = run_backtest(
            prices_a=prices[r.ticker_a],
            prices_b=prices[r.ticker_b],
            hedge_ratio=r.hedge_ratio,
            z_score=z,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            notional=notional,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        m = compute_metrics(bt, notional=notional)
        results[key] = (bt, m)

        # Charts
        slug = key.replace("/", "_")
        plot_equity_curve(
            bt.equity, m.drawdown_series,
            pair_label=key,
            out_path=f"{out_dir}/equity_{slug}.png",
            sharpe=m.sharpe,
            max_dd_pct=m.max_drawdown_pct,
        )
        plot_zscore_signals(
            z, bt.positions,
            pair_label=key,
            entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
            out_path=f"{out_dir}/zscore_{slug}.png",
        )
        plot_spread(r.spread, pair_label=key, out_path=f"{out_dir}/spread_{slug}.png")

        print(
            f"  {key}: Sharpe={m.sharpe:.2f}, "
            f"MaxDD={m.max_drawdown_pct:.1f}%, "
            f"HitRate={m.hit_rate:.0%}, "
            f"Trades={m.n_trades}"
        )

    # Pair selection table
    stats_df = pair_stats_table(
        [r for r in pairs if _pair_key(r) in ou_map],
        ou_map=ou_map,
    )
    stats_csv = f"{out_dir}/pair_selection.csv"
    stats_df.to_csv(stats_csv, index=False)

    # Performance summary table
    perf_rows = []
    for key, (bt, m) in results.items():
        perf_rows.append({
            "pair": key,
            "sharpe": round(m.sharpe, 3),
            "sortino": round(m.sortino, 3),
            "max_dd_pct": round(m.max_drawdown_pct, 1),
            "total_return_pct": round(m.total_return_pct, 1),
            "n_trades": m.n_trades,
            "hit_rate": f"{m.hit_rate:.0%}",
            "profit_factor": round(m.profit_factor, 2),
            "costs_usd": round(m.costs_total, 0),
        })
    perf_df = pd.DataFrame(perf_rows)

    # Markdown report
    md_lines = [
        "# Backtest report",
        "",
        f"Seed: `{seed}` | Notional per pair: ${notional:,.0f} | "
        f"Entry z: {entry_z} | Exit z: {exit_z} | Stop z: {stop_z}",
        f"Commission: {commission_bps} bps | Slippage: {slippage_bps} bps",
        "",
        "## Pair selection — statistical tests",
        "",
        format_markdown_table(stats_df),
        "",
        "## Performance summary",
        "",
        format_markdown_table(perf_df),
        "",
        "## Charts",
        "",
    ]

    for key in results:
        slug = key.replace("/", "_")
        md_lines += [
            f"### {key}",
            "",
            f"![Equity curve](equity_{slug}.png)",
            "",
            f"![Z-score signals](zscore_{slug}.png)",
            "",
        ]

    md_path = f"{out_dir}/report.md"
    Path(md_path).write_text("\n".join(md_lines))
    print(f"\nReport written to {md_path}")
    return results
