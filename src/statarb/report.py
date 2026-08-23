"""End-to-end report generator.

Orchestrates the full backtest pipeline: data generation, pair screening,
OU fitting, backtest execution, performance metrics, chart generation, and
Markdown report output.

Usage
-----
    from statarb.report import run_pipeline, PipelineConfig

    cfg = PipelineConfig(seed=42, n_assets=40, n_days=756, max_pairs=5)
    summary = run_pipeline(cfg, out_dir="reports")
    print(summary)
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from statarb.backtest import BacktestResult, run_backtest
from statarb.charts import plot_equity_drawdown, plot_pair_selection_table, plot_zscore_signals
from statarb.costs import CostModel
from statarb.data import generate_prices, load_prices
from statarb.metrics import PerfMetrics, compute_metrics, metrics_to_dict
from statarb.screener import PairResult, screen_pairs
from statarb.spread import OUParams, fit_ou, rolling_zscore
from statarb.stats import format_markdown_table, pair_stats_table


@dataclass
class PipelineConfig:
    """Configuration for the full backtest pipeline.

    Attributes
    ----------
    seed : int
        RNG seed for reproducible synthetic data.
    n_assets : int
        Number of synthetic assets to generate.
    n_days : int
        Number of trading days in the simulated history.
    max_pairs : int
        Maximum number of cointegrated pairs to backtest.
    pvalue_cutoff : float
        Engle-Granger p-value cutoff for pair inclusion.
    entry_z : float
        Z-score threshold to enter a position.
    exit_z : float
        Z-score threshold to exit a position.
    stop_z : float
        Z-score stop-loss threshold.
    notional : float
        Dollar notional per pair (A leg).
    commission_bps : float
        Broker commission per leg, in basis points.
    slippage_bps : float
        One-way slippage per leg, in basis points.
    data_path : str or None
        Path to a real-data CSV (columns = tickers, rows = dates).
        If None, synthetic data is generated from *seed*.
    """

    seed: int = 42
    n_assets: int = 40
    n_days: int = 756
    max_pairs: int = 5
    pvalue_cutoff: float = 0.05
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    notional: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 3.0
    data_path: str | None = None


class PairBacktestResult(NamedTuple):
    """All artefacts for one backtested pair."""

    pair: PairResult
    ou: OUParams
    z_score: pd.Series
    result: BacktestResult
    metrics: PerfMetrics


def run_pipeline(
    cfg: PipelineConfig,
    out_dir: str = "reports",
) -> str:
    """Execute the full stat-arb pipeline and write artefacts to *out_dir*.

    Steps
    -----
    1. Load or generate price data.
    2. Screen all pairs by Engle-Granger cointegration.
    3. Fit OU parameters on each selected spread.
    4. Run the vectorised backtest with cost model applied.
    5. Compute performance metrics.
    6. Generate equity curve, z-score, and pair table charts.
    7. Write a Markdown report.

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline parameters.
    out_dir : str
        Directory for output charts and report.

    Returns
    -------
    str
        Path to the generated Markdown report.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Data                                                             #
    # ------------------------------------------------------------------ #
    if cfg.data_path:
        prices = load_prices(cfg.data_path)
    else:
        prices = generate_prices(
            n_assets=cfg.n_assets,
            n_days=cfg.n_days,
            seed=cfg.seed,
        )

    # ------------------------------------------------------------------ #
    # 2. Screen pairs                                                     #
    # ------------------------------------------------------------------ #
    pairs = screen_pairs(
        prices,
        max_pairs=cfg.max_pairs,
        pvalue_cutoff=cfg.pvalue_cutoff,
    )

    if not pairs:
        # Relax cutoff and retry once
        pairs = screen_pairs(prices, max_pairs=cfg.max_pairs, pvalue_cutoff=0.20)

    # ------------------------------------------------------------------ #
    # 3-5. OU fit + backtest + metrics                                    #
    # ------------------------------------------------------------------ #
    results: list[PairBacktestResult] = []
    ou_map: dict[str, OUParams] = {}

    for pr in pairs:
        try:
            ou = fit_ou(pr.spread)
        except ValueError:
            continue
        z = rolling_zscore(pr.spread, ou_params=ou)
        bt = run_backtest(
            prices_a=prices[pr.ticker_a],
            prices_b=prices[pr.ticker_b],
            hedge_ratio=pr.hedge_ratio,
            z_score=z,
            entry_z=cfg.entry_z,
            exit_z=cfg.exit_z,
            stop_z=cfg.stop_z,
            notional=cfg.notional,
            commission_bps=cfg.commission_bps,
            slippage_bps=cfg.slippage_bps,
        )
        m = compute_metrics(bt)
        key = f"{pr.ticker_a}/{pr.ticker_b}"
        ou_map[key] = ou
        results.append(PairBacktestResult(pr, ou, z, bt, m))

    # ------------------------------------------------------------------ #
    # 6. Charts                                                           #
    # ------------------------------------------------------------------ #
    chart_paths: dict[str, dict[str, str]] = {}
    for pbr in results:
        label = f"{pbr.pair.ticker_a}/{pbr.pair.ticker_b}"
        eq_path = plot_equity_drawdown(pbr.result, pair_label=label, out_dir=out_dir)
        zs_path = plot_zscore_signals(pbr.z_score, pbr.result.positions,
                                      pair_label=label, out_dir=out_dir,
                                      entry_z=cfg.entry_z, exit_z=cfg.exit_z)
        chart_paths[label] = {"equity": eq_path, "zscore": zs_path}

    # Pair selection table chart
    stats_df = pair_stats_table(pairs, ou_map)
    table_chart = plot_pair_selection_table(stats_df, out_dir=out_dir)

    # ------------------------------------------------------------------ #
    # 7. Markdown report                                                  #
    # ------------------------------------------------------------------ #
    report_path = os.path.join(out_dir, "report.md")
    _write_markdown_report(results, stats_df, chart_paths, table_chart,
                           cfg, report_path)

    return report_path


def _write_markdown_report(
    results: list[PairBacktestResult],
    stats_df: pd.DataFrame,
    chart_paths: dict[str, dict[str, str]],
    table_chart: str,
    cfg: PipelineConfig,
    report_path: str,
) -> None:
    """Write the Markdown report to *report_path*."""
    lines: list[str] = []

    lines.append("# Statistical arbitrage backtest report\n")

    # Config summary
    lines.append("## Configuration\n")
    lines.append(f"- Seed: `{cfg.seed}`")
    lines.append(f"- Assets: {cfg.n_assets}, Days: {cfg.n_days}")
    lines.append(f"- Entry z: {cfg.entry_z}, Exit z: {cfg.exit_z}, Stop z: {cfg.stop_z}")
    lines.append(f"- Notional per pair: ${cfg.notional:,.0f}")
    lines.append(f"- Commission: {cfg.commission_bps} bps, Slippage: {cfg.slippage_bps} bps")
    lines.append("")

    # Pair selection table
    lines.append("## Pair selection\n")
    lines.append(format_markdown_table(stats_df))
    lines.append("")
    rel_table = os.path.relpath(table_chart, os.path.dirname(report_path))
    lines.append(f"![Pair selection table]({rel_table})\n")

    # Per-pair results
    lines.append("## Per-pair results\n")
    for pbr in results:
        label = f"{pbr.pair.ticker_a}/{pbr.pair.ticker_b}"
        lines.append(f"### {label}\n")

        # Statistical tests
        lines.append("**Statistical tests**\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| EG p-value | {pbr.pair.eg_pvalue:.4f} |")
        lines.append(f"| ADF p-value | {pbr.pair.adf_pvalue:.4f} |")
        lines.append(f"| Johansen trace | {pbr.pair.joh_trace:.2f} |")
        lines.append(f"| Hedge ratio | {pbr.pair.hedge_ratio:.4f} |")
        lines.append(f"| OU half-life (days) | {pbr.ou.half_life_days:.1f} |")
        lines.append(f"| OU sigma_eq | {pbr.ou.sigma_eq:.5f} |")
        lines.append("")

        # Performance metrics
        lines.append("**Performance**\n")
        m_dict = metrics_to_dict(pbr.metrics)
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in m_dict.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        # Charts
        if label in chart_paths:
            eq_rel = os.path.relpath(chart_paths[label]["equity"],
                                     os.path.dirname(report_path))
            zs_rel = os.path.relpath(chart_paths[label]["zscore"],
                                     os.path.dirname(report_path))
            lines.append(f"![Equity curve]({eq_rel})\n")
            lines.append(f"![Z-score signals]({zs_rel})\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
