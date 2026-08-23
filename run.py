"""CLI entry point for the statistical arbitrage backtest.

Run the full pipeline from a single command:

    python run.py [options]

All results (charts, report) are written to the --out-dir directory.

Examples
--------
    # Quick run on synthetic data (default settings)
    python run.py

    # Customise seed, asset count, and output dir
    python run.py --seed 123 --n-assets 30 --n-days 504 --out-dir results

    # Use real price data (CSV: date column + ticker columns)
    python run.py --data-path prices.csv --max-pairs 10

    # Tighter z-score thresholds, higher costs
    python run.py --entry-z 1.5 --exit-z 0.0 --commission-bps 8 --slippage-bps 5
"""

from __future__ import annotations

import argparse
import sys
import time

from statarb.report import PipelineConfig, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Statistical arbitrage backtest on equity pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    g = p.add_argument_group("data")
    g.add_argument("--data-path", default=None,
                   help="Path to a real-data CSV (first column = date, "
                        "remaining = tickers).  If omitted, synthetic data "
                        "is generated.")
    g.add_argument("--seed", type=int, default=42,
                   help="RNG seed for synthetic data and reproducibility.")
    g.add_argument("--n-assets", type=int, default=40,
                   help="Number of synthetic assets to simulate.")
    g.add_argument("--n-days", type=int, default=756,
                   help="Number of trading days in the simulated history.")

    # Screening
    g2 = p.add_argument_group("screening")
    g2.add_argument("--max-pairs", type=int, default=5,
                    help="Maximum number of cointegrated pairs to backtest.")
    g2.add_argument("--pvalue-cutoff", type=float, default=0.05,
                    help="Engle-Granger p-value cutoff for pair inclusion.")

    # Signals
    g3 = p.add_argument_group("signals")
    g3.add_argument("--entry-z", type=float, default=2.0,
                    help="Z-score magnitude to open a position.")
    g3.add_argument("--exit-z", type=float, default=0.5,
                    help="Z-score magnitude (toward zero) to close a position.")
    g3.add_argument("--stop-z", type=float, default=3.5,
                    help="Z-score stop-loss magnitude.")

    # Costs
    g4 = p.add_argument_group("costs")
    g4.add_argument("--notional", type=float, default=100_000.0,
                    help="Dollar notional per pair (A leg).")
    g4.add_argument("--commission-bps", type=float, default=5.0,
                    help="Broker commission per leg, in basis points.")
    g4.add_argument("--slippage-bps", type=float, default=3.0,
                    help="One-way slippage per leg, in basis points.")

    # Output
    p.add_argument("--out-dir", default="reports",
                   help="Directory for output charts and Markdown report.")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = PipelineConfig(
        seed=args.seed,
        n_assets=args.n_assets,
        n_days=args.n_days,
        max_pairs=args.max_pairs,
        pvalue_cutoff=args.pvalue_cutoff,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        notional=args.notional,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        data_path=args.data_path,
    )

    print(f"Running backtest — seed={cfg.seed}, assets={cfg.n_assets}, "
          f"days={cfg.n_days}, max_pairs={cfg.max_pairs}")
    t0 = time.perf_counter()

    report_path = run_pipeline(cfg, out_dir=args.out_dir)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s — report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
