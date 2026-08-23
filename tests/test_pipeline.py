"""End-to-end and unit tests for the stat-arb pipeline.

Covers:
- Data generation reproducibility
- Screener returns correctly typed results
- OU fit produces positive theta and finite half-life
- Backtest result shape consistency
- Cost model arithmetic
- Metrics sign and range checks
- Full pipeline runs without error
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from statarb.backtest import run_backtest
from statarb.costs import CostModel
from statarb.data import generate_prices
from statarb.metrics import compute_metrics, max_drawdown
from statarb.report import PipelineConfig, run_pipeline
from statarb.screener import screen_pairs
from statarb.spread import fit_ou, rolling_zscore
from statarb.stats import pair_stats_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prices_small():
    """25-asset, 504-day synthetic panel seeded for reproducibility."""
    return generate_prices(n_assets=25, n_days=504, seed=0)


@pytest.fixture(scope="module")
def top_pair(prices_small):
    """Best cointegrated pair from the small panel."""
    pairs = screen_pairs(prices_small, max_pairs=1, pvalue_cutoff=0.20)
    assert pairs, "No cointegrated pair found with pvalue_cutoff=0.20"
    return pairs[0]


@pytest.fixture(scope="module")
def backtest_result(prices_small, top_pair):
    """Full backtest result for top_pair."""
    ou = fit_ou(top_pair.spread)
    z = rolling_zscore(top_pair.spread, ou_params=ou)
    return run_backtest(
        prices_a=prices_small[top_pair.ticker_a],
        prices_b=prices_small[top_pair.ticker_b],
        hedge_ratio=top_pair.hedge_ratio,
        z_score=z,
    )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

class TestDataGeneration:
    def test_shape(self):
        df = generate_prices(n_assets=10, n_days=100, seed=1)
        assert df.shape == (100, 10)

    def test_reproducible(self):
        a = generate_prices(n_assets=5, n_days=50, seed=7)
        b = generate_prices(n_assets=5, n_days=50, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds(self):
        a = generate_prices(n_assets=5, n_days=50, seed=1)
        b = generate_prices(n_assets=5, n_days=50, seed=2)
        assert not a.equals(b)

    def test_prices_positive(self):
        df = generate_prices(n_assets=8, n_days=60, seed=3)
        assert (df.values > 0).all()

    def test_index_is_business_days(self):
        df = generate_prices(n_assets=4, n_days=20, seed=0)
        assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

class TestScreener:
    def test_returns_list(self, prices_small):
        results = screen_pairs(prices_small, max_pairs=5, pvalue_cutoff=0.20)
        assert isinstance(results, list)

    def test_sorted_by_pvalue(self, prices_small):
        results = screen_pairs(prices_small, max_pairs=10, pvalue_cutoff=0.20)
        if len(results) >= 2:
            pvals = [r.eg_pvalue for r in results]
            assert pvals == sorted(pvals)

    def test_pvalue_within_cutoff(self, prices_small):
        cutoff = 0.10
        results = screen_pairs(prices_small, max_pairs=10, pvalue_cutoff=cutoff)
        for r in results:
            assert r.eg_pvalue <= cutoff

    def test_spread_length_matches_prices(self, prices_small, top_pair):
        assert len(top_pair.spread) > 0
        assert len(top_pair.spread) <= len(prices_small)


# ---------------------------------------------------------------------------
# OU fitting
# ---------------------------------------------------------------------------

class TestOUFit:
    def test_theta_positive(self, top_pair):
        ou = fit_ou(top_pair.spread)
        assert ou.theta > 0

    def test_half_life_finite(self, top_pair):
        ou = fit_ou(top_pair.spread)
        assert math.isfinite(ou.half_life_days)
        assert ou.half_life_days > 0

    def test_sigma_eq_positive(self, top_pair):
        ou = fit_ou(top_pair.spread)
        assert ou.sigma_eq > 0

    def test_rolling_zscore_shape(self, top_pair):
        ou = fit_ou(top_pair.spread)
        z = rolling_zscore(top_pair.spread, ou_params=ou)
        assert len(z) == len(top_pair.spread)

    def test_rolling_zscore_window(self, top_pair):
        z = rolling_zscore(top_pair.spread, window=30)
        assert len(z) == len(top_pair.spread)

    def test_fit_ou_raises_on_short_series(self):
        short = pd.Series(np.random.randn(10))
        with pytest.raises(ValueError):
            fit_ou(short)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_equity_length(self, backtest_result, top_pair):
        assert len(backtest_result.equity) == len(top_pair.spread)

    def test_equity_is_cumsum_of_daily_pnl(self, backtest_result):
        reconstructed = backtest_result.daily_pnl.cumsum()
        pd.testing.assert_series_equal(
            backtest_result.equity, reconstructed, check_names=False
        )

    def test_positions_values(self, backtest_result):
        unique = set(backtest_result.positions.unique())
        assert unique <= {-1, 0, 1}

    def test_costs_nonnegative(self, backtest_result):
        assert backtest_result.costs_total >= 0

    def test_zero_notional_no_cost(self, prices_small, top_pair):
        ou = fit_ou(top_pair.spread)
        z = rolling_zscore(top_pair.spread, ou_params=ou)
        result = run_backtest(
            prices_small[top_pair.ticker_a],
            prices_small[top_pair.ticker_b],
            top_pair.hedge_ratio,
            z,
            notional=0.0,
        )
        assert result.costs_total == 0.0

    def test_high_entry_z_no_trades(self, prices_small, top_pair):
        ou = fit_ou(top_pair.spread)
        z = rolling_zscore(top_pair.spread, ou_params=ou)
        result = run_backtest(
            prices_small[top_pair.ticker_a],
            prices_small[top_pair.ticker_b],
            top_pair.hedge_ratio,
            z,
            entry_z=100.0,  # never triggered
        )
        assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_one_way_scales_with_notional(self):
        m = CostModel(commission_bps=5, slippage_bps=3)
        assert m.one_way(200_000) == pytest.approx(2 * m.one_way(100_000))

    def test_round_trip_twice_one_way(self):
        m = CostModel(commission_bps=4, slippage_bps=2)
        assert m.round_trip(50_000) == pytest.approx(2 * m.one_way(50_000))

    def test_pair_round_trip_symmetric(self):
        m = CostModel()
        assert m.pair_round_trip(100_000, 80_000) == pytest.approx(
            m.pair_round_trip(80_000, 100_000)
        )

    def test_zero_notional_zero_cost(self):
        m = CostModel()
        assert m.one_way(0) == 0.0

    def test_bps_arithmetic(self):
        m = CostModel(commission_bps=10, slippage_bps=0)
        # 10 bps of 100_000 = 100
        assert m.one_way(100_000) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_sharpe_finite(self, backtest_result):
        m = compute_metrics(backtest_result)
        assert math.isfinite(m.sharpe)

    def test_max_drawdown_nonpositive(self, backtest_result):
        m = compute_metrics(backtest_result)
        assert m.max_drawdown_pct <= 0

    def test_hit_rate_in_unit_interval(self, backtest_result):
        m = compute_metrics(backtest_result)
        assert 0.0 <= m.hit_rate <= 1.0

    def test_n_trades_matches_trade_list(self, backtest_result):
        m = compute_metrics(backtest_result)
        assert m.n_trades == len(backtest_result.trades)

    def test_costs_total_matches_result(self, backtest_result):
        m = compute_metrics(backtest_result)
        assert m.costs_total == pytest.approx(backtest_result.costs_total)

    def test_max_drawdown_flat_equity(self):
        flat = pd.Series([100.0] * 50)
        assert max_drawdown(flat) == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_monotone_increase(self):
        up = pd.Series(range(1, 51), dtype=float)
        assert max_drawdown(up) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Stats table
# ---------------------------------------------------------------------------

class TestStatsTable:
    def test_pair_stats_table_columns(self, prices_small):
        pairs = screen_pairs(prices_small, max_pairs=3, pvalue_cutoff=0.20)
        df = pair_stats_table(pairs)
        required = {"pair", "hedge_ratio", "eg_pvalue", "adf_pvalue", "johansen_trace"}
        assert required.issubset(set(df.columns))

    def test_pair_stats_table_rows(self, prices_small):
        pairs = screen_pairs(prices_small, max_pairs=3, pvalue_cutoff=0.20)
        df = pair_stats_table(pairs)
        assert len(df) == len(pairs)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_pipeline_runs(self, tmp_path):
        cfg = PipelineConfig(
            seed=42,
            n_assets=15,
            n_days=252,
            max_pairs=2,
            pvalue_cutoff=0.20,
        )
        report = run_pipeline(cfg, out_dir=str(tmp_path))
        assert report.endswith("report.md")
        assert (tmp_path / "report.md").exists()

    def test_pipeline_produces_charts(self, tmp_path):
        cfg = PipelineConfig(
            seed=42,
            n_assets=15,
            n_days=252,
            max_pairs=2,
            pvalue_cutoff=0.20,
        )
        run_pipeline(cfg, out_dir=str(tmp_path))
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 1  # at least pair table chart

    def test_pipeline_reproducible(self, tmp_path):
        """Two runs with the same seed produce identical report content."""
        cfg = PipelineConfig(seed=42, n_assets=15, n_days=252,
                             max_pairs=2, pvalue_cutoff=0.20)
        r1 = tmp_path / "run1"
        r2 = tmp_path / "run2"
        run_pipeline(cfg, out_dir=str(r1))
        run_pipeline(cfg, out_dir=str(r2))
        text1 = (r1 / "report.md").read_text()
        text2 = (r2 / "report.md").read_text()
        assert text1 == text2
