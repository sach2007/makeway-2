"""Unit tests for Phase 1: Event-driven backtester, purged walk-forward CV, and metrics."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.research.backtester import EventDrivenBacktester
from trading_bot.research.metrics import QuantMetrics
from trading_bot.research.purged_cv import PurgedWalkForwardCV


def test_zero_cost_is_strictly_impossible():
    """Verify backtester enforces non-zero costs by default."""
    with pytest.raises(ValueError, match="Zero-cost backtesting is prohibited"):
        EventDrivenBacktester(brokerage_per_order=0.0)

    with pytest.raises(ValueError, match="Zero-cost backtesting is prohibited"):
        EventDrivenBacktester(base_half_spread_bps=0.0)


def test_slippage_market_impact_scales_with_volume():
    """Verify non-linear slippage expands with order size relative to volume."""
    bt = EventDrivenBacktester()
    small_slip = bt.compute_slippage(order_qty=100, daily_volume=1_000_000)
    large_slip = bt.compute_slippage(order_qty=50_000, daily_volume=1_000_000)

    assert small_slip > 0.0
    assert large_slip > small_slip


def test_purged_walk_forward_cv_purges_and_embargos():
    """Verify that purging removes overlapping samples near fold boundaries."""
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 200, 200)
    })

    cv = PurgedWalkForwardCV(n_splits=3, min_train_pct=0.40, label_horizon_bars=5, embargo_pct=0.02)
    splits = list(cv.split(df))
    assert len(splits) == 3

    for train_slice, test_slice, meta in splits:
        # Check purging gap: test_start index must be >= train_end index + label_horizon_bars
        train_last_idx = train_slice.index[-1]
        test_first_idx = test_slice.index[0]
        assert test_first_idx - train_last_idx >= 5


def test_quant_metrics_calculation():
    """Verify mathematical correctness of institutional metrics."""
    # Constant 1% daily return over 20 days
    rets = [0.01] * 20
    sharpe = QuantMetrics.calculate_sharpe(rets, rf_annual=0.0)
    # Since std is 0 for identical returns, Sharpe should safely be 0.0
    assert sharpe == 0.0

    # Variable returns
    norm_rets = [0.02, -0.01, 0.015, -0.005, 0.03, -0.01, 0.02]
    sharpe_v = QuantMetrics.calculate_sharpe(norm_rets, rf_annual=0.0)
    sortino_v = QuantMetrics.calculate_sortino(norm_rets, rf_annual=0.0)
    assert sharpe_v > 0.0
    assert sortino_v > 0.0

    # Max drawdown check
    wealth = [100, 120, 110, 90, 105, 130]
    dd_amt, dd_pct = QuantMetrics.calculate_max_drawdown(wealth)
    # Peak is 120, trough is 90 -> dd = 30 / 120 = 25%
    assert dd_amt == 30.0
    assert dd_pct == 25.0

    # Calmar ratio
    calmar = QuantMetrics.calculate_calmar(cagr_pct=25.0, max_drawdown_pct=10.0)
    assert calmar == 2.5
