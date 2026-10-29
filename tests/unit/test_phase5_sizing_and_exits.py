"""Unit tests for Phase 5: Dynamic Sizing (Fractional Kelly, Vol-Targeting) and Dynamic Exits (ATR stops, trailing, time exit)."""
from __future__ import annotations
import pytest

from trading_bot.api.schemas import Portfolio, PortfolioPosition, TradeAction
from trading_bot.risk.limits import RiskLimits
from trading_bot.risk.position_sizing import PositionSizer
from trading_bot.risk.exposure import ExposureManager


def test_fractional_kelly_sizing():
    """Verify Fractional Kelly correctly scales with win probability and returns 0 for negative edge."""
    limits = RiskLimits()
    sizer = PositionSizer(limits, kelly_fraction=0.25, default_payoff_ratio=1.5)

    # 60% win rate, 1.5 payoff -> edge is positive
    f_pos = sizer.calculate_fractional_kelly(win_probability=0.60, payoff_ratio=1.5)
    # p*b - q = 0.60*1.5 - 0.40 = 0.90 - 0.40 = 0.50
    # full kelly = 0.50 / 1.5 = 0.3333... -> 0.25x = ~0.0833
    assert 0.07 < f_pos < 0.10

    # 30% win rate, 1.5 payoff -> edge is negative (0.3*1.5 - 0.7 = -0.25) -> Kelly must be 0
    f_neg = sizer.calculate_fractional_kelly(win_probability=0.30, payoff_ratio=1.5)
    assert f_neg == 0.0


def test_volatility_targeting_weight():
    """Verify higher asset volatility reduces allocation weight, while low vol receives target weight up to cap."""
    limits = RiskLimits(max_single_trade_pct=0.10)
    sizer = PositionSizer(limits, target_annualized_vol=0.15)

    # Daily target vol = 0.15 / sqrt(252) ~= 0.00945
    w_low_vol = sizer.calculate_volatility_target_weight(asset_daily_vol=0.01)
    w_high_vol = sizer.calculate_volatility_target_weight(asset_daily_vol=0.04)

    assert w_low_vol > w_high_vol
    assert w_high_vol <= 1.0


def test_correlation_aware_limit():
    """Verify high correlation (>0.70) with existing holding throttles allocation to 1.5x cap."""
    limits = RiskLimits(max_position_pct=0.15, max_single_trade_pct=0.10)
    sizer = PositionSizer(limits)

    # Existing holding: AERX worth 140,000 in a 1,000,000 portfolio (14% allocation)
    portfolio = Portfolio(
        total_wealth=1_000_000.0,
        cash=500_000.0,
        holdings=[PortfolioPosition(ticker="AERX", quantity=140, average_buy_price=1000.0, current_price=1000.0, unrealized_pnl=0.0, unrealized_pnl_pct=0.0, is_short=False)]
    )

    # Correlation matrix where MOTR is 0.85 correlated with AERX
    corr = {
        "MOTR": {"AERX": 0.85}
    }

    # Combined cap is 1.5 * 150,000 = 225,000.
    # AERX already uses 140,000 -> remaining cluster room is 85,000 (85 shares at 1000).
    qty = sizer.calculate_quantity(
        action=TradeAction.BUY,
        price=1000.0,
        conviction=0.8,
        portfolio=portfolio,
        available_cash=500_000.0,
        current_holding_qty=0,
        volatility=0.015,
        correlation_matrix=corr,
        ticker="MOTR"
    )

    assert qty <= 85
    assert qty > 0


def test_dynamic_atr_stop_loss():
    """Verify ExposureManager triggers ATR stop loss based on volatility."""
    mgr = ExposureManager(atr_stop_multiplier=2.0)
    # Entry at 100.0, in Low-Vol Trend k1=2.5, stop level = 100 - (2.5 * 3.0) = 92.5. Price 92.0 breaches stop.
    portfolio = Portfolio(
        total_wealth=100_000.0,
        cash=50_000.0,
        holdings=[PortfolioPosition(ticker="TEST", quantity=100, average_buy_price=100.0, current_price=92.0, unrealized_pnl=-800.0, unrealized_pnl_pct=-8.0, is_short=False)]
    )
    context = {"TEST": {"atr": 3.0, "regime": "MarketRegimeState.LOW_VOL_TREND"}}

    exits = mgr.check_exits(portfolio, market_context=context)
    assert len(exits) == 1
    ticker, action, qty, reason = exits[0]
    assert ticker == "TEST"
    assert action == TradeAction.SELL
    assert "ATR_STOP_LOSS" in reason


def test_dynamic_atr_trailing_stop():
    """Verify ExposureManager triggers ATR trailing stop when price pulls back from peak."""
    mgr = ExposureManager(atr_trail_multiplier=1.5)
    # Step 1: Initial position at 100.0
    port1 = Portfolio(
        total_wealth=100_000.0,
        cash=50_000.0,
        holdings=[PortfolioPosition(ticker="RUNNER", quantity=100, average_buy_price=100.0, current_price=100.0, unrealized_pnl=0.0, unrealized_pnl_pct=0.0, is_short=False)]
    )
    context = {"RUNNER": {"atr": 2.0, "regime": "LOW_VOL_TREND"}}
    mgr.check_exits(port1, market_context=context)

    # Step 2: Price surges to 115.0
    port2 = Portfolio(
        total_wealth=115_000.0,
        cash=50_000.0,
        holdings=[PortfolioPosition(ticker="RUNNER", quantity=100, average_buy_price=100.0, current_price=115.0, unrealized_pnl=1500.0, unrealized_pnl_pct=15.0, is_short=False)]
    )
    mgr.check_exits(port2, market_context=context)

    # Step 3: Price pulls back to 110.0 (Peak was 115.0. Trailing stop level = 115 - 1.5 * 1.25 * 2.0 = 111.25 in trending regime)
    port3 = Portfolio(
        total_wealth=110_000.0,
        cash=50_000.0,
        holdings=[PortfolioPosition(ticker="RUNNER", quantity=100, average_buy_price=100.0, current_price=110.0, unrealized_pnl=1000.0, unrealized_pnl_pct=10.0, is_short=False)]
    )
    exits = mgr.check_exits(port3, market_context=context)
    assert len(exits) == 1
    assert "ATR_TRAILING_STOP" in exits[0][3]


def test_time_based_exit():
    """Verify time-based exit fires when trade has gone nowhere after max bars."""
    mgr = ExposureManager(max_bars_held=5)
    context = {"FLAT": {"atr": 4.0, "regime": "HIGH_VOL_MEAN_REVERT"}}

    for bar in range(7):
        port = Portfolio(
            total_wealth=100_000.0,
            cash=50_000.0,
            holdings=[PortfolioPosition(ticker="FLAT", quantity=50, average_buy_price=100.0, current_price=100.5, unrealized_pnl=25.0, unrealized_pnl_pct=0.5, is_short=False)]
        )
        exits = mgr.check_exits(port, market_context=context)

    # By bar 7 (> 5*0.50 = 2 bars in mean-revert), with price move 0.5 < 0.5*4.0 (2.0), time exit should trigger
    assert len(exits) == 1
    assert "TIME_EXIT" in exits[0][3]
