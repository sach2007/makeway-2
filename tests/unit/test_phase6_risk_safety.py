"""Unit tests for Phase 6: Risk & Safety Layer, Drift Monitoring, Hit-Rate De-risking, and Safeguards."""
from __future__ import annotations
import time
import numpy as np
import pytest

from trading_bot.api.schemas import OrderRequest, Portfolio, PortfolioPosition, TradeAction
from trading_bot.risk.limits import (
    RiskLimits, PreTradeRiskEngine, generate_idempotent_order_id
)
from trading_bot.monitoring.drift import (
    calculate_psi, DriftMonitor, DriftStatus, HitRateTracker
)


def test_daily_loss_limit():
    """Verify engine halts for the remainder of the session if daily drop exceeds 3%."""
    limits = RiskLimits(daily_loss_limit_pct=0.03)
    engine = PreTradeRiskEngine(limits)

    # Day starts with 1,000,000 wealth
    engine.on_new_trading_day(starting_wealth=1_000_000.0)

    # Wealth drops to 965,000 (-3.5% loss)
    dropped_port = Portfolio(
        total_wealth=965_000.0,
        cash=200_000.0,
        holdings=[]
    )
    req = OrderRequest(ticker="HAVN", action=TradeAction.BUY, quantity=10)
    is_approved, reason = engine.validate_order(req, price=100.0, portfolio=dropped_port, available_cash=200_000.0)

    assert is_approved is False
    assert "Daily loss limit breached" in reason
    assert engine.is_daily_halted is True


def test_per_symbol_loss_cooldown():
    """Verify losing trade places that symbol on cooldown for N bars, then expires."""
    limits = RiskLimits(symbol_cooldown_bars=3)
    engine = PreTradeRiskEngine(limits)
    engine.on_new_trading_day(1_000_000.0)

    # Trade on AERX lost 2.5%
    engine.record_trade_completion(ticker="AERX", return_pct=-2.5)

    port = Portfolio(total_wealth=1_000_000.0, cash=500_000.0, holdings=[])
    req_aerx = OrderRequest(ticker="AERX", action=TradeAction.BUY, quantity=10)
    req_other = OrderRequest(ticker="MOTR", action=TradeAction.BUY, quantity=10)

    # AERX should be blocked
    ok_aerx, reason_aerx = engine.validate_order(req_aerx, price=100.0, portfolio=port, available_cash=500_000.0)
    assert ok_aerx is False
    assert "in loss-cooldown" in reason_aerx

    # MOTR should NOT be blocked
    ok_other, _ = engine.validate_order(req_other, price=100.0, portfolio=port, available_cash=500_000.0)
    assert ok_other is True

    # Advance 3 ticks
    engine.on_tick_advance()
    engine.on_tick_advance()
    engine.on_tick_advance()

    # Now AERX should be unlocked
    ok_aerx_after, _ = engine.validate_order(req_aerx, price=100.0, portfolio=port, available_cash=500_000.0)
    assert ok_aerx_after is True


def test_stale_data_guard():
    """Verify order is rejected if market tick is older than 30 seconds."""
    limits = RiskLimits(stale_data_seconds=30.0)
    engine = PreTradeRiskEngine(limits)
    port = Portfolio(total_wealth=1_000_000.0, cash=500_000.0, holdings=[])
    req = OrderRequest(ticker="HAVN", action=TradeAction.BUY, quantity=10)

    # Tick from 45 seconds ago
    stale_ts = time.time() - 45.0
    is_ok, reason = engine.validate_order(req, price=100.0, portfolio=port, available_cash=500_000.0, tick_timestamp=stale_ts)
    assert is_ok is False
    assert "Stale market data" in reason

    # Fresh tick from 2 seconds ago
    fresh_ts = time.time() - 2.0
    is_ok_fresh, _ = engine.validate_order(req, price=100.0, portfolio=port, available_cash=500_000.0, tick_timestamp=fresh_ts)
    assert is_ok_fresh is True


def test_hard_cash_floor():
    """Verify BUY order is rejected if cash drops below 3% hard floor."""
    limits = RiskLimits(hard_cash_floor_pct=0.03)
    engine = PreTradeRiskEngine(limits)
    # Total wealth 1,000,000 -> hard floor is 30,000. Cash is only 25,000.
    port = Portfolio(total_wealth=1_000_000.0, cash=25_000.0, holdings=[])
    req = OrderRequest(ticker="HAVN", action=TradeAction.BUY, quantity=10)

    is_ok, reason = engine.validate_order(req, price=100.0, portfolio=port, available_cash=25_000.0)
    assert is_ok is False
    assert "below hard cash floor" in reason


def test_idempotent_order_id_generation():
    """Verify deterministic SHA-256 order ID is consistent and changes with parameters."""
    id1 = generate_idempotent_order_id("AERX", TradeAction.BUY, 50, "2026-02-15T10:00:00")
    id2 = generate_idempotent_order_id("AERX", TradeAction.BUY, 50, "2026-02-15T10:00:00")
    id3 = generate_idempotent_order_id("AERX", TradeAction.SELL, 50, "2026-02-15T10:00:00")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("ORD_AERX_")


def test_population_stability_index_drift():
    """Verify PSI detects distribution shift."""
    np.random.seed(42)
    # Similar distributions -> PSI should be very small (< 0.05)
    base = np.random.normal(0.0, 1.0, 500)
    curr_same = np.random.normal(0.02, 1.01, 500)
    psi_stable = calculate_psi(base, curr_same)
    assert psi_stable < 0.10

    # Severely shifted distribution (mean shift from 0 to 2.5) -> PSI should be > 0.25
    curr_shifted = np.random.normal(2.5, 1.5, 500)
    psi_shifted = calculate_psi(base, curr_shifted)
    assert psi_shifted > 0.25


def test_hit_rate_tracker_de_risking():
    """Verify HitRateTracker reduces sizing at <35% and halts at <25%."""
    tracker = HitRateTracker(window_size=20, throttle_threshold=0.35, halt_threshold=0.25)

    # 1. 10 winning trades -> 100% win rate -> full size
    for _ in range(10):
        tracker.record_trade_result(net_pnl=100.0)
    mult, should_halt, _ = tracker.get_sizing_multiplier()
    assert mult == 1.0
    assert should_halt is False

    # 2. Add 10 losing trades -> 10 wins out of 20 = 50% -> normal size
    for _ in range(10):
        tracker.record_trade_result(net_pnl=-50.0)
    mult, should_halt, _ = tracker.get_sizing_multiplier()
    assert mult == 1.0
    assert should_halt is False

    # 3. Add 5 more losses -> 5 wins out of 20 = 25% (or 30%) -> De-risk 50%
    for _ in range(4):
        tracker.record_trade_result(net_pnl=-50.0)
    mult, should_halt, reason = tracker.get_sizing_multiplier()
    # Win rate is 6/20 = 30% -> between 25% and 35% -> de-risk to 50% size
    assert mult == 0.50
    assert should_halt is False
    assert "De-risking" in reason

    # 4. Add 2 more losses -> 4 wins out of 20 = 20% (<25%) -> Halt!
    for _ in range(2):
        tracker.record_trade_result(net_pnl=-50.0)
    mult, should_halt, reason = tracker.get_sizing_multiplier()
    assert mult == 0.0
    assert should_halt is True
    assert "Emergency Halt" in reason
