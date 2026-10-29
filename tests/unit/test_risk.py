"""Unit tests for pre-trade risk engine, sizing, take-profit, and exposure controls."""
import pytest

from trading_bot.api.schemas import OrderRequest, Portfolio, PortfolioPosition, TradeAction
from trading_bot.risk.exposure import ExposureManager
from trading_bot.risk.limits import PreTradeRiskEngine, RiskLimits
from trading_bot.risk.position_sizing import PositionSizer


def test_risk_rejects_insufficient_cash():
    limits = RiskLimits(min_cash_reserve_pct=0.05)
    engine = PreTradeRiskEngine(limits)
    portfolio = Portfolio(
        initial_capital=100_000,
        capital_remaining=10_000,
        total_wealth=100_000
    )
    req = OrderRequest(ticker="AETX", action=TradeAction.BUY, quantity=80)
    is_ok, reason = engine.validate_order(req, price=100.0, portfolio=portfolio, available_cash=7_000)
    assert not is_ok
    assert "Insufficient cash" in reason


def test_risk_rejects_invalid_short_selling():
    limits = RiskLimits(short_selling_enabled=False)
    engine = PreTradeRiskEngine(limits)
    portfolio = Portfolio(initial_capital=100_000, capital_remaining=100_000, total_wealth=100_000)
    req = OrderRequest(ticker="AETX", action=TradeAction.SHORT, quantity=50)
    is_ok, reason = engine.validate_order(req, price=100.0, portfolio=portfolio, available_cash=100_000)
    assert not is_ok
    assert "Short selling is disabled" in reason


def test_position_sizer_returns_safe_quantities():
    limits = RiskLimits(max_position_pct=0.15, max_single_trade_pct=0.10)
    sizer = PositionSizer(limits, base_trade_pct=0.05)
    portfolio = Portfolio(initial_capital=1_000_000, capital_remaining=800_000, total_wealth=1_000_000)

    qty = sizer.calculate_quantity(
        action=TradeAction.BUY,
        price=100.0,
        conviction=1.0,
        portfolio=portfolio,
        available_cash=800_000
    )
    assert qty > 0
    assert qty * 100 <= 100_000


def test_exposure_manager_emergency_halt():
    manager = ExposureManager(emergency_halt_drawdown_pct=0.20)
    manager.update_portfolio(Portfolio(total_wealth=1_000_000))
    assert not manager.is_halted

    manager.update_portfolio(Portfolio(total_wealth=750_000))
    assert manager.is_halted


def test_exposure_manager_take_profit_and_stop_loss():
    manager = ExposureManager(
        per_position_stop_loss_pct=0.05,
        take_profit_pct=0.15
    )

    portfolio = Portfolio(
        total_wealth=1_000_000,
        holdings=[
            PortfolioPosition(ticker="MOTR", quantity=100, unrealized_pnl_pct=25.0, is_short=False),
            PortfolioPosition(ticker="BLUX", quantity=50, unrealized_pnl_pct=-8.0, is_short=False),
            PortfolioPosition(ticker="ROUT", quantity=80, unrealized_pnl_pct=4.0, is_short=False),
            PortfolioPosition(ticker="HMDP", quantity=-60, unrealized_pnl_pct=18.0, is_short=True),
        ]
    )

    exits = manager.check_exits(portfolio)
    # MOTR should trigger Take Profit (+25%)
    # BLUX should trigger Stop Loss (-8%)
    # HMDP short should trigger Take Profit (+18%)
    # ROUT (+4%) should NOT trigger exit
    assert len(exits) == 3

    exit_tickers = {e[0]: e[1] for e in exits}
    assert exit_tickers["MOTR"] == TradeAction.SELL
    assert exit_tickers["BLUX"] == TradeAction.SELL
    assert exit_tickers["HMDP"] == TradeAction.COVER
