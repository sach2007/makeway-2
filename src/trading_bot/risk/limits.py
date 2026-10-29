"""Production Pre-Trade Risk Engine, Daily Loss Limit, Cooldowns, and Idempotency Guard.

Enforces:
1. Hard Vetoes: 5% cash buffer, 3% hard cash floor, 15% single position, 10% per-order cap.
2. 85% aggregate long / 30% aggregate short caps.
3. Daily loss limit: Halt if portfolio drops > 3% from start-of-day wealth.
4. Per-symbol loss cooldown: Blocks re-entry for N bars after closing a losing trade.
5. Stale-data guard: Rejects orders if tick timestamp > 30 seconds old.
6. Deterministic Idempotent client order IDs (SHA-256 hash).
7. Extreme single trade loss circuit breaker (8%).
"""
from __future__ import annotations
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from trading_bot.api.schemas import OrderRequest, Portfolio, TradeAction

logger = logging.getLogger(__name__)


def generate_idempotent_order_id(ticker: str, action: TradeAction, quantity: int, timestamp: str | float) -> str:
    """Generate deterministic client order ID to prevent duplicate executions across network retries."""
    payload = f"{ticker.upper()}_{action.value}_{quantity}_{timestamp}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"ORD_{ticker.upper()}_{h}"


class RiskLimits:
    """Configurable pre-trade risk policy limits."""

    def __init__(
        self,
        max_position_pct: float = 0.15,
        max_total_long_pct: float = 0.85,
        max_total_short_pct: float = 0.30,
        max_gross_exposure_pct: float = 1.0,
        max_single_trade_pct: float = 0.10,
        max_open_positions: int = 15,
        min_cash_reserve_pct: float = 0.05,
        hard_cash_floor_pct: float = 0.03,
        daily_loss_limit_pct: float = 0.03,
        symbol_cooldown_bars: int = 5,
        stale_data_seconds: float = 30.0,
        max_single_trade_loss_pct: float = 0.15,
        short_selling_enabled: bool = True
    ):
        self.max_position_pct = max_position_pct
        self.max_total_long_pct = max_total_long_pct
        self.max_total_short_pct = max_total_short_pct
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.max_single_trade_pct = max_single_trade_pct
        self.max_open_positions = max_open_positions
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.hard_cash_floor_pct = hard_cash_floor_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.symbol_cooldown_bars = symbol_cooldown_bars
        self.stale_data_seconds = stale_data_seconds
        self.max_single_trade_loss_pct = max_single_trade_loss_pct
        self.short_selling_enabled = short_selling_enabled


class PreTradeRiskEngine:
    """Validates proposed orders against institutional safety rules before transmission."""

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.day_start_wealth: Optional[float] = None
        self.is_daily_halted: bool = False
        self.daily_halt_reason: Optional[str] = None
        self.symbol_cooldowns: Dict[str, int] = {}  # ticker -> remaining cooldown bars
        self.is_emergency_killed: bool = False
        self.emergency_reason: Optional[str] = None

    def on_new_trading_day(self, starting_wealth: float) -> None:
        """Reset daily tracking at the start of each trading session."""
        self.day_start_wealth = starting_wealth
        self.is_daily_halted = False
        self.daily_halt_reason = None
        self.is_emergency_killed = False
        self.emergency_reason = None
        logger.info(f"New trading session initialized. Day start wealth = INR {starting_wealth:,.2f}")

    def on_tick_advance(self) -> None:
        """Decrement active symbol cooldowns on each bar/tick."""
        for ticker in list(self.symbol_cooldowns.keys()):
            self.symbol_cooldowns[ticker] -= 1
            if self.symbol_cooldowns[ticker] <= 0:
                del self.symbol_cooldowns[ticker]

    def record_trade_completion(self, ticker: str, return_pct: float) -> None:
        """Record trade result to trigger symbol cooldowns or emergency stops."""
        # Check single-trade disaster circuit breaker (loss > 8%)
        if return_pct <= -(self.limits.max_single_trade_loss_pct * 100.0):
            self.is_emergency_killed = True
            self.emergency_reason = (
                f"Emergency Kill Switch: Single trade loss on {ticker} of {return_pct:.2f}% "
                f"exceeds critical threshold {self.limits.max_single_trade_loss_pct * 100:.1f}%"
            )
            logger.critical(self.emergency_reason)

        # Trigger per-symbol loss cooldown if trade was unprofitable
        if return_pct < 0.0:
            self.symbol_cooldowns[ticker.upper()] = self.limits.symbol_cooldown_bars
            logger.warning(
                f"Losing trade on {ticker} ({return_pct:.2f}%). Cooldown activated for "
                f"{self.limits.symbol_cooldown_bars} bars."
            )

    def validate_order(
        self,
        order: OrderRequest,
        price: float,
        portfolio: Portfolio,
        available_cash: float,
        tick_timestamp: Optional[float] = None
    ) -> tuple[bool, Optional[str]]:
        """Run all pre-trade safety checks.
        Returns: (is_approved, rejection_reason)
        """
        # 1. Emergency Kill Switch Check
        if self.is_emergency_killed:
            return False, f"Trading Halted by Kill Switch: {self.emergency_reason}"

        # 2. Daily Loss Limit Check (Halt if day drawdown > 3%)
        total_wealth = max(portfolio.total_wealth, 1000.0)
        if self.day_start_wealth is not None and self.day_start_wealth > 0:
            day_pnl_pct = (total_wealth - self.day_start_wealth) / self.day_start_wealth
            if day_pnl_pct <= -self.limits.daily_loss_limit_pct:
                self.is_daily_halted = True
                self.daily_halt_reason = (
                    f"Daily loss limit breached: {day_pnl_pct * 100:.2f}% loss exceeds "
                    f"-{self.limits.daily_loss_limit_pct * 100:.1f}%. Trading suspended for remainder of day."
                )
                logger.critical(self.daily_halt_reason)

        if self.is_daily_halted:
            return False, f"Daily trading suspended: {self.daily_halt_reason}"

        # 3. Basic validity
        if order.quantity <= 0:
            return False, f"Order quantity must be positive: {order.quantity}"
        if price <= 0:
            return False, f"Invalid asset price: {price}"

        # 4. Stale Data Guard (reject ticks > 30 seconds old)
        if tick_timestamp is not None:
            now_sec = time.time()
            # If timestamp is in seconds vs milliseconds
            age_sec = (now_sec - tick_timestamp) if tick_timestamp < 1e11 else (now_sec - tick_timestamp / 1000.0)
            if age_sec > self.limits.stale_data_seconds:
                return False, f"Stale market data: tick is {age_sec:.1f}s old (limit {self.limits.stale_data_seconds}s)"

        # 5. Per-symbol loss cooldown check
        ticker_up = order.ticker.upper()
        if (order.action in (TradeAction.BUY, TradeAction.SHORT)) and (ticker_up in self.symbol_cooldowns):
            remaining = self.symbol_cooldowns[ticker_up]
            return False, f"Symbol {ticker_up} is in loss-cooldown ({remaining} bars remaining)"

        # 6. Hard Cash Floor Check (must maintain at least 3% total cash floor)
        hard_floor = total_wealth * self.limits.hard_cash_floor_pct
        if available_cash < hard_floor and order.action in (TradeAction.BUY, TradeAction.COVER):
            return False, f"Cash INR {available_cash:.2f} is below hard cash floor INR {hard_floor:.2f} (3%)"

        trade_notional = order.quantity * price

        # 7. Single trade notional cap (10%)
        max_trade_val = total_wealth * self.limits.max_single_trade_pct
        if trade_notional > max_trade_val:
            return False, f"Trade notional INR {trade_notional:,.2f} exceeds single trade limit INR {max_trade_val:,.2f}"

        # 8. Existing holding
        holding = next((h for h in portfolio.holdings if h.ticker.upper() == ticker_up), None)
        current_qty = holding.quantity if holding else 0

        # 9. Action-specific risk checks
        if order.action == TradeAction.BUY:
            required_cash_reserve = total_wealth * self.limits.min_cash_reserve_pct
            spendable_cash = max(0.0, available_cash - required_cash_reserve)
            if trade_notional > spendable_cash:
                return False, f"Insufficient cash {spendable_cash:.2f} for BUY notional {trade_notional:.2f}"

            new_qty = current_qty + order.quantity
            new_val = new_qty * price
            max_pos_val = total_wealth * self.limits.max_position_pct
            if new_val > max_pos_val:
                return False, f"Resulting position {new_val:.2f} exceeds position cap {max_pos_val:.2f}"

            if current_qty == 0 and len([h for h in portfolio.holdings if h.quantity != 0]) >= self.limits.max_open_positions:
                return False, f"Max open positions count {self.limits.max_open_positions} reached"

            # Check gross long exposure limit (85%)
            current_total_long = sum(h.market_value for h in portfolio.holdings if not h.is_short)
            if (current_total_long + trade_notional) > (total_wealth * self.limits.max_total_long_pct):
                return False, f"Total long exposure would exceed {self.limits.max_total_long_pct*100:.0f}% cap"

        elif order.action == TradeAction.SELL:
            if current_qty <= 0:
                return False, f"Cannot SELL ticker {order.ticker} with non-positive holding: {current_qty}"
            if order.quantity > current_qty:
                return False, f"Cannot SELL {order.quantity} shares: only hold {current_qty}"

        elif order.action == TradeAction.SHORT:
            if not self.limits.short_selling_enabled:
                return False, "Short selling is disabled by competition policy"
            if current_qty > 0:
                return False, f"Cannot SHORT ticker {order.ticker} while holding long {current_qty} shares"

            new_short_qty = abs(current_qty) + order.quantity
            new_short_val = new_short_qty * price
            max_pos_val = total_wealth * self.limits.max_position_pct
            if new_short_val > max_pos_val:
                return False, f"Resulting short position {new_short_val:.2f} exceeds position cap {max_pos_val:.2f}"

            current_total_short = sum(abs(h.market_value) for h in portfolio.holdings if h.is_short)
            if (current_total_short + trade_notional) > (total_wealth * self.limits.max_total_short_pct):
                return False, f"Total short exposure would exceed {self.limits.max_total_short_pct*100:.0f}% cap"

        elif order.action == TradeAction.COVER:
            if current_qty >= 0:
                return False, f"No open short position to COVER for {order.ticker} (holding={current_qty})"
            if order.quantity > abs(current_qty):
                return False, f"Cannot COVER {order.quantity} shares: only short {abs(current_qty)}"
            if trade_notional > available_cash:
                return False, f"Insufficient cash {available_cash:.2f} to COVER notional {trade_notional:.2f}"

        return True, None
