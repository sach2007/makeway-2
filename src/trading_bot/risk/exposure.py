"""Regime-Aware Dynamic ATR Exits and Portfolio Exposure Manager.

Features:
1. Dynamic ATR-based Stop Loss: entry - k1 * ATR (scales with asset volatility).
2. Dynamic ATR-based Trailing Stop: highest - k2 * ATR (ratchets up, never down).
3. Dynamic ATR Profit Target: entry + k3 * ATR.
4. Time-based Bar Exit: closes dead positions after N bars if return < 0.5 * ATR.
5. Regime-dependent multipliers (tighter stops in High-Vol, wider in Low-Vol Trending).
6. 25% Emergency portfolio drawdown circuit breaker.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from trading_bot.api.schemas import Portfolio, TradeAction

logger = logging.getLogger(__name__)


@dataclass
class PositionTrackingState:
    ticker: str
    entry_price: float
    highest_price: float
    lowest_price: float
    bars_held: int = 0
    peak_gain_pct: float = 0.0


class ExposureManager:
    """Monitors drawdowns, trailing stops, and regime-dependent ATR exits."""

    def __init__(
        self,
        emergency_halt_drawdown_pct: float = 0.25,
        per_position_stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        trailing_stop_pct: float = 0.05,
        trailing_activation_pct: float = 0.10,
        atr_stop_multiplier: float = 2.0,
        atr_trail_multiplier: float = 1.5,
        atr_profit_multiplier: float = 3.5,
        max_bars_held: int = 20
    ):
        self.emergency_halt_drawdown_pct = emergency_halt_drawdown_pct
        self.per_position_stop_loss_pct = per_position_stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_trail_multiplier = atr_trail_multiplier
        self.atr_profit_multiplier = atr_profit_multiplier
        self.max_bars_held = max_bars_held

        self.peak_wealth: float = 0.0
        self.current_drawdown_pct: float = 0.0
        self.is_halted: bool = False
        self.halt_reason: Optional[str] = None

        # Tracking state per position
        self._positions: dict[str, PositionTrackingState] = {}
        self._peak_gains: dict[str, float] = {}

    def update_portfolio(self, portfolio: Portfolio) -> None:
        """Update metrics and check circuit breakers against latest portfolio snapshot."""
        wealth = portfolio.total_wealth
        if wealth > self.peak_wealth:
            self.peak_wealth = wealth

        if self.peak_wealth > 0:
            dd = (self.peak_wealth - wealth) / self.peak_wealth
            self.current_drawdown_pct = max(0.0, dd)

            if self.current_drawdown_pct >= self.emergency_halt_drawdown_pct and not self.is_halted:
                self.is_halted = True
                self.halt_reason = (
                    f"Emergency circuit breaker triggered: drawdown {self.current_drawdown_pct * 100:.1f}% "
                    f"exceeds limit {self.emergency_halt_drawdown_pct * 100:.1f}%"
                )
                logger.critical(self.halt_reason)

    def check_exits(
        self,
        portfolio: Portfolio,
        market_context: Optional[Dict[str, Dict]] = None
    ) -> list[tuple[str, TradeAction, int, str]]:
        """Identify open positions that hit ATR-based stops, trailing stops, profit targets, or time stops.

        market_context: optional dict mapping ticker -> {
            "atr": float,
            "regime": str,
            "current_price": float
        }
        Returns list of (ticker, exit_action, quantity, reason).
        """
        exits = []
        active_tickers = {h.ticker for h in portfolio.holdings}

        # Purge tracking for closed positions
        for t in list(self._positions.keys()):
            if t not in active_tickers:
                del self._positions[t]
        for t in list(self._peak_gains.keys()):
            if t not in active_tickers:
                del self._peak_gains[t]

        for holding in portfolio.holdings:
            ticker = holding.ticker
            qty = abs(holding.quantity)
            if qty == 0:
                continue

            current_p = holding.current_price if holding.current_price > 0 else holding.average_buy_price
            avg_p = holding.average_buy_price if holding.average_buy_price > 0 else current_p
            is_short = holding.is_short
            exit_action = TradeAction.COVER if is_short else TradeAction.SELL

            if avg_p > 0:
                unrealized_pct = ((current_p - avg_p) / avg_p) if not is_short else ((avg_p - current_p) / avg_p)
            else:
                unrealized_pct = holding.unrealized_pnl_pct / 100.0 if abs(holding.unrealized_pnl_pct) > 1.0 else holding.unrealized_pnl_pct

            # Update or initialize tracking state
            if ticker not in self._positions:
                self._positions[ticker] = PositionTrackingState(
                    ticker=ticker,
                    entry_price=avg_p,
                    highest_price=max(avg_p, current_p),
                    lowest_price=min(avg_p, current_p),
                    bars_held=1,
                    peak_gain_pct=max(0.0, unrealized_pct)
                )
            else:
                state = self._positions[ticker]
                state.bars_held += 1
                state.highest_price = max(state.highest_price, current_p)
                state.lowest_price = min(state.lowest_price, current_p)
                state.peak_gain_pct = max(state.peak_gain_pct, unrealized_pct)

            state = self._positions[ticker]
            self._peak_gains[ticker] = state.peak_gain_pct

            # ----------------------------------------------------
            # Advanced Regime & ATR Dynamic Exit Evaluation
            # ----------------------------------------------------
            ctx = (market_context or {}).get(ticker, {})
            atr = ctx.get("atr")
            regime = ctx.get("regime", "UNKNOWN")

            # Adjust multipliers based on detected regime
            k1 = self.atr_stop_multiplier
            k2 = self.atr_trail_multiplier
            k3 = self.atr_profit_multiplier
            max_bars = self.max_bars_held

            if "MEAN_REVERT" in regime or "CRISIS" in regime or regime == "MarketRegimeState.HIGH_VOL_MEAN_REVERT":
                # High volatility: tighter stops, quicker profit taking, shorter holding horizon
                k1 *= 0.75
                k2 *= 0.75
                k3 *= 0.70
                max_bars = max(5, int(max_bars * 0.50))
            elif "TREND" in regime or regime == "MarketRegimeState.LOW_VOL_TREND":
                # Low volatility trending: give trades breathing room to run
                k1 *= 1.25
                k2 *= 1.25
                k3 *= 1.30
                max_bars = int(max_bars * 1.50)

            # 1. Dynamic ATR Stop Loss (Adaptive volatility stop)
            if atr and atr > 0:
                if not is_short:
                    stop_level = state.entry_price - (k1 * atr)
                    if current_p <= stop_level:
                        exits.append((ticker, exit_action, qty, f"ATR_STOP_LOSS (Price {current_p:.2f} <= {stop_level:.2f})"))
                        continue
                else:
                    stop_level = state.entry_price + (k1 * atr)
                    if current_p >= stop_level:
                        exits.append((ticker, exit_action, qty, f"ATR_STOP_LOSS (Price {current_p:.2f} >= {stop_level:.2f})"))
                        continue

            # 2. Unconditional Hard Stop Loss Safety Net (-5% or configured per_position_stop_loss_pct)
            if unrealized_pct <= -self.per_position_stop_loss_pct:
                exits.append((ticker, exit_action, qty, f"STOP_LOSS ({unrealized_pct*100:.1f}%)"))
                continue

            # 3. Unconditional Hard Take Profit (+15% or configured take_profit_pct)
            if unrealized_pct >= self.take_profit_pct:
                exits.append((ticker, exit_action, qty, f"TAKE_PROFIT (+{unrealized_pct*100:.1f}%)"))
                continue

            # 4. Dynamic ATR Trailing Stop and Time Exits
            if atr and atr > 0:

                # Dynamic ATR Trailing Stop
                if not is_short:
                    trail_level = state.highest_price - (k2 * atr)
                    if current_p <= trail_level and state.highest_price > (state.entry_price + 0.5 * atr):
                        exits.append((ticker, exit_action, qty, f"ATR_TRAILING_STOP (Peak {state.highest_price:.2f} -> {current_p:.2f})"))
                        continue
                else:
                    trail_level = state.lowest_price + (k2 * atr)
                    if current_p >= trail_level and state.lowest_price < (state.entry_price - 0.5 * atr):
                        exits.append((ticker, exit_action, qty, f"ATR_TRAILING_STOP (Trough {state.lowest_price:.2f} -> {current_p:.2f})"))
                        continue

                # Time-based Dead Trade Exit
                if state.bars_held >= max_bars:
                    price_change = abs(current_p - state.entry_price)
                    if price_change < (0.5 * atr):
                        exits.append((ticker, exit_action, qty, f"TIME_EXIT ({state.bars_held} bars held with flat momentum)"))
                        continue

            # 4. Trailing Stop from Peak Unrealized Gain
            if state.peak_gain_pct >= self.trailing_activation_pct and (state.peak_gain_pct - unrealized_pct) >= self.trailing_stop_pct:
                exits.append((ticker, exit_action, qty, f"TRAILING_STOP (Peak +{state.peak_gain_pct*100:.1f}% -> {unrealized_pct*100:.1f}%)"))
                continue

        return exits

    def check_stop_losses(self, portfolio: Portfolio) -> list[tuple[str, TradeAction, int]]:
        """Backward-compatible wrapper returning (ticker, exit_action, quantity)."""
        exits = self.check_exits(portfolio)
        return [(ticker, action, qty) for (ticker, action, qty, reason) in exits]
