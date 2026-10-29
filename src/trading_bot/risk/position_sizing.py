"""Institutional Volatility-Adjusted & Fractional Kelly Position Sizing.

Implements:
1. Fractional Kelly sizing (0.25x) based on calibrated win probabilities and payoff ratios.
2. Volatility targeting (15% annualized portfolio volatility target, scaling inversely with asset ATR/vol).
3. Correlation-aware portfolio caps (rolling 30d correlation > 0.7 limits combined cluster exposure to 1.5x single cap).
4. Hard veto risk constraints: 15% single position, 10% per-order, 5% cash buffer.
"""
from __future__ import annotations
import math
import logging
from typing import Dict, Optional, Tuple
import numpy as np

from trading_bot.api.schemas import Portfolio, TradeAction
from trading_bot.risk.limits import RiskLimits

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates risk-managed integer trade quantities using Kelly and volatility parity."""

    def __init__(
        self,
        limits: RiskLimits,
        target_annualized_vol: float = 0.15,
        kelly_fraction: float = 0.25,
        default_payoff_ratio: float = 1.50,
        max_notional_cap: float = 250_000.0,
        base_trade_pct: float = 0.06
    ):
        self.limits = limits
        self.target_annualized_vol = target_annualized_vol
        self.daily_target_vol = target_annualized_vol / np.sqrt(252)
        self.kelly_fraction = kelly_fraction
        self.default_payoff_ratio = default_payoff_ratio
        self.max_notional_cap = max_notional_cap
        self.base_trade_pct = base_trade_pct

    def calculate_fractional_kelly(
        self,
        win_probability: float,
        payoff_ratio: Optional[float] = None
    ) -> float:
        """Compute fractional Kelly allocation fraction f* = (p*b - q)/b * kelly_fraction."""
        p = max(0.01, min(0.99, win_probability))
        q = 1.0 - p
        b = payoff_ratio if (payoff_ratio and payoff_ratio > 0.1) else self.default_payoff_ratio

        full_kelly = (p * b - q) / b
        if full_kelly <= 0.0:
            return 0.0

        fractional_kelly = full_kelly * self.kelly_fraction
        return float(fractional_kelly)

    def calculate_volatility_target_weight(self, asset_daily_vol: float) -> float:
        """Scale allocation weight inversely to asset realized volatility or ATR%."""
        vol = max(1e-4, asset_daily_vol)
        weight = self.daily_target_vol / vol
        return float(np.clip(weight, 0.02, 1.0))

    def calculate_quantity(
        self,
        action: TradeAction,
        price: float,
        conviction: float,
        portfolio: Portfolio,
        available_cash: float,
        current_holding_qty: int = 0,
        volatility: float = 0.02,
        win_probability: Optional[float] = None,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        ticker: str = ""
    ) -> int:
        """Calculate optimal order size adhering to Kelly, Vol-Targeting, and Hard Vetoes."""
        if price <= 0:
            return 0

        # SELL: close out long position
        if action == TradeAction.SELL:
            return max(0, current_holding_qty)

        # COVER: buy back short position
        if action == TradeAction.COVER:
            return abs(current_holding_qty) if current_holding_qty < 0 else 0

        total_wealth = max(portfolio.total_wealth, 1000.0)

        # 1. Determine Win Probability
        if win_probability is not None:
            p_win = win_probability
        else:
            # Map conviction [-1.0, 1.0] to win probability [0.50, 0.90]
            abs_conv = min(1.0, abs(conviction))
            p_win = 0.50 + (abs_conv * 0.40)

        # 2. Fractional Kelly Sizing (0.25x)
        f_kelly = self.calculate_fractional_kelly(p_win, self.default_payoff_ratio)
        if f_kelly <= 0.0:
            return 0

        # 3. Volatility Targeting Weight
        w_vol = self.calculate_volatility_target_weight(volatility)

        # Base target fraction: Kelly constrained by vol target and per-order limit (10%)
        target_pct = min(f_kelly, w_vol, self.limits.max_single_trade_pct)

        # 4. Correlation-Aware Limits:
        # If correlated with existing positions (>0.70), limit combined cluster allocation to 1.5x single position cap
        if correlation_matrix and ticker in correlation_matrix:
            correlated_exposure = 0.0
            for h in portfolio.holdings:
                if h.ticker != ticker and abs(h.quantity) > 0:
                    corr = correlation_matrix[ticker].get(h.ticker, 0.0)
                    if corr > 0.70:
                        correlated_exposure += abs(h.quantity) * h.current_price

            max_cluster_val = total_wealth * (self.limits.max_position_pct * 1.50)
            remaining_cluster_budget = max(0.0, max_cluster_val - correlated_exposure)
            target_notional = min(total_wealth * target_pct, remaining_cluster_budget)
        else:
            target_notional = total_wealth * target_pct

        # Institutional liquidity/compounding cap
        target_notional = min(target_notional, self.max_notional_cap)

        # 5. Hard Risk Limits Vetoes
        # Cash reserve veto (must preserve 5% cash buffer)
        if action == TradeAction.BUY:
            min_reserve = total_wealth * self.limits.min_cash_reserve_pct
            spendable_cash = max(0.0, available_cash - min_reserve)
            target_notional = min(target_notional, spendable_cash)

        # Single position cap veto (15%)
        existing_val = abs(current_holding_qty) * price
        max_pos_val = total_wealth * self.limits.max_position_pct
        room_for_growth = max(0.0, max_pos_val - existing_val)
        target_notional = min(target_notional, room_for_growth)

        qty = math.floor(target_notional / price)
        return max(0, qty)
