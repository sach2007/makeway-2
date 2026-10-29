"""Comprehensive quantitative risk and performance metrics engine."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd


class QuantMetrics:
    """Calculates institutional performance and risk metrics for backtest evaluation."""

    @staticmethod
    def calculate_cagr(start_wealth: float, end_wealth: float, total_days: float) -> float:
        """Compound Annual Growth Rate assuming 252 trading days/year."""
        if start_wealth <= 0 or end_wealth <= 0 or total_days <= 0:
            return 0.0
        years = total_days / 252.0
        if years <= 0:
            return 0.0
        return float((end_wealth / start_wealth) ** (1.0 / years) - 1.0) * 100.0

    @staticmethod
    def calculate_sharpe(returns: list[float] | np.ndarray, rf_annual: float = 0.06, periods_per_year: int = 252) -> float:
        """Annualized Sharpe Ratio with risk-free rate deduction."""
        rets = np.array(returns, dtype=float)
        if len(rets) < 2:
            return 0.0
        rf_daily = rf_annual / periods_per_year
        excess_rets = rets - rf_daily
        std = float(np.std(excess_rets, ddof=1))
        if std < 1e-9:
            return 0.0
        mean = float(np.mean(excess_rets))
        return round(float((mean / std) * math.sqrt(periods_per_year)), 2)

    @staticmethod
    def calculate_sortino(returns: list[float] | np.ndarray, rf_annual: float = 0.06, periods_per_year: int = 252) -> float:
        """Annualized Sortino Ratio penalizing only downside volatility."""
        rets = np.array(returns, dtype=float)
        if len(rets) < 2:
            return 0.0
        rf_daily = rf_annual / periods_per_year
        excess_rets = rets - rf_daily
        downside = excess_rets[excess_rets < 0]
        if len(downside) < 1:
            return 10.0 if np.mean(excess_rets) > 0 else 0.0
        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        if downside_std < 1e-9:
            return 0.0
        mean = float(np.mean(excess_rets))
        return round(float((mean / downside_std) * math.sqrt(periods_per_year)), 2)

    @staticmethod
    def calculate_calmar(cagr_pct: float, max_drawdown_pct: float) -> float:
        """Calmar Ratio = CAGR% / MaxDrawdown%."""
        if max_drawdown_pct <= 0:
            return 0.0
        return round(float(cagr_pct / max_drawdown_pct), 2)

    @staticmethod
    def calculate_max_drawdown(wealth_history: list[float] | np.ndarray) -> tuple[float, float]:
        """Returns: (max_drawdown_dollar, max_drawdown_percent)."""
        arr = np.array(wealth_history, dtype=float)
        if len(arr) < 2:
            return 0.0, 0.0
        peak = np.maximum.accumulate(arr)
        drawdowns = (peak - arr) / np.maximum(peak, 1e-10)
        max_dd_pct = float(np.max(drawdowns))
        max_dd_amt = float(np.max(peak - arr))
        return round(max_dd_amt, 2), round(max_dd_pct * 100.0, 2)

    @staticmethod
    def calculate_trade_metrics(realized_pnls: list[float]) -> dict[str, float]:
        """Calculates win rate, profit factor, average win/loss."""
        if not realized_pnls:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0
            }
        wins = [p for p in realized_pnls if p > 0]
        losses = [abs(p) for p in realized_pnls if p < 0]

        win_rate = (len(wins) / len(realized_pnls)) * 100.0
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        return {
            "total_trades": len(realized_pnls),
            "win_rate_pct": round(float(win_rate), 2),
            "profit_factor": round(float(profit_factor), 2),
            "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
            "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
        }

    @staticmethod
    def calculate_turnover(total_traded_value: float, avg_wealth: float) -> float:
        """Portfolio turnover ratio."""
        if avg_wealth <= 0:
            return 0.0
        return round(float(total_traded_value / avg_wealth), 2)
