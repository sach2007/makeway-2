"""Quantitative performance metrics calculation."""
from __future__ import annotations
import math
import numpy as np


class PerformanceMetrics:
    """Calculates risk-adjusted return and trade quality metrics."""

    @staticmethod
    def calculate_sharpe(returns: list[float], risk_free_rate: float = 0.0) -> float:
        """Annualized or sample Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        std = np.std(arr, ddof=1)
        if std <= 1e-8:
            return 0.0
        mean = np.mean(arr) - risk_free_rate
        return float((mean / std) * math.sqrt(252))

    @staticmethod
    def calculate_max_drawdown(wealth_history: list[float]) -> tuple[float, float]:
        """Returns (max_drawdown_amount, max_drawdown_pct)."""
        if not wealth_history:
            return 0.0, 0.0
        peak = wealth_history[0]
        max_dd_amt = 0.0
        max_dd_pct = 0.0

        for w in wealth_history:
            if w > peak:
                peak = w
            dd_amt = peak - w
            dd_pct = (dd_amt / peak) if peak > 0 else 0.0
            if dd_amt > max_dd_amt:
                max_dd_amt = dd_amt
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        return max_dd_amt, max_dd_pct

    @staticmethod
    def calculate_trade_metrics(realized_pnls: list[float]) -> dict[str, float]:
        """Calculates win rate, profit factor, average win, average loss."""
        if not realized_pnls:
            return {
                "total_closed_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        wins = [p for p in realized_pnls if p > 0]
        losses = [p for p in realized_pnls if p < 0]

        win_count = len(wins)
        loss_count = len(losses)
        total_closed = len(realized_pnls)

        win_rate = (win_count / total_closed * 100.0) if total_closed > 0 else 0.0
        sum_wins = sum(wins)
        sum_losses = abs(sum(losses))

        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else (99.0 if sum_wins > 0 else 0.0)
        avg_win = (sum_wins / win_count) if win_count > 0 else 0.0
        avg_loss = (sum_losses / loss_count) if loss_count > 0 else 0.0

        return {
            "total_closed_trades": total_closed,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
        }
