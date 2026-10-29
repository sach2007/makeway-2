"""Backtesting and strategy benchmarking engine for MANTHAN 2026 with full trade ledger generation."""
from __future__ import annotations
import math
import logging
from typing import Any, Optional
import pandas as pd

from trading_bot.api.schemas import TradeAction
from trading_bot.features.indicators import FeatureExtractor
from trading_bot.portfolio.performance import PerformanceMetrics
from trading_bot.strategy.ensemble import StrategyEnsemble
from trading_bot.strategy.signals import (
    DipBuyingStrategy, MeanReversionStrategy, MomentumStrategy, TrendFollowingStrategy
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Event-driven historical backtester simulating realistic execution, slippage, DP charges, and trade ledger generation."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        fee_bps: float = 5.0,        # retained for backward-compatible slippage sensitivity only
        slippage_bps: float = 5.0,   # 5 bps slippage
        dp_charge: float = 15.34,    # DP flat charge per sell/close
        stop_loss_pct: float = 0.05, # 5% stop loss
        take_profit_pct: float = 0.15 # 15% take profit
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_bps / 10000.0  # legacy attribute; contest brokerage is flat below
        self.brokerage_per_order = 20.0
        self.slippage_rate = slippage_bps / 10000.0
        self.dp_charge = dp_charge
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.all_trade_records: list[dict[str, Any]] = []

    def run_single_symbol(
        self,
        ticker: str,
        df: pd.DataFrame,
        strategy_mode: str = "ensemble"
    ) -> dict[str, Any]:
        """Run backtest on a single symbol's historical OHLCV dataframe."""
        if df.empty or len(df) < 30:
            return {"error": "Insufficient history (< 30 bars)"}

        cash = self.initial_capital
        holding_qty = 0
        is_short = False
        avg_entry_price = 0.0
        entry_date = ""
        entry_idx = 0

        wealth_history = []
        returns_history = []
        realized_pnls = []
        trade_count = 0
        total_fees = 0.0
        trade_id_seq = 1

        ensemble = StrategyEnsemble(signal_threshold=0.22, no_trade_band=0.08, cooldown_ticks=2)

        # Baseline: Buy & Hold
        if strategy_mode == "baseline":
            first_p = df["close"].iloc[20]
            shares = math.floor(max(0.0, cash * 0.95 - self.brokerage_per_order) / (first_p * (1.0 + self.slippage_rate)))
            cost = shares * first_p * (1.0 + self.slippage_rate) + self.brokerage_per_order
            cash -= cost
            holding_qty = shares
            for i in range(20, len(df)):
                price = df["close"].iloc[i]
                w = cash + (holding_qty * price)
                wealth_history.append(w)
                if len(wealth_history) > 1:
                    returns_history.append((w - wealth_history[-2]) / wealth_history[-2])
            last_p = df["close"].iloc[-1]
            final_w = cash + (holding_qty * last_p)
            dd_amt, dd_pct = PerformanceMetrics.calculate_max_drawdown(wealth_history)
            sharpe = PerformanceMetrics.calculate_sharpe(returns_history)
            return {
                "strategy": "Baseline (Buy & Hold)",
                "initial_wealth": self.initial_capital,
                "final_wealth": round(final_w, 2),
                "total_return_pct": round((final_w - self.initial_capital) / self.initial_capital * 100.0, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown_pct": round(dd_pct * 100.0, 2),
                "total_trades": 1,
                "win_rate_pct": 100.0 if final_w > self.initial_capital else 0.0,
            }

        # Step through bars chronologically
        for i in range(25, len(df)):
            sub_df = df.iloc[:i]
            current_bar = df.iloc[i]
            price = current_bar["close"]
            date_str = str(current_bar.get("date", f"bar_{i}"))[:10]

            features = FeatureExtractor.compute_features(sub_df)
            ensemble.on_tick_advance()

            # 1. Check Take-Profit & Stop-Loss exits
            exit_triggered = False
            exit_reason = ""
            if holding_qty > 0 and not is_short:
                gain_pct = (price - avg_entry_price) / avg_entry_price
                if gain_pct >= self.take_profit_pct:
                    exit_triggered = True
                    exit_reason = f"TAKE_PROFIT (+{gain_pct*100:.1f}%)"
                elif gain_pct <= -self.stop_loss_pct:
                    exit_triggered = True
                    exit_reason = f"STOP_LOSS ({gain_pct*100:.1f}%)"

            elif holding_qty > 0 and is_short:
                gain_pct = (avg_entry_price - price) / avg_entry_price
                if gain_pct >= self.take_profit_pct:
                    exit_triggered = True
                    exit_reason = f"TAKE_PROFIT (+{gain_pct*100:.1f}%)"
                elif gain_pct <= -self.stop_loss_pct:
                    exit_triggered = True
                    exit_reason = f"STOP_LOSS ({gain_pct*100:.1f}%)"

            if exit_triggered:
                # Close position
                exec_price = price * (1.0 - self.slippage_rate) if not is_short else price * (1.0 + self.slippage_rate)
                fee = self.brokerage_per_order
                gross_pnl = ((exec_price - avg_entry_price) * holding_qty) if not is_short else ((avg_entry_price - exec_price) * holding_qty)
                dp_exit = 0.0 if is_short else self.dp_charge
                net_pnl = gross_pnl - fee - dp_exit
                total_fees += fee + dp_exit
                # Long sale receives proceeds; short close realizes P&L on top of
                # cash already retained after the short-entry fee.
                cash += (holding_qty * exec_price) - fee - dp_exit if not is_short else net_pnl
                realized_pnls.append(net_pnl)

                # Record trade in ledger
                self.all_trade_records.append({
                    "trade_id": f"TRD_{ticker}_{trade_id_seq:04d}",
                    "ticker": ticker,
                    "strategy": strategy_mode,
                    "action": "SELL" if not is_short else "COVER",
                    "entry_date": entry_date,
                    "exit_date": date_str,
                    "holding_days": i - entry_idx,
                    "shares": holding_qty,
                    "entry_price": round(avg_entry_price, 2),
                    "exit_price": round(exec_price, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "brokerage_fees": round(fee, 2),
                    "dp_charges": dp_exit,
                    "net_pnl": round(net_pnl, 2),
                    "return_pct": round(net_pnl / (avg_entry_price * holding_qty) * 100.0, 2),
                    "exit_reason": exit_reason
                })
                trade_id_seq += 1
                trade_count += 1
                holding_qty = 0
                is_short = False
                ensemble.record_trade(ticker)

            # 2. Strategy evaluation
            action = None
            conviction = 0.0

            if strategy_mode == "momentum":
                sig = MomentumStrategy.evaluate(ticker, features)
                if sig.score > 0.25 and holding_qty == 0:
                    action = TradeAction.BUY
                    conviction = abs(sig.score)
                elif sig.score < -0.25 and holding_qty > 0 and not is_short:
                    action = TradeAction.SELL
                    conviction = abs(sig.score)

            elif strategy_mode == "trend":
                sig = TrendFollowingStrategy.evaluate(ticker, features)
                if sig.score > 0.25 and holding_qty == 0:
                    action = TradeAction.BUY
                    conviction = abs(sig.score)
                elif sig.score < -0.25 and holding_qty > 0 and not is_short:
                    action = TradeAction.SELL
                    conviction = abs(sig.score)

            elif strategy_mode == "mean_reversion":
                sig = MeanReversionStrategy.evaluate(ticker, features)
                if sig.score > 0.25 and holding_qty == 0:
                    action = TradeAction.BUY
                    conviction = abs(sig.score)
                elif sig.score < -0.25 and holding_qty > 0 and not is_short:
                    action = TradeAction.SELL
                    conviction = abs(sig.score)

            else:  # Ensemble + Dip Buying
                dec = ensemble.evaluate(ticker, features, news_sentiment=0.0, current_holding_qty=holding_qty)
                if dec.should_trade:
                    action = dec.action
                    conviction = dec.conviction

            # 3. Execute simulated entry
            if action == TradeAction.BUY and holding_qty == 0 and cash > (price * 10):
                exec_price = price * (1.0 + self.slippage_rate)
                spend = min(cash * 0.15, cash * 0.08 * max(1.0, conviction * 2.0))
                qty = math.floor(spend / exec_price)
                if qty > 0:
                    fee = self.brokerage_per_order
                    cash -= (qty * exec_price) + fee
                    total_fees += fee
                    avg_entry_price = exec_price
                    holding_qty = qty
                    is_short = False
                    entry_date = date_str
                    entry_idx = i
                    trade_count += 1
                    ensemble.record_trade(ticker)

            elif action == TradeAction.SHORT and holding_qty == 0 and cash > (price * 10):
                exec_price = price * (1.0 - self.slippage_rate)
                spend = min(cash * 0.12, cash * 0.06 * max(1.0, conviction * 2.0))
                qty = math.floor(spend / exec_price)
                if qty > 0:
                    fee = self.brokerage_per_order
                    cash -= fee
                    total_fees += fee
                    avg_entry_price = exec_price
                    holding_qty = qty
                    is_short = True
                    entry_date = date_str
                    entry_idx = i
                    trade_count += 1
                    ensemble.record_trade(ticker)

            # Mark to market wealth
            pos_val = (holding_qty * price) if not is_short else -(holding_qty * (price - avg_entry_price))
            w = cash + pos_val
            wealth_history.append(w)
            if len(wealth_history) > 1:
                returns_history.append((w - wealth_history[-2]) / wealth_history[-2])

        # Final liquidation at last close
        if holding_qty > 0:
            last_p = df["close"].iloc[-1]
            exec_price = last_p * (1.0 - self.slippage_rate) if not is_short else last_p * (1.0 + self.slippage_rate)
            fee = self.brokerage_per_order
            gross_pnl = ((exec_price - avg_entry_price) * holding_qty) if not is_short else ((avg_entry_price - exec_price) * holding_qty)
            dp_exit = 0.0 if is_short else self.dp_charge
            net_pnl = gross_pnl - fee - dp_exit
            cash += (holding_qty * exec_price) - fee - dp_exit if not is_short else net_pnl
            realized_pnls.append(net_pnl)
            total_fees += fee + dp_exit
            self.all_trade_records.append({
                "trade_id": f"TRD_{ticker}_{trade_id_seq:04d}",
                "ticker": ticker,
                "strategy": strategy_mode,
                "action": "SELL_FINAL" if not is_short else "COVER_FINAL",
                "entry_date": entry_date,
                "exit_date": str(df["date"].iloc[-1])[:10],
                "holding_days": len(df) - entry_idx,
                "shares": holding_qty,
                "entry_price": round(avg_entry_price, 2),
                "exit_price": round(exec_price, 2),
                "gross_pnl": round(gross_pnl, 2),
                "brokerage_fees": round(fee, 2),
                "dp_charges": dp_exit,
                "net_pnl": round(net_pnl, 2),
                "return_pct": round(net_pnl / (avg_entry_price * holding_qty) * 100.0, 2),
                "exit_reason": "SIMULATION_END"
            })
            holding_qty = 0

        final_w = cash
        total_ret = (final_w - self.initial_capital) / self.initial_capital * 100.0
        dd_amt, dd_pct = PerformanceMetrics.calculate_max_drawdown(wealth_history)
        sharpe = PerformanceMetrics.calculate_sharpe(returns_history)
        trade_metrics = PerformanceMetrics.calculate_trade_metrics(realized_pnls)

        return {
            "strategy": strategy_mode.capitalize(),
            "initial_wealth": self.initial_capital,
            "final_wealth": round(final_w, 2),
            "total_return_pct": round(total_ret, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(dd_pct * 100.0, 2),
            "total_trades": trade_count,
            "win_rate_pct": trade_metrics["win_rate_pct"],
            "profit_factor": trade_metrics["profit_factor"],
            "total_fees_paid": round(total_fees, 2)
        }
