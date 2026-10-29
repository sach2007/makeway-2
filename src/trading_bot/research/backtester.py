"""Institutional event-driven backtesting engine replaying ticks through the live trading bot pipeline."""
from __future__ import annotations
import math
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

from trading_bot.api.schemas import Portfolio, PortfolioPosition, TradeAction
from trading_bot.features.indicators import FeatureExtractor
from trading_bot.research.metrics import QuantMetrics
from trading_bot.research.purged_cv import PurgedWalkForwardCV
from trading_bot.risk.exposure import ExposureManager
from trading_bot.risk.limits import PreTradeRiskEngine, RiskLimits
from trading_bot.risk.position_sizing import PositionSizer
from trading_bot.strategy.ensemble import StrategyEnsemble
from trading_bot.strategy.regime import MarketRegime, RegimeDetector

logger = logging.getLogger(__name__)


class EventDrivenBacktester:
    """Event-driven historical backtester executing bars through identical logic as live trading.
    Enforces realistic transaction friction: brokerage, DP charges, STT, non-linear market impact slippage.
    Zero-cost results are strictly impossible by default.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        brokerage_per_order: float = 20.0,
        dp_charge_per_sell: float = 15.34,
        stt_rate: float = 0.00025,           # 2.5 bps Securities Transaction Tax
        base_half_spread_bps: float = 5.0,   # 5 bps minimum bid-ask spread
        market_impact_gamma: float = 0.10,    # Square-root impact coefficient
        latency_penalty_bps: float = 2.0,    # 2 bps execution latency slippage
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15
    ):
        # Strict assertion: zero-cost results are impossible by default
        if brokerage_per_order <= 0 or base_half_spread_bps <= 0:
            raise ValueError("Zero-cost backtesting is prohibited. Brokerage and spread must be positive.")

        self.initial_capital = initial_capital
        self.brokerage = brokerage_per_order
        self.dp_charge = dp_charge_per_sell
        self.stt_rate = stt_rate
        self.base_spread = base_half_spread_bps / 10000.0
        self.market_impact_gamma = market_impact_gamma
        self.latency_penalty = latency_penalty_bps / 10000.0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def compute_slippage(self, order_qty: int, daily_volume: float) -> float:
        """Compute non-linear market impact slippage: Spread + Gamma * sqrt(Qty / Volume) + Latency."""
        vol = max(daily_volume, 1000.0)
        participation = min(0.30, float(order_qty) / vol)
        impact = self.market_impact_gamma * math.sqrt(participation)
        total_slippage_rate = self.base_spread + impact + self.latency_penalty
        return total_slippage_rate

    def run(
        self,
        symbol_dfs: dict[str, pd.DataFrame],
        strategy_mode: str = "ensemble",
        ml_inference_fn: Optional[Any] = None
    ) -> dict[str, Any]:
        """Execute event-driven multi-symbol backtest across time-aligned historical bars."""
        # Align all dates across symbols
        all_dates = set()
        for df in symbol_dfs.values():
            if "date" in df.columns:
                all_dates.update(df["date"].astype(str).str[:10].tolist())
        sorted_dates = sorted(list(all_dates))

        if len(sorted_dates) < 30:
            return {"error": "Insufficient history dates (< 30 dates)"}

        # Portfolio State
        cash = self.initial_capital
        holdings: dict[str, dict[str, Any]] = {}  # ticker -> {qty, avg_price, is_short, peak_gain}
        trade_history: list[dict[str, Any]] = []
        wealth_history: list[float] = [cash]
        returns_history: list[float] = []
        regime_returns: dict[str, list[float]] = {r.value: [] for r in MarketRegime}
        gross_exposures: list[float] = []
        net_exposures: list[float] = []

        total_turnover_dollars = 0.0
        total_brokerage_paid = 0.0
        total_taxes_paid = 0.0
        total_dp_paid = 0.0

        # Institutional Risk & Strategy Setup
        risk_limits = RiskLimits(
            max_position_pct=0.15,
            max_total_long_pct=0.85,
            max_total_short_pct=0.30,
            max_gross_exposure_pct=1.0,
            max_single_trade_pct=0.10,
            min_cash_reserve_pct=0.05,
            short_selling_enabled=True
        )
        risk_engine = PreTradeRiskEngine(risk_limits)
        position_sizer = PositionSizer(risk_limits, max_notional_cap=150_000.0)
        exposure_mgr = ExposureManager(
            per_position_stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            atr_stop_multiplier=2.0,
            atr_trail_multiplier=1.5,
            atr_profit_multiplier=3.5
        )
        ensemble = StrategyEnsemble(signal_threshold=0.22, no_trade_band=0.08, cooldown_ticks=2)

        # Baseline mode shortcut
        if strategy_mode == "baseline":
            # Buy and hold equal weight across all symbols at bar 20
            first_date = sorted_dates[20]
            alloc_per_sym = (cash * 0.90) / len(symbol_dfs)
            for ticker, df in symbol_dfs.items():
                m_bar = df[df["date"].astype(str).str[:10] == first_date]
                if not m_bar.empty:
                    p = float(m_bar["close"].iloc[0])
                    v = float(m_bar.get("volume", pd.Series([100000])).iloc[0])
                    slip = self.compute_slippage(100, v)
                    exec_p = p * (1.0 + slip)
                    qty = math.floor(alloc_per_sym / exec_p)
                    if qty > 0:
                        cost = qty * exec_p
                        fee = self.brokerage + (cost * self.stt_rate)
                        cash -= (cost + fee)
                        total_brokerage_paid += self.brokerage
                        total_taxes_paid += (cost * self.stt_rate)
                        total_turnover_dollars += cost
                        holdings[ticker] = {"qty": qty, "avg_price": exec_p, "is_short": False, "peak_gain": 0.0}

            for d in sorted_dates[21:]:
                day_val = cash
                for ticker, h in holdings.items():
                    df = symbol_dfs[ticker]
                    m_bar = df[df["date"].astype(str).str[:10] == d]
                    p = float(m_bar["close"].iloc[0]) if not m_bar.empty else h["avg_price"]
                    day_val += h["qty"] * p
                wealth_history.append(day_val)
                if len(wealth_history) > 1:
                    returns_history.append((day_val - wealth_history[-2]) / wealth_history[-2])

            final_w = wealth_history[-1]
            dd_amt, dd_pct = QuantMetrics.calculate_max_drawdown(wealth_history)
            cagr = QuantMetrics.calculate_cagr(self.initial_capital, final_w, len(sorted_dates))
            sharpe = QuantMetrics.calculate_sharpe(returns_history)
            sortino = QuantMetrics.calculate_sortino(returns_history)
            calmar = QuantMetrics.calculate_calmar(cagr, dd_pct)
            return {
                "strategy": "Baseline (Buy & Hold)",
                "initial_wealth": self.initial_capital,
                "final_wealth": round(final_w, 2),
                "net_return_pct": round((final_w - self.initial_capital) / self.initial_capital * 100.0, 2),
                "total_return_pct": round((final_w - self.initial_capital) / self.initial_capital * 100.0, 2),
                "cagr_pct": round(cagr, 2),
                "sharpe_ratio": round(sharpe, 2),
                "sortino_ratio": round(sortino, 2),
                "calmar_ratio": round(calmar, 2),
                "max_drawdown_pct": round(dd_pct, 2),
                "win_rate_pct": 100.0 if final_w > self.initial_capital else 0.0,
                "profit_factor": 1.0,
                "total_trades": len(holdings),
                "turnover_ratio": QuantMetrics.calculate_turnover(total_turnover_dollars, np.mean(wealth_history)),
                "total_fees_and_taxes": round(total_brokerage_paid + total_taxes_paid, 2),
                "total_frictions_paid": round(total_brokerage_paid + total_taxes_paid, 2)
            }

        # Step through all trading dates sequentially
        trade_id_seq = 1

        for d_idx in range(25, len(sorted_dates)):
            curr_date = sorted_dates[d_idx]
            ensemble.on_tick_advance()
            risk_engine.on_tick_advance()

            # Mark to market current portfolio wealth
            current_port_positions = []
            long_val = 0.0
            short_liab = 0.0

            for ticker, h in list(holdings.items()):
                df = symbol_dfs[ticker]
                sub = df[df["date"].astype(str).str[:10] == curr_date]
                if sub.empty:
                    continue
                p = float(sub["close"].iloc[0])
                qty = h["qty"]
                is_s = h["is_short"]

                if not is_s:
                    mkt_v = qty * p
                    unrealized = mkt_v - (qty * h["avg_price"])
                    unrealized_pct = unrealized / (qty * h["avg_price"]) * 100.0
                    long_val += mkt_v
                else:
                    mkt_v = -(qty * p)
                    unrealized = (h["avg_price"] - p) * qty
                    unrealized_pct = unrealized / (qty * h["avg_price"]) * 100.0
                    short_liab += abs(mkt_v)

                current_port_positions.append(PortfolioPosition(
                    ticker=ticker,
                    quantity=qty if not is_s else -qty,
                    average_buy_price=h["avg_price"],
                    current_price=p,
                    market_value=mkt_v,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=unrealized_pct,
                    is_short=is_s
                ))

            tot_wealth = cash + long_val - short_liab
            wealth_history.append(tot_wealth)
            if len(wealth_history) > 1:
                r_day = (tot_wealth - wealth_history[-2]) / wealth_history[-2]
                returns_history.append(r_day)
            else:
                r_day = 0.0

            gross_exposures.append((long_val + short_liab) / max(tot_wealth, 1.0) * 100.0)
            net_exposures.append((long_val - short_liab) / max(tot_wealth, 1.0) * 100.0)

            # Portfolio snapshot for risk engine
            portfolio_snap = Portfolio(
                initial_capital=self.initial_capital,
                capital_remaining=cash,
                total_wealth=tot_wealth,
                holdings=current_port_positions
            )
            exposure_mgr.update_portfolio(portfolio_snap)
            risk_engine.on_new_trading_day(tot_wealth)

            # Precompute features once per symbol on curr_date for maximum performance
            day_features = {}
            for t_sym, df_sym in symbol_dfs.items():
                d_idx_list = df_sym[df_sym["date"].astype(str).str[:10] <= curr_date].index
                if len(d_idx_list) >= 25:
                    sub_df = df_sym.iloc[:d_idx_list[-1] + 1]
                    day_features[t_sym] = (FeatureExtractor.compute_features(sub_df), sub_df.iloc[-1])

            # 1. Automated Dynamic ATR, Trailing-Stop, Take-Profit, and Time Exits
            market_context = {}
            for t_held in holdings:
                if t_held in day_features:
                    f_held, bar_held = day_features[t_held]
                    atr_val = f_held.get("atr", f_held.get("atr_14", float(bar_held["close"]) * 0.02))
                    reg_val = RegimeDetector.classify_ticker(f_held)
                    market_context[t_held] = {
                        "atr": atr_val,
                        "regime": reg_val.value,
                        "current_price": float(bar_held["close"])
                    }

            exits = exposure_mgr.check_exits(portfolio_snap, market_context=market_context)
            for exit_ticker, exit_action, exit_qty, exit_reason in exits:
                if exit_ticker not in holdings:
                    continue
                h = holdings[exit_ticker]
                df = symbol_dfs[exit_ticker]
                sub = df[df["date"].astype(str).str[:10] == curr_date]
                if sub.empty:
                    continue
                p = float(sub["close"].iloc[0])
                v = float(sub.get("volume", pd.Series([100000])).iloc[0])

                slip = self.compute_slippage(exit_qty, v)
                is_s = h["is_short"]
                exec_p = p * (1.0 - slip) if not is_s else p * (1.0 + slip)

                # Friction calculations
                gross_pnl = ((exec_p - h["avg_price"]) * exit_qty) if not is_s else ((h["avg_price"] - exec_p) * exit_qty)
                order_val = exit_qty * exec_p
                tax = order_val * self.stt_rate
                dp = self.dp_charge if not is_s else 0.0
                fee = self.brokerage + tax + dp
                net_pnl = gross_pnl - fee
                if not is_s:
                    cash += (order_val - fee)
                else:
                    cash -= (order_val + fee)
                total_brokerage_paid += self.brokerage
                total_taxes_paid += tax
                total_dp_paid += dp
                total_turnover_dollars += order_val

                return_pct = round(net_pnl / (h["avg_price"] * exit_qty) * 100.0, 2)
                trade_history.append({
                    "trade_id": f"TRD_{trade_id_seq:04d}",
                    "ticker": exit_ticker,
                    "action": exit_action.value,
                    "shares": exit_qty,
                    "entry_price": round(h["avg_price"], 2),
                    "exit_price": round(exec_p, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "brokerage": self.brokerage,
                    "taxes_and_dp": round(tax + dp, 2),
                    "net_pnl": round(net_pnl, 2),
                    "return_pct": return_pct,
                    "exit_date": curr_date,
                    "reason": exit_reason
                })
                trade_id_seq += 1
                del holdings[exit_ticker]
                ensemble.record_trade(exit_ticker)
                risk_engine.record_trade_completion(exit_ticker, return_pct=return_pct)

            # 2. Evaluate Signals & ML Predictions across Universe
            for ticker, (features, curr_bar) in day_features.items():
                price = float(curr_bar["close"])
                vol = float(curr_bar.get("volume", 100000))

                regime = RegimeDetector.classify_ticker(features)
                regime_returns[regime.value].append(r_day)

                ml_conv = ml_inference_fn(features) if ml_inference_fn else 0.0
                curr_qty = holdings[ticker]["qty"] if ticker in holdings else 0
                if ticker in holdings and holdings[ticker]["is_short"]:
                    curr_qty = -curr_qty

                decision = ensemble.evaluate(
                    ticker=ticker,
                    features=features,
                    news_sentiment=0.0,
                    ml_conviction=ml_conv,
                    current_holding_qty=curr_qty
                )

                if not decision.should_trade:
                    continue

                # Position sizing
                qty = position_sizer.calculate_quantity(
                    action=decision.action,
                    price=price,
                    conviction=decision.conviction,
                    portfolio=portfolio_snap,
                    available_cash=cash,
                    current_holding_qty=curr_qty,
                    volatility=features.get("volatility_20", 0.02),
                    ticker=ticker
                )

                if qty <= 0:
                    continue

                # Pre-trade risk validation
                slip = self.compute_slippage(qty, vol)
                exec_p = price * (1.0 + slip) if decision.action == TradeAction.BUY else price * (1.0 - slip)

                from trading_bot.api.schemas import OrderRequest
                req = OrderRequest(ticker=ticker, action=decision.action, quantity=qty)
                is_ok, reason = risk_engine.validate_order(req, exec_p, portfolio_snap, cash)
                if not is_ok:
                    continue

                # Execute entry
                order_val = qty * exec_p
                tax = order_val * self.stt_rate
                fee = self.brokerage + tax
                total_brokerage_paid += self.brokerage
                total_taxes_paid += tax
                total_turnover_dollars += order_val

                if decision.action == TradeAction.BUY:
                    cash -= (order_val + fee)
                    holdings[ticker] = {"qty": qty, "avg_price": exec_p, "is_short": False, "peak_gain": 0.0}
                    ensemble.record_trade(ticker)

                elif decision.action == TradeAction.SHORT:
                    cash += (order_val - fee)
                    holdings[ticker] = {"qty": qty, "avg_price": exec_p, "is_short": True, "peak_gain": 0.0}
                    ensemble.record_trade(ticker)

        # Final Liquidation of open holdings on last date
        last_date = sorted_dates[-1]
        for ticker, h in list(holdings.items()):
            df = symbol_dfs[ticker]
            last_p = float(df["close"].iloc[-1])
            v = float(df.get("volume", pd.Series([100000])).iloc[-1])
            qty = h["qty"]
            is_s = h["is_short"]
            slip = self.compute_slippage(qty, v)
            exec_p = last_p * (1.0 - slip) if not is_s else last_p * (1.0 + slip)
            gross_pnl = ((exec_p - h["avg_price"]) * qty) if not is_s else ((h["avg_price"] - exec_p) * qty)
            order_val = qty * exec_p
            tax = order_val * self.stt_rate
            dp = self.dp_charge if not is_s else 0.0
            fee = self.brokerage + tax + dp
            net_pnl = gross_pnl - fee
            if not is_s:
                cash += (order_val - fee)
            else:
                cash -= (order_val + fee)
            total_brokerage_paid += self.brokerage
            total_taxes_paid += tax
            total_dp_paid += dp
            total_turnover_dollars += order_val
            trade_history.append({
                "trade_id": f"TRD_{trade_id_seq:04d}",
                "ticker": ticker,
                "action": "SELL_FINAL" if not is_s else "COVER_FINAL",
                "shares": qty,
                "entry_price": round(h["avg_price"], 2),
                "exit_price": round(exec_p, 2),
                "gross_pnl": round(gross_pnl, 2),
                "brokerage": self.brokerage,
                "taxes_and_dp": round(tax + dp, 2),
                "net_pnl": round(net_pnl, 2),
                "return_pct": round(net_pnl / (h["avg_price"] * qty) * 100.0, 2),
                "exit_date": last_date,
                "reason": "SIMULATION_END"
            })
            trade_id_seq += 1

        final_w = cash
        total_ret = (final_w - self.initial_capital) / self.initial_capital * 100.0
        dd_amt, dd_pct = QuantMetrics.calculate_max_drawdown(wealth_history)
        cagr = QuantMetrics.calculate_cagr(self.initial_capital, final_w, len(sorted_dates))
        sharpe = QuantMetrics.calculate_sharpe(returns_history)
        sortino = QuantMetrics.calculate_sortino(returns_history)
        calmar = QuantMetrics.calculate_calmar(cagr, dd_pct)
        trade_metrics = QuantMetrics.calculate_trade_metrics([t["net_pnl"] for t in trade_history])

        # Per-regime returns and Sharpes
        regime_stats = {}
        for reg_name, rets in regime_returns.items():
            if len(rets) > 5:
                regime_stats[reg_name] = {
                    "count": len(rets),
                    "mean_daily_return_pct": round(float(np.mean(rets)) * 100.0, 3),
                    "annualized_sharpe": QuantMetrics.calculate_sharpe(rets)
                }

        return {
            "strategy": strategy_mode.capitalize(),
            "initial_wealth": self.initial_capital,
            "final_wealth": round(final_w, 2),
            "total_return_pct": round(total_ret, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "max_drawdown_pct": round(dd_pct, 2),
            "max_drawdown_dollars": round(dd_amt, 2),
            "win_rate_pct": trade_metrics["win_rate_pct"],
            "profit_factor": trade_metrics["profit_factor"],
            "total_trades": len(trade_history),
            "turnover_ratio": QuantMetrics.calculate_turnover(total_turnover_dollars, np.mean(wealth_history)),
            "avg_gross_exposure_pct": round(float(np.mean(gross_exposures)), 2) if gross_exposures else 0.0,
            "avg_net_exposure_pct": round(float(np.mean(net_exposures)), 2) if net_exposures else 0.0,
            "total_brokerage_paid": round(total_brokerage_paid, 2),
            "total_taxes_and_dp_paid": round(total_taxes_paid + total_dp_paid, 2),
            "total_frictions_paid": round(total_brokerage_paid + total_taxes_paid + total_dp_paid, 2),
            "per_regime_breakdown": regime_stats,
            "trade_records": trade_history
        }
