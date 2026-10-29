"""Main lifecycle controller and autonomous trading bot execution engine."""
from __future__ import annotations
import logging
import signal
import sys
import time
from typing import Optional

from trading_bot.api.client import TradingAPIClient
from trading_bot.api.schemas import MarketTick, NewsItem, TradeAction
from trading_bot.config import Settings, get_settings
from trading_bot.execution.order_manager import OrderManager
from trading_bot.features.indicators import FeatureExtractor
from trading_bot.features.news_features import NewsManager
from trading_bot.logging_config import setup_logging
from trading_bot.market_data.history import SymbolHistoryStore
from trading_bot.market_data.kafka_consumer import MarketDataConsumer
from trading_bot.ml.inference import MLInferenceEngine
from trading_bot.portfolio.accounting import PortfolioTracker
from trading_bot.portfolio.performance import PerformanceMetrics
from trading_bot.risk.exposure import ExposureManager
from trading_bot.risk.limits import PreTradeRiskEngine, RiskLimits
from trading_bot.risk.position_sizing import PositionSizer
from trading_bot.strategy.ensemble import StrategyEnsemble

logger = logging.getLogger("trading_bot.main")


class AutonomousTradingBot:
    """Competition-grade autonomous trading bot for MANTHAN 2026."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.is_running = False

        # Core Components
        self.api_client = TradingAPIClient(self.settings)
        self.history_store = SymbolHistoryStore(max_bars=self.settings.tick_window_size)
        self.portfolio_tracker = PortfolioTracker(initial_capital=1_000_000.0)
        self.news_manager = NewsManager()
        self.ml_engine = MLInferenceEngine(
            models_dir=self.settings.models_dir,
            enabled=self.settings.ml_enabled
        )

        # Risk and Execution
        risk_limits = RiskLimits(
            max_position_pct=self.settings.max_position_pct,
            max_total_long_pct=self.settings.max_total_long_pct,
            max_total_short_pct=self.settings.max_total_short_pct,
            max_gross_exposure_pct=self.settings.max_gross_exposure_pct,
            max_single_trade_pct=self.settings.max_single_trade_pct,
            max_open_positions=self.settings.max_open_positions,
            min_cash_reserve_pct=self.settings.min_cash_reserve_pct,
            short_selling_enabled=True,
        )
        self.risk_engine = PreTradeRiskEngine(risk_limits)
        self.position_sizer = PositionSizer(risk_limits)
        self.exposure_manager = ExposureManager(
            emergency_halt_drawdown_pct=self.settings.emergency_halt_drawdown_pct,
            per_position_stop_loss_pct=self.settings.per_position_loss_pct,
        )
        self.ensemble = StrategyEnsemble(
            signal_threshold=self.settings.signal_threshold,
            no_trade_band=self.settings.no_trade_band,
            cooldown_ticks=self.settings.cooldown_ticks,
            short_selling_enabled=True,
            ml_weight=self.settings.ml_weight
        )
        self.order_manager = OrderManager(
            api_client=self.api_client,
            risk_engine=self.risk_engine,
            portfolio_tracker=self.portfolio_tracker,
        )

        # Kafka Consumer
        self.consumer = MarketDataConsumer(
            settings=self.settings,
            api_client=self.api_client,
            on_tick_callback=self.handle_tick,
            on_news_callback=self.handle_news,
            on_preflight_callback=self.handle_preflight,
        )

        # State tracking
        self.active_tickers: list[str] = []
        self.current_stage: str = "unknown"
        self.wealth_history: list[float] = []
        self.last_sync_time = 0.0
        self.last_stage_check_time = 0.0
        self.stage_check_interval_seconds = 2.0
        self.stage_status: str = "UNKNOWN"
        self.stage_gate_reason: str = "Stage has not been verified"

    def _trading_allowed(self, force_refresh: bool = False) -> bool:
        """Fail-closed stage gate. Only RUNNING simulation stage may place orders."""
        now = time.monotonic()
        if force_refresh or now - self.last_stage_check_time >= self.stage_check_interval_seconds:
            self.last_stage_check_time = now
            try:
                info = self.api_client.get_stage()
                stage = str(info.get("stage", "")).strip().upper()
                status = str(info.get("status", "")).strip().upper()
                self.current_stage = stage or "UNKNOWN"
                self.stage_status = status or "UNKNOWN"
                self.stage_gate_reason = f"stage={self.current_stage}, status={self.stage_status}"
            except Exception as exc:
                self.current_stage = "UNKNOWN"
                self.stage_status = "UNKNOWN"
                self.stage_gate_reason = f"stage check failed: {exc}"
                logger.error("Trading disabled because platform stage could not be verified: %s", exc)
                return False
        stage_ok = self.current_stage in {"SIMULATION", "SIM"}
        status_ok = self.stage_status == "RUNNING"
        if not (stage_ok and status_ok):
            logger.info("Order blocked by platform stage gate (%s)", self.stage_gate_reason)
            return False
        return True

    def _periodic_sync(self) -> None:
        """Synchronize local accounting even when strategy paths return early."""
        now = time.monotonic()
        if now - self.last_sync_time > 30.0 and self.settings.api_token:
            try:
                self.portfolio_tracker.sync_from_gateway(self.api_client)
                self.last_sync_time = now
            except Exception as exc:
                logger.warning("Periodic portfolio sync failed: %s", exc)

    def initialize(self) -> bool:
        """Run pre-flight discovery, indicator warm-up, and account sync."""
        logger.info("Initializing MANTHAN Autonomous Trading Bot...")

        # 1. Connectivity Check
        conn = self.api_client.check_connectivity()
        logger.info(f"API Gateway Connectivity: {conn}")

        # 2. Stage and Clock
        try:
            stage_info = self.api_client.get_stage()
            self.current_stage = str(stage_info.get("stage", "unknown")).upper()
            self.stage_status = str(stage_info.get("status", "unknown")).upper()
            self.last_stage_check_time = time.monotonic()
            self.stage_gate_reason = f"stage={self.current_stage}, status={self.stage_status}"
            logger.info(f"Platform Stage: {self.current_stage} (status: {self.stage_status})")
        except Exception as e:
            logger.warning(f"Could not fetch stage: {e}")

        # 3. Stock Universe Discovery
        try:
            stocks = self.api_client.get_stocks()
            self.active_tickers = [s.get("ticker", s.get("symbol", "")) for s in stocks if s.get("ticker") or s.get("symbol")]
            logger.info(f"Discovered {len(self.active_tickers)} tradable stocks: {self.active_tickers[:8]}...")
        except Exception as e:
            logger.error(f"Failed to discover stocks: {e}")

        # 4. Indicator Warm-up via Historical OHLCV
        if self.active_tickers:
            logger.info(f"Warming up technical indicators for {len(self.active_tickers)} symbols...")
            self.history_store.warm_up_universe(self.api_client, self.active_tickers, days=365)

        # 5. Authoritative Portfolio Sync
        if self.settings.api_token:
            self.portfolio_tracker.sync_from_gateway(self.api_client)

        snap = self.portfolio_tracker.get_portfolio_snapshot()
        print("\n" + "=" * 65)
        print("          MAKEWAY AUTONOMOUS TRADING BOT INITIALIZED          ")
        print("=" * 65)
        print(f"Team:             Makeway (MN2672)")
        print(f"Starting Wealth:  INR {snap.total_wealth:,.2f}")
        print(f"Liquid Cash:      INR {snap.capital_remaining:,.2f}")
        print(f"Realized P&L:     INR {snap.realized_pnl:,.2f}")
        print(f"Unrealized P&L:   INR {snap.unrealized_pnl:,.2f}")
        print(f"ML Model:         {self.ml_engine.stats['model_name']} (Enabled: {self.settings.ml_enabled})")
        print("=" * 65 + "\n")

        return True

    def handle_preflight(self, seq: int) -> None:
        """Invoked when a pre-flight standby heartbeat is received and ACKed."""
        try:
            stocks = self.api_client.get_stocks()
            self.active_tickers = [s.get("ticker", s.get("symbol", "")) for s in stocks if s.get("ticker") or s.get("symbol")]
            logger.info(f"Pre-flight universe refreshed ({len(self.active_tickers)} symbols)")
        except Exception:
            pass

    def handle_news(self, news: NewsItem) -> None:
        """Handle incoming news flash event from Kafka."""
        score = self.news_manager.on_news(news)
        ticker = news.ticker or "GLOBAL"
        logger.debug(f"News [{ticker}]: score={score:+.2f} | {news.headline[:60]}")

    def handle_tick(self, tick: MarketTick) -> None:
        """Critical real-time path: tick -> features -> risk -> decision -> execution."""
        ticker = tick.ticker.upper()

        # 1. Update history store and mark price
        self.history_store.on_tick(tick)
        self.portfolio_tracker.on_price_update(ticker, tick.close)
        self.ensemble.on_tick_advance()
        self.news_manager.on_tick_cycle()
        self._periodic_sync()

        # Do not place entry or exit orders outside the authorized live simulation.
        # Market/portfolio marks above are still updated while paused/preflight.
        if not self._trading_allowed():
            return

        # 2. Circuit Breaker / Stop-Loss Exit Check
        portfolio_snap = self.portfolio_tracker.get_portfolio_snapshot()
        self.exposure_manager.update_portfolio(portfolio_snap)
        self.wealth_history.append(portfolio_snap.total_wealth)

        if self.exposure_manager.is_halted:
            logger.warning("Trading is HALTED due to maximum drawdown threshold.")
            return

        # Check automated Take-Profit (+15%), Trailing-Stop, and Stop-Loss (-5%) exits
        market_context = {}
        for held_ticker in self.active_tickers:
            held_df = self.history_store.get_dataframe(held_ticker)
            if len(held_df) >= 2:
                held_features = FeatureExtractor.compute_features(held_df, self.settings.lookback_short, self.settings.lookback_medium, self.settings.lookback_long)
                market_context[held_ticker] = {"atr": held_features.get("atr", 0.0), "regime": "UNKNOWN"}
        exits = self.exposure_manager.check_exits(portfolio_snap, market_context=market_context)
        for exit_ticker, exit_action, exit_qty, exit_reason in exits:
            if not self._trading_allowed():
                break
            p = self.history_store.get_latest_price(exit_ticker) or tick.close
            logger.info(f"Triggering automated exit on {exit_ticker}: {exit_action} {exit_qty} @ ~{p:.2f} ({exit_reason})")
            self.order_manager.execute_decision(
                ticker=exit_ticker,
                action=exit_action,
                quantity=exit_qty,
                estimated_price=p,
                simulated_date=tick.simulated_date
            )

        # 3. Strategy Evaluation & ML Feature Extraction
        df = self.history_store.get_dataframe(ticker)
        if len(df) < 5:
            return

        features = FeatureExtractor.compute_features(
            df,
            short_win=self.settings.lookback_short,
            med_win=self.settings.lookback_medium,
            long_win=self.settings.lookback_long,
        )

        news_sentiment = self.news_manager.get_sentiment(ticker)
        current_qty = self.portfolio_tracker.get_holding_quantity(ticker)

        # Generate ML conviction if enabled
        ml_conviction = self.ml_engine.predict_conviction(features) if self.settings.ml_enabled else 0.0

        decision = self.ensemble.evaluate(
            ticker=ticker,
            features=features,
            news_sentiment=news_sentiment,
            ml_conviction=ml_conviction,
            current_holding_qty=current_qty
        )

        if not decision.should_trade:
            return
        if not self._trading_allowed():
            return

        # 4. Position Sizing
        available_cash = self.portfolio_tracker.get_cash_snapshot()
        qty = self.position_sizer.calculate_quantity(
            action=decision.action,
            price=tick.close,
            conviction=decision.conviction,
            portfolio=portfolio_snap,
            available_cash=available_cash,
            current_holding_qty=current_qty,
            volatility=features.get("volatility_20", 0.02)
        )

        if qty <= 0:
            return

        # 5. Order Execution
        order_record = self.order_manager.execute_decision(
            ticker=ticker,
            action=decision.action,
            quantity=qty,
            estimated_price=tick.close,
            simulated_date=tick.simulated_date
        )

        if order_record and order_record.state.value == "FILLED":
            self.ensemble.record_trade(ticker)
            # Re-read portfolio snapshot after fill
            portfolio_snap = self.portfolio_tracker.get_portfolio_snapshot()
            print(
                f"[FILL] {decision.action.value} {qty} {ticker} @ {order_record.execution_price:.2f} | "
                f"Realized PnL: INR {portfolio_snap.realized_pnl:+,.2f} | "
                f"Unrealized PnL: INR {portfolio_snap.unrealized_pnl:+,.2f} | "
                f"Total Wealth: INR {portfolio_snap.total_wealth:,.2f}"
            )


    def run(self) -> None:
        """Start the autonomous trading bot loop."""
        self.is_running = True
        logger.info("Connecting Kafka consumer and starting real-time loop...")
        self.consumer.connect()

        try:
            while self.is_running:
                msgs = self.consumer.poll_once()
                if msgs == 0:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown and final report output."""
        self.is_running = False
        logger.info("Stopping bot...")
        self.consumer.stop()
        self.api_client.close()

        # Performance summary with distinct Realized & Unrealized P&L
        snap = self.portfolio_tracker.get_portfolio_snapshot()
        oms_stats = self.order_manager.stats
        dd_amt, dd_pct = PerformanceMetrics.calculate_max_drawdown(self.wealth_history)

        print("\n" + "=" * 65)
        print("          MANTHAN AUTONOMOUS TRADING BOT - REPORT          ")
        print("=" * 65)
        print(f"Final Total Wealth:       INR {snap.total_wealth:,.2f}")
        print(f"Total Combined Net P&L:   INR {snap.total_pnl:,.2f} ({snap.total_pnl_pct:+.2f}%)")
        print(f"  * Realized P&L:         INR {snap.realized_pnl:,.2f}")
        print(f"  * Unrealized P&L:       INR {snap.unrealized_pnl:,.2f}")
        print(f"Liquid Cash Remaining:    INR {snap.capital_remaining:,.2f}")
        print(f"Max Portfolio Drawdown:   {dd_pct * 100:.2f}% (INR {dd_amt:,.2f})")
        print(f"Orders Submitted:         {oms_stats['total_submitted']}")
        print(f"Orders Filled:            {oms_stats['filled']}")
        print(f"Orders Rejected:          {oms_stats['rejected']}")
        print(f"Open Positions Count:     {len(snap.holdings)}")
        print(f"ML Model In Use:          {self.ml_engine.stats['model_name']}")
        print(f"ML Inference Latency:     {self.ml_engine.stats['last_latency_ms']} ms")
        print("=" * 65 + "\n")


def main() -> None:
    setup_logging()
    settings = get_settings()
    bot = AutonomousTradingBot(settings)

    def handle_sig(sig, frame):
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    if bot.initialize():
        bot.run()


if __name__ == "__main__":
    main()
