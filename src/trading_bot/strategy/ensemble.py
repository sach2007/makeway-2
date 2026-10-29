"""Regime-aware ensemble strategy combining technical, dip-buying, NLP, and machine learning alpha signals."""
from __future__ import annotations
import logging
from typing import Optional

from trading_bot.api.schemas import TradeAction
from trading_bot.strategy.regime import MarketRegime, RegimeDetector
from trading_bot.strategy.signals import (
    DipBuyingStrategy, MeanReversionStrategy, MLSignalStrategy,
    MomentumStrategy, NewsSignalStrategy, SignalScore, TrendFollowingStrategy
)

logger = logging.getLogger(__name__)


class EnsembleDecision:
    """Actionable decision output from strategy evaluation."""
    def __init__(
        self,
        ticker: str,
        action: Optional[TradeAction],
        conviction: float,
        regime: MarketRegime,
        rationale: str = "",
        candidate_scores: Optional[dict[str, float]] = None
    ):
        self.ticker = ticker
        self.action = action  # None if HOLD / NO_TRADE
        self.conviction = conviction  # Absolute strength [0.0, 1.0]
        self.regime = regime
        self.rationale = rationale
        self.candidate_scores = candidate_scores or {}

    @property
    def should_trade(self) -> bool:
        return self.action is not None and self.conviction > 0.0


class StrategyEnsemble:
    """Combines individual signals using dynamic regime weighting, ML predictions, and automated dip buying."""

    def __init__(
        self,
        signal_threshold: float = 0.25,
        no_trade_band: float = 0.10,
        cooldown_ticks: int = 3,
        short_selling_enabled: bool = True,
        ml_weight: float = 0.30
    ):
        self.signal_threshold = signal_threshold
        self.no_trade_band = no_trade_band
        self.cooldown_ticks = cooldown_ticks
        self.short_selling_enabled = short_selling_enabled
        self.ml_weight = ml_weight
        self._cooldowns: dict[str, int] = {}

    def on_tick_advance(self) -> None:
        """Decrement cooldown counters on each tick advance."""
        for ticker in list(self._cooldowns.keys()):
            if self._cooldowns[ticker] > 0:
                self._cooldowns[ticker] -= 1
            if self._cooldowns[ticker] <= 0:
                del self._cooldowns[ticker]

    def record_trade(self, ticker: str) -> None:
        """Put ticker on cooldown after an order execution."""
        self._cooldowns[ticker.upper()] = self.cooldown_ticks

    def evaluate(
        self,
        ticker: str,
        features: dict[str, float],
        news_sentiment: float = 0.0,
        ml_conviction: float = 0.0,
        current_holding_qty: int = 0
    ) -> EnsembleDecision:
        """Evaluate all alpha strategies and return an actionable trading decision."""
        ticker = ticker.upper()

        # 1. Cooldown check
        if self._cooldowns.get(ticker, 0) > 0:
            return EnsembleDecision(
                ticker=ticker,
                action=None,
                conviction=0.0,
                regime=MarketRegime.UNKNOWN,
                rationale=f"In cooldown ({self._cooldowns[ticker]} ticks remaining)"
            )

        # 2. Classify market regime
        regime = RegimeDetector.classify_ticker(features)
        weights = RegimeDetector.get_regime_weights(regime)

        # 3. Calculate candidate signals
        mom_sig = MomentumStrategy.evaluate(ticker, features)
        trend_sig = TrendFollowingStrategy.evaluate(ticker, features)
        mr_sig = MeanReversionStrategy.evaluate(ticker, features)
        dip_sig = DipBuyingStrategy.evaluate(ticker, features, ml_conviction)
        news_sig = NewsSignalStrategy.evaluate(ticker, news_sentiment)
        ml_sig = MLSignalStrategy.evaluate(ticker, ml_conviction)

        candidate_scores = {
            "momentum": mom_sig.score,
            "trend": trend_sig.score,
            "mean_reversion": mr_sig.score,
            "dip_buyer": dip_sig.score,
            "news": news_sig.score,
            "ml": ml_sig.score
        }

        # 4. Compute rule-based ensemble score [-1.0, 1.0]
        base_rule = (
            mom_sig.score * weights["momentum"] +
            trend_sig.score * weights["trend"] +
            mr_sig.score * weights["mean_reversion"] +
            news_sig.score * weights["news"]
        )

        # Add dip-buying conviction overlay
        if dip_sig.score != 0.0:
            rule_score = (base_rule * 0.70) + (dip_sig.score * 0.30)
        else:
            rule_score = base_rule

        # Blend ML prediction if non-zero
        if ml_conviction != 0.0:
            ensemble_score = (rule_score * (1.0 - self.ml_weight)) + (ml_sig.score * self.ml_weight)
        else:
            ensemble_score = rule_score

        conviction = abs(ensemble_score)

        # 5. Deadband / No-Trade filter
        if conviction < self.no_trade_band or conviction < self.signal_threshold:
            return EnsembleDecision(
                ticker=ticker,
                action=None,
                conviction=0.0,
                regime=regime,
                rationale=f"Score {ensemble_score:.2f} within deadband/below threshold ({self.signal_threshold})",
                candidate_scores=candidate_scores
            )

        # 6. Map conviction and current holding to explicit order action
        action: Optional[TradeAction] = None

        if ensemble_score > 0:  # Bullish conviction
            if current_holding_qty < 0:
                action = TradeAction.COVER
            elif current_holding_qty == 0:
                action = TradeAction.BUY
            else:
                action = TradeAction.BUY if conviction > 0.60 else None

        else:  # Bearish conviction (ensemble_score < 0)
            if current_holding_qty > 0:
                action = TradeAction.SELL
            elif current_holding_qty == 0:
                action = TradeAction.SHORT if self.short_selling_enabled else None
            else:
                action = None

        rationale = (
            f"Regime={regime.value}, Score={ensemble_score:+.2f} (Rule={rule_score:+.2f}, Dip={dip_sig.score:+.2f}, ML={ml_sig.score:+.2f}), "
            f"Conv={conviction:.2f}, Pos={current_holding_qty}"
        )

        return EnsembleDecision(
            ticker=ticker,
            action=action,
            conviction=conviction,
            regime=regime,
            rationale=rationale,
            candidate_scores=candidate_scores
        )
