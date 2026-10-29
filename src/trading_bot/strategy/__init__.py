"""Quantitative alpha signals, regime classification, and strategy ensemble."""

from trading_bot.strategy.ensemble import EnsembleDecision, StrategyEnsemble
from trading_bot.strategy.regime import MarketRegime, RegimeDetector
from trading_bot.strategy.signals import (
    MeanReversionStrategy,
    MLSignalStrategy,
    MomentumStrategy,
    NewsSignalStrategy,
    SignalScore,
    TrendFollowingStrategy,
)

__all__ = [
    "MarketRegime",
    "RegimeDetector",
    "SignalScore",
    "MomentumStrategy",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "NewsSignalStrategy",
    "MLSignalStrategy",
    "EnsembleDecision",
    "StrategyEnsemble",
]
