"""Market regime detection and classification."""
from __future__ import annotations
from enum import Enum
from typing import Any


class MarketRegime(str, Enum):
    BULL_TRENDING = "BULL_TRENDING"
    BEAR_TRENDING = "BEAR_TRENDING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    RANGE_BOUND = "RANGE_BOUND"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """Classifies current market condition per ticker based on trend strength and volatility."""

    @classmethod
    def classify_ticker(cls, features: dict[str, float]) -> MarketRegime:
        """Classify symbol market regime using technical indicators."""
        close = features.get("close", 0.0)
        if close <= 0:
            return MarketRegime.UNKNOWN

        sma_s = features.get("sma_short", close)
        sma_m = features.get("sma_med", close)
        sma_l = features.get("sma_long", close)
        vol_20 = features.get("volatility_20", 0.0)
        rsi = features.get("rsi_14", 50.0)

        # High Volatility filter: daily log return std > 3.5%
        if vol_20 > 0.035:
            return MarketRegime.HIGH_VOLATILITY

        # Bull Trending: short > med > long and RSI > 50
        if sma_s > sma_m > sma_l and rsi > 50.0:
            return MarketRegime.BULL_TRENDING

        # Bear Trending: short < med < long and RSI < 50
        if sma_s < sma_m < sma_l and rsi < 50.0:
            return MarketRegime.BEAR_TRENDING

        # Otherwise Range Bound / Mean Reverting
        return MarketRegime.RANGE_BOUND

    @classmethod
    def get_regime_weights(cls, regime: MarketRegime) -> dict[str, float]:
        """Returns strategy weights dynamically tuned for the active regime."""
        if regime == MarketRegime.BULL_TRENDING:
            return {"momentum": 0.45, "trend": 0.40, "mean_reversion": 0.05, "news": 0.10}
        elif regime == MarketRegime.BEAR_TRENDING:
            return {"momentum": 0.40, "trend": 0.45, "mean_reversion": 0.05, "news": 0.10}
        elif regime == MarketRegime.RANGE_BOUND:
            return {"momentum": 0.10, "trend": 0.10, "mean_reversion": 0.70, "news": 0.10}
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # In high volatility, dampen momentum to avoid whipsaws, emphasize mean reversion and news
            return {"momentum": 0.20, "trend": 0.20, "mean_reversion": 0.45, "news": 0.15}
        else:
            return {"momentum": 0.35, "trend": 0.35, "mean_reversion": 0.20, "news": 0.10}
