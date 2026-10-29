"""Unit tests for individual, dip-buying, and ensemble strategy signals."""
import pytest

from trading_bot.strategy.ensemble import StrategyEnsemble
from trading_bot.strategy.regime import MarketRegime, RegimeDetector
from trading_bot.strategy.signals import (
    DipBuyingStrategy, MeanReversionStrategy, MomentumStrategy, TrendFollowingStrategy
)
from trading_bot.features.news_features import NewsSentimentAnalyzer


def test_momentum_positive_roc():
    features = {
        "roc_short": 5.0,
        "roc_med": 3.0,
        "volume_ratio": 1.2
    }
    sig = MomentumStrategy.evaluate("AETX", features)
    assert sig.score > 0.5
    assert sig.action.value == "BUY"


def test_mean_reversion_oversold():
    features = {
        "bb_percent_b": 0.05,
        "rsi_14": 20.0,
        "z_score_price": -2.2
    }
    sig = MeanReversionStrategy.evaluate("AETX", features)
    assert sig.score >= 0.5


def test_dip_buying_in_bull_trend():
    features = {
        "close": 105.0,
        "ema_med": 104.0,
        "sma_long": 100.0,
        "rsi_14": 34.0,  # Oversold dip
        "bb_percent_b": 0.12,  # Near lower band
        "macd_hist": 0.10
    }
    sig = DipBuyingStrategy.evaluate("AETX", features, ml_conviction=0.30)
    assert sig.score >= 0.50
    assert "BUY_THE_DIP" in sig.rationale


def test_rally_shorting_in_bear_trend():
    features = {
        "close": 85.0,
        "ema_med": 88.0,
        "sma_long": 100.0,
        "rsi_14": 65.0,  # Overbought rally in downtrend
        "bb_percent_b": 0.88,
        "macd_hist": -0.15
    }
    sig = DipBuyingStrategy.evaluate("AETX", features, ml_conviction=-0.30)
    assert sig.score <= -0.50
    assert "SHORT_THE_RALLY" in sig.rationale


def test_news_sentiment_analysis():
    bullish = "Aethelgard Corp reports record quarterly revenue and surges in profits"
    assert NewsSentimentAnalyzer.score_text(bullish) > 0.4

    bearish = "Regulatory probe and lawsuit triggers massive plunge in shares"
    assert NewsSentimentAnalyzer.score_text(bearish) < -0.4

    neutral = "Company announced date for annual shareholder meeting"
    assert abs(NewsSentimentAnalyzer.score_text(neutral)) < 0.2


def test_ensemble_deadband():
    ensemble = StrategyEnsemble(signal_threshold=0.3, no_trade_band=0.1)
    neutral_features = {
        "close": 100.0,
        "sma_short": 100.0,
        "sma_med": 100.0,
        "sma_long": 100.0,
        "ema_short": 100.0,
        "ema_med": 100.0,
        "rsi_14": 50.0,
        "macd_hist": 0.0,
        "bb_percent_b": 0.5,
        "roc_short": 0.0,
        "roc_med": 0.0,
        "volatility_20": 0.01,
        "z_score_price": 0.0,
        "volume_ratio": 1.0,
    }
    decision = ensemble.evaluate("AETX", neutral_features, news_sentiment=0.0)
    assert not decision.should_trade
    assert decision.action is None
