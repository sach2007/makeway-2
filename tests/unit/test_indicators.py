"""Unit tests for technical indicator computations."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.features.indicators import TechnicalIndicators, FeatureExtractor


def test_sma_calculation():
    data = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma = TechnicalIndicators.sma(data, period=3)
    assert len(sma) == 5
    assert np.isclose(sma.iloc[-1], 40.0)


def test_ema_calculation():
    data = pd.Series([10.0, 10.0, 10.0, 20.0, 20.0])
    ema = TechnicalIndicators.ema(data, period=3)
    assert len(ema) == 5
    assert ema.iloc[-1] > 10.0


def test_rsi_bounds():
    # Uptrend series should yield high RSI
    uptrend = pd.Series([float(i) for i in range(1, 30)])
    rsi_up = TechnicalIndicators.rsi(uptrend, 14)
    assert rsi_up.iloc[-1] > 70.0

    # Downtrend series should yield low RSI
    downtrend = pd.Series([float(30 - i) for i in range(1, 30)])
    rsi_down = TechnicalIndicators.rsi(downtrend, 14)
    assert rsi_down.iloc[-1] < 30.0


def test_bollinger_bands():
    data = pd.Series([100.0 + np.sin(i) * 5 for i in range(50)])
    mid, up, low, pct_b = TechnicalIndicators.bollinger_bands(data, period=20)
    assert len(mid) == 50
    assert (up >= mid).all()
    assert (mid >= low).all()


def test_feature_extractor_returns_valid_dict():
    df = pd.DataFrame({
        "open": [100.0 + i for i in range(30)],
        "high": [105.0 + i for i in range(30)],
        "low": [95.0 + i for i in range(30)],
        "close": [102.0 + i for i in range(30)],
        "volume": [1000.0 for _ in range(30)],
    })
    features = FeatureExtractor.compute_features(df)
    assert "close" in features
    assert "rsi_14" in features
    assert "macd" in features
    assert "bb_percent_b" in features
    assert features["close"] == 131.0
