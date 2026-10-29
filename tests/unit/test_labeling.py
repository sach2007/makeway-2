"""Unit tests for Phase 3: Triple-Barrier labeling and sample uniqueness weights."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.ml.labeling import TripleBarrierLabeler


@pytest.fixture
def trending_ohlcv():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    # Sharp upward trend: 100 to 160
    base = np.linspace(100, 160, 60)
    df = pd.DataFrame({
        "date": dates,
        "open": base,
        "high": base + 2.0,
        "low": base - 1.0,
        "close": base + 1.0,
        "atr_14": [2.0] * 60
    })
    return df


def test_triple_barrier_upward_trend_generates_positive_labels(trending_ohlcv):
    labeler = TripleBarrierLabeler(k_upper=1.5, k_lower=1.5, vertical_bars=5)
    labeled_df = labeler.apply_triple_barrier(trending_ohlcv)

    assert not labeled_df.empty
    assert len(labeled_df) == len(trending_ohlcv) - 5
    assert "label" in labeled_df.columns
    assert "sample_weight" in labeled_df.columns
    assert "barrier_touched" in labeled_df.columns

    # In a sharp upward trend, upper barrier should be touched predominantly
    upper_hits = (labeled_df["label"] == 1).sum()
    assert upper_hits > len(labeled_df) * 0.70


def test_sample_weights_are_positive_and_normalized(trending_ohlcv):
    labeler = TripleBarrierLabeler(k_upper=2.0, k_lower=2.0, vertical_bars=5)
    labeled_df = labeler.apply_triple_barrier(trending_ohlcv)

    weights = labeled_df["sample_weight"].values
    assert (weights > 0.0).all()
    # Average weight should be near 1.0
    assert abs(np.mean(weights) - 1.0) < 0.20


def test_downward_trend_generates_negative_labels():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    # Sharp downward trend: 160 down to 100
    base = np.linspace(160, 100, 60)
    df = pd.DataFrame({
        "date": dates,
        "open": base,
        "high": base + 1.0,
        "low": base - 2.0,
        "close": base - 1.0,
        "atr_14": [2.0] * 60
    })
    labeler = TripleBarrierLabeler(k_upper=1.5, k_lower=1.5, vertical_bars=5)
    labeled_df = labeler.apply_triple_barrier(df)

    lower_hits = (labeled_df["label"] == -1).sum()
    assert lower_hits > len(labeled_df) * 0.70
