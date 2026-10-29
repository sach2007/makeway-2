import numpy as np
import pandas as pd

from trading_bot.features.indicators import FeatureExtractor
from trading_bot.ml.dataset import FEATURE_COLUMNS, TimeSeriesFeaturePipeline


def sample_bars(n=90):
    close = pd.Series(100.0 + np.arange(n) * 0.3 + np.sin(np.arange(n) / 4.0))
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": 1000 + np.arange(n) * 3,
    })


def test_live_feature_extractor_emits_all_training_columns():
    features = FeatureExtractor.compute_features(sample_bars())
    assert set(FEATURE_COLUMNS).issubset(features)
    assert all(np.isfinite(float(features[c])) for c in FEATURE_COLUMNS)


def test_training_features_do_not_backward_fill_future_values():
    pipeline = TimeSeriesFeaturePipeline(target_horizon=3)
    result = pipeline.build_features_for_dataframe(sample_bars())
    assert not result.empty
    assert result[FEATURE_COLUMNS].notna().all().all()
    assert np.isfinite(result[FEATURE_COLUMNS].to_numpy()).all()


def test_chronological_split_keeps_each_date_in_one_partition():
    rows = []
    for ticker in ("AAA", "BBB"):
        for date in pd.date_range("2025-01-01", periods=100, freq="D"):
            rows.append({"ticker": ticker, "date": date, "x": 1})
    df = pd.DataFrame(rows)
    train, val, test = TimeSeriesFeaturePipeline.chronological_split(df, purge_days=3)
    sets = [set(pd.to_datetime(x["date"])) for x in (train, val, test)]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    for part in (train, val, test):
        counts = part.groupby("date")["ticker"].nunique()
        assert (counts <= 2).all()
