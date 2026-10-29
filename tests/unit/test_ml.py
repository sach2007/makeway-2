"""Unit tests for ML feature pipeline, model training, and inference engine."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.ml.dataset import FEATURE_COLUMNS, TimeSeriesFeaturePipeline
from trading_bot.ml.inference import MLInferenceEngine
from trading_bot.ml.models import MLModelWrapper, get_candidate_models


@pytest.fixture
def synthetic_ohlcv():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    base_price = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, 100))
    df = pd.DataFrame({
        "date": [str(d.date()) for d in dates],
        "open": base_price,
        "high": base_price + 2.0,
        "low": base_price - 2.0,
        "close": base_price + 0.5,
        "volume": [10000.0 + i * 10 for i in range(100)],
    })
    return df


def test_feature_pipeline_no_lookahead(synthetic_ohlcv):
    pipeline = TimeSeriesFeaturePipeline(target_horizon=3, cost_hurdle_bps=10.0)
    dataset = pipeline.build_features_for_dataframe(synthetic_ohlcv)

    assert not dataset.empty
    assert "target" in dataset.columns
    assert "forward_return" in dataset.columns
    # Check that all feature columns are present and non-empty
    for col in FEATURE_COLUMNS:
        assert col in dataset.columns
        assert not dataset[col].isna().any()

    # The dataset length should be 100 - 50 (warmup) - 3 (horizon) = 47 rows
    assert len(dataset) == 47


def test_candidate_models_fit_and_predict():
    X = np.random.normal(0, 1, (100, len(FEATURE_COLUMNS)))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    candidates = get_candidate_models()
    for name, model in candidates.items():
        model.fit(X, y, feature_names=FEATURE_COLUMNS)
        assert model.is_fitted

        probs = model.predict_proba(X[:10])
        assert len(probs) == 10
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "brier_score" in metrics


def test_chronological_split(synthetic_ohlcv):
    pipeline = TimeSeriesFeaturePipeline(target_horizon=2)
    dataset = pipeline.build_features_for_dataframe(synthetic_ohlcv)

    train_df, val_df, test_df = TimeSeriesFeaturePipeline.chronological_split(dataset, 0.60, 0.20)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Ensure dates strictly follow chronological ordering
    assert train_df["date"].iloc[-1] <= val_df["date"].iloc[0]
    assert val_df["date"].iloc[-1] <= test_df["date"].iloc[0]


def test_inference_engine_handles_nans_and_missing_keys():
    engine = MLInferenceEngine(models_dir="non_existent_dir", enabled=True)
    # When no model is loaded, engine should safely return neutral 0.50 probability
    prob = engine.predict_probability({"ret_1d": np.nan, "rsi_14_norm": None})
    assert prob == 0.50
    conviction = engine.predict_conviction({})
    assert conviction == 0.0
