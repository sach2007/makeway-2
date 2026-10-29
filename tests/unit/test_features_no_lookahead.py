"""Unit tests for Phase 2: Assert strictly causal features with ZERO look-ahead leakage."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.features.advanced_features import (
    AdvancedFeaturePipeline, EXPANDED_FEATURE_COLUMNS
)


@pytest.fixture
def sample_universe():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    universe = {}
    for sym in ["AETX", "BETATECH", "GAMMACORP"]:
        np.random.seed(hash(sym) % 10000)
        base = 100.0 + np.cumsum(np.random.normal(0.1, 1.5, 120))
        df = pd.DataFrame({
            "date": dates,
            "open": base + np.random.uniform(-0.5, 0.5, 120),
            "high": base + np.random.uniform(1.0, 3.0, 120),
            "low": base - np.random.uniform(1.0, 3.0, 120),
            "close": base,
            "volume": np.random.uniform(10000, 500000, 120),
        })
        universe[sym] = df
    return universe


def test_zero_lookahead_leakage_guarantee(sample_universe):
    """CRITICAL TEST: Verifies that tampering with future data has ZERO effect on past features.
    Fails immediately if any feature peeks into t+1 or beyond.
    """
    decision_bar_idx = 70

    # 1. Compute features on original data
    orig_matrix = AdvancedFeaturePipeline.compute_multi_symbol_universe_matrix(sample_universe)
    target_date = str(sample_universe["AETX"]["date"].iloc[decision_bar_idx])[:10]
    orig_slice = orig_matrix[orig_matrix["date_str"] == target_date].sort_values("ticker")

    # 2. Maliciously tamper with all future bars (t+1 through t+50)
    tampered_universe = {}
    for sym, df in sample_universe.items():
        tampered_df = df.copy()
        # Multiply future prices by 10x and set massive future volumes
        tampered_df.loc[decision_bar_idx + 1:, "open"] *= 10.0
        tampered_df.loc[decision_bar_idx + 1:, "high"] *= 10.0
        tampered_df.loc[decision_bar_idx + 1:, "low"] *= 10.0
        tampered_df.loc[decision_bar_idx + 1:, "close"] *= 10.0
        tampered_df.loc[decision_bar_idx + 1:, "volume"] *= 100.0
        tampered_universe[sym] = tampered_df

    # 3. Compute features on tampered data
    tampered_matrix = AdvancedFeaturePipeline.compute_multi_symbol_universe_matrix(tampered_universe)
    tampered_slice = tampered_matrix[tampered_matrix["date_str"] == target_date].sort_values("ticker")

    # 4. Assert exact equality across all expanded feature columns at decision_bar
    for col in EXPANDED_FEATURE_COLUMNS:
        orig_vals = orig_slice[col].values
        tamp_vals = tampered_slice[col].values
        np.testing.assert_allclose(
            orig_vals,
            tamp_vals,
            rtol=1e-7,
            atol=1e-7,
            err_msg=f"LOOKAHEAD DETECTED in feature '{col}'! Future tampering altered past feature values."
        )


def test_cross_sectional_ranks_bounded(sample_universe):
    """Verify that cross-sectional ranks strictly lie in [0.0, 1.0]."""
    matrix = AdvancedFeaturePipeline.compute_multi_symbol_universe_matrix(sample_universe)
    rank_cols = [c for c in matrix.columns if c.startswith("cs_rank_")]
    assert len(rank_cols) >= 5

    for col in rank_cols:
        vals = matrix[col].values
        assert (vals >= 0.0).all()
        assert (vals <= 1.0).all()


def test_market_breadth_bounded(sample_universe):
    """Verify that market breadth strictly lies in [0.0, 1.0]."""
    matrix = AdvancedFeaturePipeline.compute_multi_symbol_universe_matrix(sample_universe)
    assert "market_breadth_above_ema20" in matrix.columns
    breadth = matrix["market_breadth_above_ema20"].values
    assert (breadth >= 0.0).all()
    assert (breadth <= 1.0).all()
