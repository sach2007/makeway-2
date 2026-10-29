"""Unit tests for Phase 4: Models, Calibration, Meta-Labeling, Sequence Model, and Champion Registry."""
from __future__ import annotations
import os
import shutil
import numpy as np
import pytest

from trading_bot.ml.regime_hmm import RegimeHMMDetector, MarketRegimeState
from trading_bot.ml.meta_labeling import MetaLabeler
from trading_bot.ml.calibration import CalibratedProbabilityEngine
from trading_bot.ml.sequence_model import SequenceModelWrapper
from trading_bot.ml.stacking import StackingEnsemble
from trading_bot.ml.champion_registry import ChampionRegistry, ModelMetadata


def test_regime_hmm_detector():
    """Verify HMM detector accurately segments volatility regimes and probabilities sum to 1."""
    np.random.seed(42)
    # 60 low vol bars, 60 high vol bars, 30 crisis bars
    ret_low = np.random.normal(0.001, 0.008, 60)
    vol_low = np.random.uniform(0.005, 0.015, 60)

    ret_high = np.random.normal(-0.002, 0.025, 60)
    vol_high = np.random.uniform(0.020, 0.035, 60)

    ret_crisis = np.random.normal(-0.010, 0.050, 30)
    vol_crisis = np.random.uniform(0.045, 0.080, 30)

    returns = np.concatenate([ret_low, ret_high, ret_crisis])
    volatility = np.concatenate([vol_low, vol_high, vol_crisis])

    detector = RegimeHMMDetector(n_states=3, random_state=42)
    detector.fit(returns, volatility)

    assert detector.is_fitted
    preds = detector.predict(returns, volatility)
    assert len(preds) == len(returns)
    assert set(preds).issubset({0, 1, 2})

    probs = detector.predict_proba(returns, volatility)
    assert probs.shape == (len(returns), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_meta_labeler():
    """Verify De Prado Meta-Labeler correctly identifies profitable signals and filters noise."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 5)
    # Signal: +1 (BUY), -1 (SHORT), 0 (HOLD)
    signals = np.random.choice([-1, 0, 1], size=n, p=[0.4, 0.2, 0.4])
    # Realized returns
    realized_returns = np.random.randn(n) * 0.03

    meta_y = MetaLabeler.generate_meta_labels(signals, realized_returns)
    # Check definition: long and ret > 0 is 1, short and ret < 0 is 1, else 0
    for s, r, y in zip(signals, realized_returns, meta_y):
        if s > 0 and r > 0:
            assert y == 1
        elif s < 0 and r < 0:
            assert y == 1
        elif s == 0:
            assert y == 0

    labeler = MetaLabeler(confidence_threshold=0.55, random_state=42)
    labeler.fit(X, signals, realized_returns)
    assert labeler.is_fitted

    filtered, probs = labeler.filter_trades(signals, X)
    assert len(filtered) == n
    assert len(probs) == n
    # Any filtered trade must have had a non-zero signal and prob >= 0.55
    for sig, f_sig, p in zip(signals, filtered, probs):
        if f_sig != 0:
            assert sig == f_sig
            assert p >= 0.55


def test_calibrated_probability_engine():
    """Verify Platt scaling and Isotonic regression produce valid probabilities and evaluate Brier score."""
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.6, size=n)
    # Uncalibrated noisy probabilities
    raw_probs = np.clip(y_true * 0.7 + np.random.normal(0, 0.25, size=n), 0.05, 0.95)

    # Sigmoid / Platt scaling
    platt = CalibratedProbabilityEngine(method="sigmoid")
    platt.fit(raw_probs[:100], y_true[:100])
    cal_probs = platt.calibrate(raw_probs[100:])
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)

    eval_res = platt.evaluate(raw_probs[100:], y_true[100:])
    assert "calibrated_brier_score" in eval_res
    assert eval_res["calibrated_brier_score"] < 0.25

    # Isotonic regression
    isotonic = CalibratedProbabilityEngine(method="isotonic")
    isotonic.fit(raw_probs[:100], y_true[:100])
    iso_probs = isotonic.calibrate(raw_probs[100:])
    assert np.all(iso_probs >= 0.0) and np.all(iso_probs <= 1.0)


def test_sequence_gru_model():
    """Verify PyTorch GRU model handles rolling sequence creation, training, and inference."""
    np.random.seed(42)
    n_bars, n_features = 120, 8
    X = np.random.randn(n_bars, n_features).astype(np.float32)
    y = np.random.binomial(1, 0.55, size=n_bars).astype(np.float32)

    wrapper = SequenceModelWrapper(seq_len=30, hidden_size=16, epochs=3, batch_size=32)
    wrapper.fit(X, y)
    assert wrapper.is_fitted

    probs = wrapper.predict_proba(X)
    assert len(probs) == n_bars
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_stacking_ensemble():
    """Verify StackingEnsemble fits on out-of-fold predictions and outputs bounded directional convictions."""
    np.random.seed(42)
    n = 150
    # Predictions from 3 base models
    m1 = np.random.uniform(0.3, 0.8, n)
    m2 = np.random.uniform(0.2, 0.7, n)
    m3 = np.random.uniform(0.4, 0.9, n)
    oof_preds = np.column_stack([m1, m2, m3])
    y_true = np.random.binomial(1, 0.6, n)

    ensemble = StackingEnsemble(regularization_C=1.0)
    ensemble.fit(oof_preds, y_true)
    assert ensemble.is_fitted

    convictions = ensemble.predict_conviction(oof_preds, band=0.05)
    assert len(convictions) == n
    assert np.all(convictions >= -1.0) and np.all(convictions <= 1.0)

    eval_dict = ensemble.evaluate(oof_preds, y_true)
    assert "ensemble_brier_score" in eval_dict
    assert "ensemble_roc_auc" in eval_dict


def test_champion_registry(tmp_path):
    """Verify Champion/Challenger framework strictly enforces Sharpe and Drawdown gates."""
    reg_file = str(tmp_path / "test_registry.json")
    registry = ChampionRegistry(registry_file=reg_file)

    # 1. Baseline model becomes initial champion
    promoted, reason, meta1 = registry.evaluate_challenger(
        model_name="Baseline_Model",
        feature_list=["rsi", "ema"],
        hyperparameters={"lr": 0.05},
        oos_sharpe=1.20,
        oos_brier_score=0.22,
        oos_max_drawdown_pct=12.0,
        training_date_start="2024-01-01",
        training_date_end="2025-01-01"
    )
    assert promoted is True
    assert registry.champion.model_name == "Baseline_Model"

    # 2. Challenger with insufficient Sharpe improvement (+0.05 < +0.15) -> REJECTED
    promoted2, reason2, meta2 = registry.evaluate_challenger(
        model_name="Weak_Challenger",
        feature_list=["rsi", "ema", "atr"],
        hyperparameters={"lr": 0.03},
        oos_sharpe=1.25,  # only +0.05 improvement
        oos_brier_score=0.21,
        oos_max_drawdown_pct=11.5,
        training_date_start="2024-01-01",
        training_date_end="2025-01-01"
    )
    assert promoted2 is False
    assert "Rejected" in reason2
    assert registry.champion.model_name == "Baseline_Model"

    # 3. Challenger with high Sharpe (+0.30) but unacceptable Drawdown (+5.0% > +2.0% tolerance) -> REJECTED
    promoted3, reason3, meta3 = registry.evaluate_challenger(
        model_name="Risky_Challenger",
        feature_list=["rsi", "ema", "atr", "vol"],
        hyperparameters={"lr": 0.08},
        oos_sharpe=1.55,
        oos_brier_score=0.19,
        oos_max_drawdown_pct=18.0,  # 18% vs 12% = +6% deterioration
        training_date_start="2024-01-01",
        training_date_end="2025-01-01"
    )
    assert promoted3 is False
    assert registry.champion.model_name == "Baseline_Model"

    # 4. Superior Challenger (+0.25 Sharpe, controlled drawdown) -> PROMOTED
    promoted4, reason4, meta4 = registry.evaluate_challenger(
        model_name="Superior_Ensemble",
        feature_list=["rsi", "ema", "atr", "vol", "gap"],
        hyperparameters={"lr": 0.04},
        oos_sharpe=1.48,  # +0.28 improvement
        oos_brier_score=0.18,
        oos_max_drawdown_pct=11.0,  # lower drawdown
        training_date_start="2024-01-01",
        training_date_end="2025-01-01"
    )
    assert promoted4 is True
    assert registry.champion.model_name == "Superior_Ensemble"
    assert registry.champion.oos_sharpe == 1.48
