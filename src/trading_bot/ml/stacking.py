"""Stacking Ensemble Meta-Learner with Out-of-Fold Calibration.

Combines predictions from multiple distinct model architectures:
- Tree-based Gradient Boosters (XGBoost, LightGBM, Random Forest)
- Deep Sequential Architecture (PyTorch GRU)
- Regime state probabilities (Gaussian HMM)
Uses regularized Logistic Regression meta-learner and Platt probability calibration.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from trading_bot.ml.calibration import CalibratedProbabilityEngine

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Stacking meta-learner combining diverse model probability outputs."""

    def __init__(self, regularization_C: float = 0.5, calibration_method: str = "sigmoid"):
        self.regularization_C = regularization_C
        self.meta_learner = LogisticRegression(
            C=regularization_C,
            solver="lbfgs",
            random_state=42
        )
        self.calibrator = CalibratedProbabilityEngine(method=calibration_method)
        self.is_fitted = False
        self.model_weights: Optional[np.ndarray] = None

    def fit(self, oof_predictions: np.ndarray, y_true: np.ndarray) -> StackingEnsemble:
        """Fit meta-learner on out-of-fold probability matrix.

        oof_predictions: shape (N, num_base_models)
        y_true: shape (N,)
        """
        oof = np.asarray(oof_predictions, dtype=float)
        y = np.asarray(y_true, dtype=int)

        if len(oof) < 30 or len(np.unique(y)) < 2:
            logger.warning("Insufficient samples or classes to fit StackingEnsemble. Using simple equal average.")
            self.model_weights = np.ones(oof.shape[1]) / oof.shape[1]
            self.is_fitted = True
            return self

        # Fit regularized meta logistic regression
        self.meta_learner.fit(oof, y)
        raw_meta_probs = self.meta_learner.predict_proba(oof)[:, 1]

        # Fit probability calibrator on meta-learner probabilities
        self.calibrator.fit(raw_meta_probs, y)
        self.model_weights = self.meta_learner.coef_.flatten()
        self.is_fitted = True
        return self

    def predict_proba(self, model_predictions: np.ndarray) -> np.ndarray:
        """Predict calibrated probability of positive outcome from base model predictions.

        model_predictions: shape (N, num_base_models)
        """
        preds = np.asarray(model_predictions, dtype=float)
        if not self.is_fitted:
            # Fallback to simple average across models
            return np.mean(preds, axis=1)

        if hasattr(self.meta_learner, "predict_proba"):
            raw_meta_probs = self.meta_learner.predict_proba(preds)[:, 1]
            return self.calibrator.calibrate(raw_meta_probs)
        else:
            return np.mean(preds, axis=1)

    def predict_conviction(self, model_predictions: np.ndarray, band: float = 0.05) -> np.ndarray:
        """Convert calibrated probabilities to directional conviction in range [-1.0, +1.0].

        Probability > 0.50 + band -> Positive conviction (BUY)
        Probability < 0.50 - band -> Negative conviction (SHORT)
        Within [0.50 - band, 0.50 + band] -> 0.0 (Neutral band)
        """
        probs = self.predict_proba(model_predictions)
        convictions = np.zeros_like(probs)

        # Scale from [0.5 + band, 1.0] -> [0.0, 1.0]
        upper_mask = probs > (0.50 + band)
        convictions[upper_mask] = (probs[upper_mask] - 0.50) * 2.0

        # Scale from [0.0, 0.5 - band] -> [-1.0, 0.0]
        lower_mask = probs < (0.50 - band)
        convictions[lower_mask] = (probs[lower_mask] - 0.50) * 2.0

        return np.clip(convictions, -1.0, 1.0)

    def evaluate(self, model_predictions: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
        """Compute Brier score, ROC-AUC, and log-loss for the stacked ensemble."""
        probs = self.predict_proba(model_predictions)
        y = np.asarray(y_true, dtype=int)

        brier = brier_score_loss(y, probs)
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = 0.50

        return {
            "ensemble_brier_score": round(float(brier), 4),
            "ensemble_roc_auc": round(float(auc), 4),
            "meets_brier_target": bool(brier < 0.20),
            "meta_weights": [round(float(w), 3) for w in (self.model_weights if self.model_weights is not None else [])]
        }
