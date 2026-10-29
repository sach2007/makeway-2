"""Probability calibration engine using Platt scaling and Isotonic regression.

Ensures predicted probabilities reflect true empirical odds and achieve Brier score < 0.20.
"""
from __future__ import annotations
import logging
from typing import Literal
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class CalibratedProbabilityEngine:
    """Calibrates model probability outputs via Platt scaling or Isotonic regression."""

    def __init__(self, method: Literal["sigmoid", "isotonic"] = "sigmoid"):
        self.method = method
        self.calibrator = (
            LogisticRegression(C=1.0, solver="lbfgs")
            if method == "sigmoid"
            else IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        )
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> CalibratedProbabilityEngine:
        """Fit calibration curve on validation/out-of-fold probabilities."""
        probs = np.asarray(raw_probs, dtype=float).clip(1e-6, 1.0 - 1e-6)
        y = np.asarray(y_true, dtype=int)

        if len(probs) < 20 or len(np.unique(y)) < 2:
            logger.warning("Insufficient samples or classes to calibrate probabilities. Bypassing.")
            return self

        if self.method == "sigmoid":
            # Platt scaling: fit univariate logistic regression on logit(p)
            eps = 1e-6
            logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
            self.calibrator.fit(logits, y)
        else:
            self.calibrator.fit(probs, y)

        self.is_fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities."""
        probs = np.asarray(raw_probs, dtype=float).clip(1e-6, 1.0 - 1e-6)
        if not self.is_fitted:
            return probs

        if self.method == "sigmoid":
            logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(logits)[:, 1]
        else:
            calibrated = self.calibrator.predict(probs)

        return np.clip(calibrated, 0.01, 0.99)

    def evaluate(self, raw_probs: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Compute Brier score before and after calibration."""
        probs = np.asarray(raw_probs, dtype=float)
        y = np.asarray(y_true, dtype=int)

        raw_brier = brier_score_loss(y, probs)
        calibrated_probs = self.calibrate(probs)
        cal_brier = brier_score_loss(y, calibrated_probs)

        return {
            "method": self.method,
            "raw_brier_score": round(float(raw_brier), 4),
            "calibrated_brier_score": round(float(cal_brier), 4),
            "meets_target": bool(cal_brier < 0.20)
        }
