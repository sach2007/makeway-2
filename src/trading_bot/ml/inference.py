"""Lightweight, low-latency CPU inference engine for real-time model predictions."""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np

from trading_bot.ml.dataset import FEATURE_COLUMNS
from trading_bot.ml.models import MLModelWrapper

logger = logging.getLogger(__name__)


class MLInferenceEngine:
    """Loads champion model and provides sub-millisecond CPU predictions during live simulation."""

    def __init__(self, models_dir: str = "models", enabled: bool = True):
        self.models_dir = Path(models_dir)
        self.enabled = enabled
        self.model: Optional[MLModelWrapper] = None
        self.feature_names: list[str] = FEATURE_COLUMNS
        self.is_loaded = False
        self._last_inference_ms: float = 0.0
        self._prediction_count: int = 0
        self._feature_schema_valid: bool = True

        if self.enabled:
            self.load_model()

    def load_model(self) -> bool:
        """Load trained champion model and feature definitions from disk."""
        model_path = self.models_dir / "champion_model.joblib"
        features_path = self.models_dir / "feature_names.json"

        if not model_path.exists():
            logger.info(f"No trained ML model found at {model_path}. ML features disabled.")
            self.is_loaded = False
            return False

        try:
            self.model = joblib.load(model_path)
            if features_path.exists():
                with open(features_path, "r", encoding="utf-8") as f:
                    self.feature_names = json.load(f)
            expected = set(FEATURE_COLUMNS)
            actual = set(self.feature_names)
            self._feature_schema_valid = self.feature_names == FEATURE_COLUMNS and actual == expected
            if not self._feature_schema_valid:
                logger.error("ML model feature schema mismatch. Expected training schema %s; artifact schema %s. ML predictions disabled until retrained.", FEATURE_COLUMNS, self.feature_names)
                self.is_loaded = False
                return False
            # Prefer estimator's trained feature order when available, but only if
            # it is the exact shared schema persisted by the training pipeline.
            self.is_loaded = True
            logger.info(f"Loaded ML champion model [{self.model.model_name}] successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load ML model from {model_path}: {e}")
            self.is_loaded = False
            return False

    def predict_probability(self, feature_dict: dict[str, Any]) -> float:
        """Generate probability of positive forward return [0.0, 1.0].
        Returns neutral 0.50 if model is not loaded or disabled.
        """
        if not self.enabled or not self.is_loaded or self.model is None or not self._feature_schema_valid:
            return 0.50

        t0 = time.perf_counter()
        try:
            # Build input vector matching exact training feature names
            missing = [col for col in self.feature_names if col not in feature_dict]
            if missing:
                logger.warning("Skipping ML prediction: live feature vector missing required columns: %s", missing)
                return 0.50
            row = []
            for col in self.feature_names:
                val = feature_dict[col]
                try:
                    val_float = float(val) if val is not None else 0.0
                    if np.isnan(val_float) or np.isinf(val_float):
                        val_float = 0.0
                except (ValueError, TypeError):
                    val_float = 0.0
                row.append(val_float)

            X = np.array([row], dtype=float)
            prob = float(self.model.predict_proba(X)[0])
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
            self._prediction_count += 1
            return max(0.0, min(1.0, prob))

        except Exception as e:
            logger.warning(f"Inference error: {e}. Falling back to 0.50.")
            return 0.50

    def predict_conviction(self, feature_dict: dict[str, Any]) -> float:
        """Map return probability to normalized conviction in [-1.0, 1.0].
        P = 0.5 -> Conviction = 0.0
        P = 0.8 -> Conviction = +0.60
        P = 0.2 -> Conviction = -0.60
        """
        prob = self.predict_probability(feature_dict)
        return (prob - 0.50) * 2.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "is_loaded": self.is_loaded,
            "model_name": self.model.model_name if self.model else "None",
            "last_latency_ms": round(self._last_inference_ms, 3),
            "predictions_count": self._prediction_count,
        }
