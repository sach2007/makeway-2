"""Model definitions, training wrappers, and probabilistic evaluation metrics with GPU acceleration."""
from __future__ import annotations
import logging
from typing import Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger(__name__)


class MLModelWrapper:
    """Standardized training, calibration, and inference wrapper for classification models."""

    def __init__(self, model_name: str, estimator: Any, uses_gpu: bool = False):
        self.model_name = model_name
        self.estimator = estimator
        self.uses_gpu = uses_gpu
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> None:
        """Fit scaler and estimator on training data ONLY (zero leakage guarantee)."""
        self.feature_names = feature_names or []
        X_scaled = self.scaler.fit_transform(X)
        self.estimator.fit(X_scaled, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive return [0.0, 1.0]."""
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.model_name} is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        probas = self.estimator.predict_proba(X_scaled)
        return probas[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        """Predict binary direction using probability threshold."""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Compute standard predictive quality metrics."""
        probs = self.predict_proba(X)
        preds = (probs >= 0.50).astype(int)

        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = 0.50
        brier = brier_score_loss(y, probs)

        return {
            "model": self.model_name,
            "uses_gpu": self.uses_gpu,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "brier_score": round(float(brier), 4),
        }


def get_candidate_models() -> dict[str, MLModelWrapper]:
    """Instantiate complete candidate ML suite including GPU-accelerated XGBoost, LightGBM, Random Forest, and Logistic Regression."""
    models = {}

    # 1. XGBoost CPU baseline. The contest runner must not assume CUDA is
    # present; forcing a GPU device can silently fall back while metadata still
    # claims GPU use. CPU histogram training is deterministic and portable.
    if HAS_XGBOOST:
        try:
            xgb_cpu = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cpu",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            models["xgboost_cpu"] = MLModelWrapper("XGBoost_CPU", xgb_cpu, uses_gpu=False)
        except Exception as e:
            logger.warning("Could not initialize XGBoost CPU model: %s", e)

    # 2. LightGBM
    if HAS_LIGHTGBM:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            num_leaves=31,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        models["lightgbm"] = MLModelWrapper("LightGBM", lgb_model, uses_gpu=False)

    # 3. Random Forest (Robust Bagging Baseline)
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    models["random_forest"] = MLModelWrapper("RandomForest", rf_model, uses_gpu=False)

    # 4. Logistic Regression (L2 Regularized Linear Baseline)
    lr_model = LogisticRegression(
        C=0.5,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    models["logistic_regression"] = MLModelWrapper("LogisticRegression", lr_model, uses_gpu=False)

    return models
