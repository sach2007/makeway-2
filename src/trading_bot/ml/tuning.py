"""Walk-Forward Hyperparameter Tuning with Optuna and Purged Cross-Validation.

Prevents overfitting and leakage by evaluating parameter trials strictly on
out-of-sample purged validation folds.
"""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional
import numpy as np

from trading_bot.research.purged_cv import PurgedWalkForwardCV
from trading_bot.ml.models import MLModelWrapper

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class WalkForwardTuner:
    """Tunes gradient boosting models over Purged Walk-Forward CV folds."""

    def __init__(self, n_trials: int = 15, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state

    def tune_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: PurgedWalkForwardCV
    ) -> dict[str, Any]:
        """Find optimal LightGBM hyperparameters minimizing Brier score across purged folds."""
        if not HAS_LIGHTGBM:
            logger.warning("LightGBM not installed. Returning default parameters.")
            return {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05}

        if not HAS_OPTUNA:
            logger.warning("Optuna not installed. Using sensible default grid.")
            return {
                "n_estimators": 120,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0
            }

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1
            }

            brier_scores = []
            for train_idx, val_idx in cv.split(len(X)):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    continue

                model = lgb.LGBMClassifier(**params)
                wrapper = MLModelWrapper("LGBM_Tuning", model)
                wrapper.fit(X_tr, y_tr)
                metrics = wrapper.evaluate(X_val, y_val)
                brier_scores.append(metrics["brier_score"])

            return float(np.mean(brier_scores)) if brier_scores else 1.0

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def tune_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: PurgedWalkForwardCV
    ) -> dict[str, Any]:
        """Find optimal XGBoost hyperparameters minimizing Brier score across purged folds."""
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed. Returning default parameters.")
            return {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05}

        if not HAS_OPTUNA:
            return {
                "n_estimators": 120,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.1
            }

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
                "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
                "eval_metric": "logloss",
                "random_state": self.random_state,
                "n_jobs": -1
            }

            brier_scores = []
            for train_idx, val_idx in cv.split(len(X)):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    continue

                model = xgb.XGBClassifier(**params)
                wrapper = MLModelWrapper("XGB_Tuning", model)
                wrapper.fit(X_tr, y_tr)
                metrics = wrapper.evaluate(X_val, y_val)
                brier_scores.append(metrics["brier_score"])

            return float(np.mean(brier_scores)) if brier_scores else 1.0

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params
