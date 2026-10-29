"""De Prado's Meta-Labeling secondary classifier.

The primary model / rule-based strategy predicts the trading DIRECTION (-1, +1).
The secondary meta-model predicts the BET SIZE / PROFITABILITY (0 or 1):
Will this trade hit profit before stop loss?
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


class MetaLabeler:
    """Secondary binary classifier filtering primary trade signals."""

    def __init__(self, confidence_threshold: float = 0.55, random_state: int = 42):
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    @staticmethod
    def generate_meta_labels(primary_signals: np.ndarray, realized_returns: np.ndarray) -> np.ndarray:
        """Create binary meta-labels: 1 if trade made profit in primary direction, 0 otherwise.

        primary_signals: array of {-1, 0, +1}
        realized_returns: array of percentage returns for the trade horizon
        """
        primary = np.asarray(primary_signals)
        returns = np.asarray(realized_returns)

        # 1 if (BUY and positive return) or (SHORT and negative return)
        profitable_long = (primary > 0) & (returns > 0)
        profitable_short = (primary < 0) & (returns < 0)
        meta_y = np.where(profitable_long | profitable_short, 1, 0)
        return meta_y

    def fit(self, X: np.ndarray, primary_signals: np.ndarray, realized_returns: np.ndarray) -> MetaLabeler:
        """Fit secondary classifier on features combined with primary signal direction."""
        meta_y = self.generate_meta_labels(primary_signals, realized_returns)
        # Augment features with primary signal column
        primary_col = primary_signals.reshape(-1, 1)
        X_aug = np.hstack([X, primary_col])

        # Filter only to rows where a trade was actually signaled (primary != 0)
        trade_mask = (primary_signals != 0)
        if np.sum(trade_mask) < 20:
            logger.warning("Fewer than 20 trade signals available for meta-labeling. Skipping fit.")
            return self

        X_train = X_aug[trade_mask]
        y_train = meta_y[trade_mask]

        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict_trade_probability(self, X: np.ndarray, primary_signals: np.ndarray) -> np.ndarray:
        """Return probability that the primary trade will be profitable."""
        if not self.is_fitted:
            # If not fitted, do not filter (return 1.0 for all active signals)
            return np.where(primary_signals != 0, 1.0, 0.0)

        primary_col = primary_signals.reshape(-1, 1)
        X_aug = np.hstack([X, primary_col])
        X_scaled = self.scaler.transform(X_aug)

        probs = self.model.predict_proba(X_scaled)[:, 1]
        return probs

    def filter_trades(self, primary_signals: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply meta-model filter: execute trade only if meta probability > confidence_threshold.

        Returns (filtered_signals, meta_probabilities).
        """
        if not self.is_fitted:
            return primary_signals, np.full(len(primary_signals), 1.0)

        probs = self.predict_trade_probability(X, primary_signals)
        filtered = np.where(
            (primary_signals != 0) & (probs >= self.confidence_threshold),
            primary_signals,
            0
        )
        return filtered, probs

    def evaluate(self, X: np.ndarray, primary_signals: np.ndarray, realized_returns: np.ndarray) -> dict[str, float]:
        """Evaluate Brier score and AUC of the meta-labeling model."""
        if not self.is_fitted:
            return {"meta_brier_score": 1.0, "meta_auc": 0.5}

        meta_y = self.generate_meta_labels(primary_signals, realized_returns)
        trade_mask = (primary_signals != 0)
        if np.sum(trade_mask) == 0:
            return {"meta_brier_score": 0.0, "meta_auc": 0.5}

        probs = self.predict_trade_probability(X, primary_signals)[trade_mask]
        y_true = meta_y[trade_mask]

        brier = brier_score_loss(y_true, probs)
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = 0.5
        return {
            "meta_brier_score": round(float(brier), 4),
            "meta_auc": round(float(auc), 4)
        }
