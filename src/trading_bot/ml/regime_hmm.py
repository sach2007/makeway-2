"""Hidden Markov Model and probabilistic regime detection.

Classifies asset and market dynamics into:
- State 0: Low-Vol Trending
- State 1: High-Vol Mean-Reverting
- State 2: Crisis / High Volatility Shock
"""
from __future__ import annotations
import logging
from enum import Enum
from typing import Optional, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


class MarketRegimeState(Enum):
    LOW_VOL_TREND = 0
    HIGH_VOL_MEAN_REVERT = 1
    CRISIS_SHOCK = 2


class RegimeHMMDetector:
    """Regime detector using Gaussian HMM on returns and realized volatility."""

    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[GaussianHMM | GaussianMixture] = None
        self.is_fitted = False
        self.state_map: dict[int, MarketRegimeState] = {}

    def _prepare_data(self, returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
        """Construct 2D feature matrix from returns and volatility."""
        ret = np.asarray(returns, dtype=float).flatten()
        vol = np.asarray(volatility, dtype=float).flatten()
        if len(ret) != len(vol):
            raise ValueError(f"Returns length ({len(ret)}) != Volatility length ({len(vol)})")
        
        # Replace NaNs/Infs safely
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.05, neginf=-0.05)
        vol = np.nan_to_num(vol, nan=0.02, posinf=0.10, neginf=0.005)
        vol = np.maximum(vol, 1e-6)

        return np.column_stack([ret, vol])

    def fit(self, returns: np.ndarray, volatility: np.ndarray) -> RegimeHMMDetector:
        """Fit HMM (or GMM fallback) on historical series without lookahead."""
        X = self._prepare_data(returns, volatility)
        if len(X) < 30:
            logger.warning("Insufficient samples (<30) to fit Regime HMM. Skipping.")
            return self

        X_scaled = self.scaler.fit_transform(X)

        if HAS_HMMLEARN:
            try:
                hmm = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=150,
                    random_state=self.random_state
                )
                hmm.fit(X_scaled)
                self.model = hmm
            except Exception as e:
                logger.warning(f"GaussianHMM fit failed: {e}. Falling back to GaussianMixture.")
                gmm = GaussianMixture(
                    n_components=self.n_states,
                    covariance_type="full",
                    random_state=self.random_state
                )
                gmm.fit(X_scaled)
                self.model = gmm
        else:
            gmm = GaussianMixture(
                n_components=self.n_states,
                covariance_type="full",
                random_state=self.random_state
            )
            gmm.fit(X_scaled)
            self.model = gmm

        self._map_states_by_volatility(X[:, 1])
        self.is_fitted = True
        return self

    def _map_states_by_volatility(self, raw_vol: np.ndarray) -> None:
        """Map latent cluster IDs to semantic regimes ordered by ascending volatility."""
        if not self.model:
            return

        # Assign states to training data to compute mean volatility per cluster
        if hasattr(self.model, "predict"):
            preds = self.model.predict(self.scaler.transform(np.column_stack([np.zeros_like(raw_vol), raw_vol])))
        else:
            preds = np.zeros(len(raw_vol), dtype=int)

        mean_vols = {}
        for s in range(self.n_states):
            mask = preds == s
            mean_vols[s] = float(np.mean(raw_vol[mask])) if np.any(mask) else float(s)

        # Sort clusters by mean volatility: lowest vol = Trend, mid vol = Mean Revert, highest = Crisis
        sorted_clusters = sorted(mean_vols.keys(), key=lambda k: mean_vols[k])
        self.state_map = {
            sorted_clusters[0]: MarketRegimeState.LOW_VOL_TREND,
            sorted_clusters[1]: MarketRegimeState.HIGH_VOL_MEAN_REVERT,
        }
        if self.n_states >= 3 and len(sorted_clusters) >= 3:
            self.state_map[sorted_clusters[2]] = MarketRegimeState.CRISIS_SHOCK

    def predict(self, returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
        """Predict regime state IDs [0, 1, 2] for the given series."""
        if not self.is_fitted or self.model is None:
            # Default to LOW_VOL_TREND (0)
            return np.zeros(len(returns), dtype=int)

        X = self._prepare_data(returns, volatility)
        X_scaled = self.scaler.transform(X)
        raw_states = self.model.predict(X_scaled)

        # Map to standard semantic state values
        mapped_states = np.array([self.state_map.get(s, MarketRegimeState.LOW_VOL_TREND).value for s in raw_states])
        return mapped_states

    def predict_proba(self, returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
        """Predict posterior probability distribution over regime states."""
        if not self.is_fitted or self.model is None:
            n = len(returns)
            default = np.zeros((n, self.n_states))
            default[:, 0] = 1.0  # 100% low vol
            return default

        X = self._prepare_data(returns, volatility)
        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
        else:
            probs = np.zeros((len(X), self.n_states))
            preds = self.predict(returns, volatility)
            for i, p in enumerate(preds):
                probs[i, p] = 1.0
            return probs

        # Reorder columns to align with semantic regimes
        ordered_probs = np.zeros_like(probs)
        for cluster_id, semantic_enum in self.state_map.items():
            if cluster_id < probs.shape[1]:
                ordered_probs[:, semantic_enum.value] = probs[:, cluster_id]

        # Normalize row-wise
        row_sums = ordered_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return ordered_probs / row_sums
