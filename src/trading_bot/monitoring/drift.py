"""Statistical Feature Drift, Prediction Drift (PSI), and Rolling Hit-Rate Monitoring.

Provides:
1. Population Stability Index (PSI) calculation across quantile bins.
2. DriftMonitor for early warning (PSI > 0.10) and emergency kill switch (PSI > 0.25).
3. Rolling HitRateTracker: throttles size to 50% if hit-rate < 35%, halts if < 25%.
"""
from __future__ import annotations
import logging
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DriftStatus(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL_HALT = "CRITICAL_HALT"


def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> float:
    """Compute Population Stability Index (PSI) between baseline and current distributions."""
    base = np.asarray(baseline, dtype=float).flatten()
    curr = np.asarray(current, dtype=float).flatten()

    base = base[~np.isnan(base)]
    curr = curr[~np.isnan(curr)]

    if len(base) < 20 or len(curr) < 20:
        return 0.0

    # Determine quantile bin edges from baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(base, quantiles)
    # Deduplicate bin edges in case of repeated values
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate bucket counts
    base_counts, _ = np.histogram(base, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    # Convert to fractions with smoothing epsilon
    base_pct = (base_counts / len(base)) + epsilon
    curr_pct = (curr_counts / len(curr)) + epsilon

    # Normalize fractions
    base_pct /= np.sum(base_pct)
    curr_pct /= np.sum(curr_pct)

    # PSI = sum((Actual - Expected) * ln(Actual / Expected))
    psi_val = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(max(0.0, psi_val))


class DriftMonitor:
    """Monitors live incoming feature streams and model predictions against training baselines."""

    def __init__(self, warning_threshold: float = 0.10, halt_threshold: float = 0.25):
        self.warning_threshold = warning_threshold
        self.halt_threshold = halt_threshold
        self.feature_baselines: dict[str, np.ndarray] = {}
        self.prediction_baseline: Optional[np.ndarray] = None
        self.live_buffers: dict[str, deque] = {}
        self.pred_buffer = deque(maxlen=200)

    def register_baseline(self, feature_name: str, baseline_values: np.ndarray) -> None:
        """Register baseline values from training dataset."""
        self.feature_baselines[feature_name] = np.asarray(baseline_values, dtype=float).copy()
        self.live_buffers[feature_name] = deque(maxlen=200)

    def register_prediction_baseline(self, baseline_probs: np.ndarray) -> None:
        """Register training/validation prediction probability distribution."""
        self.prediction_baseline = np.asarray(baseline_probs, dtype=float).copy()

    def record_live_observation(self, feature_dict: Dict[str, float], prediction_prob: Optional[float] = None) -> None:
        """Append live observation to rolling monitoring window."""
        for name, val in feature_dict.items():
            if name in self.live_buffers:
                self.live_buffers[name].append(val)
        if prediction_prob is not None:
            self.pred_buffer.append(prediction_prob)

    def check_drift(self) -> Tuple[DriftStatus, Dict[str, float], Optional[str]]:
        """Evaluate PSI across all registered features and predictions.
        Returns: (status, psi_dict, halt_reason_if_any)
        """
        psi_scores = {}
        max_psi = 0.0
        worst_metric = ""

        # Check features
        for name, baseline in self.feature_baselines.items():
            buffer = self.live_buffers.get(name)
            if buffer and len(buffer) >= 30:
                psi = calculate_psi(baseline, np.array(buffer))
                psi_scores[name] = round(psi, 4)
                if psi > max_psi:
                    max_psi = psi
                    worst_metric = f"feature '{name}'"

        # Check predictions
        if self.prediction_baseline is not None and len(self.pred_buffer) >= 30:
            pred_psi = calculate_psi(self.prediction_baseline, np.array(self.pred_buffer))
            psi_scores["prediction_prob"] = round(pred_psi, 4)
            if pred_psi > max_psi:
                max_psi = pred_psi
                worst_metric = "prediction probabilities"

        if max_psi >= self.halt_threshold:
            reason = f"CRITICAL DRIFT HALT: {worst_metric} PSI={max_psi:.3f} >= {self.halt_threshold}"
            logger.critical(reason)
            return DriftStatus.CRITICAL_HALT, psi_scores, reason
        elif max_psi >= self.warning_threshold:
            logger.warning(f"Drift Warning: {worst_metric} PSI={max_psi:.3f} >= {self.warning_threshold}")
            return DriftStatus.WARNING, psi_scores, None
        else:
            return DriftStatus.HEALTHY, psi_scores, None


class HitRateTracker:
    """Tracks rolling 20-trade hit rate and enforces automatic de-risking."""

    def __init__(self, window_size: int = 20, throttle_threshold: float = 0.35, halt_threshold: float = 0.25):
        self.window_size = window_size
        self.throttle_threshold = throttle_threshold
        self.halt_threshold = halt_threshold
        self.recent_trades = deque(maxlen=window_size)  # 1 for win, 0 for loss

    def record_trade_result(self, net_pnl: float) -> None:
        """Record trade outcome."""
        self.recent_trades.append(1 if net_pnl > 0 else 0)

    @property
    def hit_rate(self) -> float:
        """Calculate rolling win rate."""
        if not self.recent_trades:
            return 0.50
        return float(np.mean(self.recent_trades))

    def get_sizing_multiplier(self) -> Tuple[float, bool, Optional[str]]:
        """Return (size_multiplier, should_halt, reason)."""
        if len(self.recent_trades) < 10:
            return 1.0, False, None  # Insufficient trades to judge

        hr = self.hit_rate
        if hr < self.halt_threshold:
            reason = f"Hit-rate collapsed to {hr * 100:.1f}% (< {self.halt_threshold * 100:.1f}%). Emergency Halt!"
            logger.critical(reason)
            return 0.0, True, reason
        elif hr < self.throttle_threshold:
            reason = f"Hit-rate degraded to {hr * 100:.1f}% (< {self.throttle_threshold * 100:.1f}%). De-risking: 50% size."
            logger.warning(reason)
            return 0.50, False, reason
        else:
            return 1.0, False, None
