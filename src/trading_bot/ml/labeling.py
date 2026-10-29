"""Triple-Barrier Labeling and Sample Weighting (Marcos Lopez de Prado methodology)."""
from __future__ import annotations
import math
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """Implements institutional Triple-Barrier labeling with dynamic ATR barriers and sample weights."""

    def __init__(
        self,
        k_upper: float = 2.0,      # Upper barrier: +k1 * ATR
        k_lower: float = 1.5,      # Lower barrier: -k2 * ATR
        vertical_bars: int = 5     # Vertical barrier: N bars
    ):
        self.k_upper = k_upper
        self.k_lower = k_lower
        self.vertical_bars = vertical_bars

    def apply_triple_barrier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply upper, lower, and vertical barriers to an OHLCV dataframe.
        Expects columns: [date, open, high, low, close, atr_14].
        Returns dataframe with added columns: [label, barrier_touched, barrier_ret, touch_bars, sample_weight].
        """
        if df.empty or len(df) < self.vertical_bars + 10:
            return pd.DataFrame()

        df = df.copy().reset_index(drop=True)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(df)

        # Dynamic barrier bandwidth based on ATR or rolling std
        if "atr_14" in df.columns:
            daily_vol = (df["atr_14"] / (df["close"] + 1e-10)).values
        else:
            log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
            daily_vol = log_ret.rolling(window=14, min_periods=2).std().fillna(0.015).values

        labels = np.zeros(n, dtype=int)
        barrier_touched = [""] * n
        barrier_rets = np.zeros(n, dtype=float)
        touch_bars = np.zeros(n, dtype=int)

        # 1. Evaluate barrier hits for each bar t
        for t in range(n - self.vertical_bars):
            entry_p = close[t]
            vol_t = max(daily_vol[t], 0.005)  # Floor at 0.5%

            upper_p = entry_p * (1.0 + self.k_upper * vol_t)
            lower_p = entry_p * (1.0 - self.k_lower * vol_t)

            hit_label = 0
            hit_reason = "VERTICAL"
            hit_step = self.vertical_bars
            hit_ret = (close[t + self.vertical_bars] - entry_p) / entry_p

            for step in range(1, self.vertical_bars + 1):
                idx = t + step
                bar_high = high[idx]
                bar_low = low[idx]

                # Check if upper barrier touched
                hit_upper = bar_high >= upper_p
                # Check if lower barrier touched
                hit_lower = bar_low <= lower_p

                if hit_upper and not hit_lower:
                    hit_label = 1
                    hit_reason = "UPPER"
                    hit_ret = (upper_p - entry_p) / entry_p
                    hit_step = step
                    break
                elif hit_lower and not hit_upper:
                    hit_label = -1
                    hit_reason = "LOWER"
                    hit_ret = (lower_p - entry_p) / entry_p
                    hit_step = step
                    break
                elif hit_upper and hit_lower:
                    # In case of dual touch on same bar, determine direction by close vs entry
                    if close[idx] >= entry_p:
                        hit_label = 1
                        hit_reason = "UPPER"
                        hit_ret = (upper_p - entry_p) / entry_p
                    else:
                        hit_label = -1
                        hit_reason = "LOWER"
                        hit_ret = (lower_p - entry_p) / entry_p
                    hit_step = step
                    break

            # If vertical barrier reached without hitting upper/lower
            if hit_label == 0:
                hit_label = 1 if hit_ret > 0 else (-1 if hit_ret < 0 else 0)

            labels[t] = hit_label
            barrier_touched[t] = hit_reason
            barrier_rets[t] = hit_ret
            touch_bars[t] = hit_step

        df["label"] = labels
        df["barrier_touched"] = barrier_touched
        df["barrier_ret"] = barrier_rets
        df["touch_bars"] = touch_bars

        # 2. Compute Sample Weights by Label Uniqueness and Return Magnitude
        # Calculate concurrency array: how many label spans cover each bar
        concurrency = np.zeros(n, dtype=float)
        for t in range(n - self.vertical_bars):
            t_end = min(n, t + touch_bars[t])
            concurrency[t:t_end] += 1.0

        concurrency = np.maximum(concurrency, 1.0)

        # Average uniqueness per sample
        uniqueness = np.zeros(n, dtype=float)
        for t in range(n - self.vertical_bars):
            t_end = min(n, t + touch_bars[t])
            uniqueness[t] = np.mean(1.0 / concurrency[t:t_end])

        # Sample weight = Uniqueness * Return Magnitude
        raw_weights = uniqueness * (np.abs(barrier_rets) + 0.001)
        # Normalize weights so sum equals number of samples
        valid_n = n - self.vertical_bars
        if valid_n > 0 and raw_weights[:valid_n].sum() > 0:
            norm_weights = raw_weights / raw_weights[:valid_n].mean()
        else:
            norm_weights = np.ones(n)

        df["sample_weight"] = np.clip(norm_weights, 0.1, 5.0)

        # Drop the last vertical_bars rows where future barriers cannot be evaluated
        valid_df = df.iloc[:-self.vertical_bars].copy().reset_index(drop=True)
        return valid_df
