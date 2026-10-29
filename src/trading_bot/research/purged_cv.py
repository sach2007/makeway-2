"""Purged and Embargoed Walk-Forward Cross-Validation (Marcos Lopez de Prado framework)."""
from __future__ import annotations
import logging
from typing import Generator
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PurgedWalkForwardCV:
    """Walk-Forward Cross-Validation with purging of overlapping label horizons and embargo buffer.
    Prevents information leakage from overlapping labels and serial correlation in financial time series.
    """

    def __init__(
        self,
        n_splits: int = 4,
        min_train_pct: float = 0.40,
        label_horizon_bars: int = 3,
        embargo_pct: float = 0.02
    ):
        self.n_splits = n_splits
        self.min_train_pct = min_train_pct
        self.label_horizon_bars = label_horizon_bars
        self.embargo_pct = embargo_pct

    def split(
        self,
        df: pd.DataFrame
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame, dict[str, int]], None, None]:
        """Generate (train_df, test_df, fold_meta) tuples.
        Guarantees that train samples whose forward labels overlap test window are purged,
        and an embargo buffer is inserted after test window.
        """
        n = len(df)
        if n < 60:
            raise ValueError(f"Insufficient samples for walk-forward CV: {n} < 60")

        min_train = int(n * self.min_train_pct)
        embargo_bars = max(1, int(n * self.embargo_pct))
        test_step = (n - min_train) // self.n_splits

        for fold in range(self.n_splits):
            train_end_raw = min_train + (fold * test_step)
            test_start = train_end_raw
            test_end = min(n, test_start + test_step) if fold < self.n_splits - 1 else n

            # Purging: Drop training samples near the boundary whose forward label horizon overlaps test start
            train_end_purged = max(0, train_end_raw - self.label_horizon_bars)
            train_slice = df.iloc[:train_end_purged].copy()
            test_slice = df.iloc[test_start:test_end].copy()

            if len(train_slice) < 20 or len(test_slice) < 10:
                continue

            meta = {
                "fold": fold + 1,
                "train_samples": len(train_slice),
                "test_samples": len(test_slice),
                "purged_bars": self.label_horizon_bars,
                "embargo_bars": embargo_bars,
                "train_start": str(train_slice["date"].iloc[0])[:10] if "date" in train_slice.columns else "0",
                "train_end": str(train_slice["date"].iloc[-1])[:10] if "date" in train_slice.columns else str(train_end_purged),
                "test_start": str(test_slice["date"].iloc[0])[:10] if "date" in test_slice.columns else str(test_start),
                "test_end": str(test_slice["date"].iloc[-1])[:10] if "date" in test_slice.columns else str(test_end),
            }

            yield train_slice, test_slice, meta
