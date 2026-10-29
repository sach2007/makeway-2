"""Time-series dataset construction, feature engineering, and walk-forward splitting for ML."""
from __future__ import annotations
import logging
from typing import Generator, Optional
import numpy as np
import pandas as pd

from trading_bot.features.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ema_5_20_ratio",
    "sma_50_ratio",
    "rsi_14_norm",
    "macd_hist_ratio",
    "bb_percent_b",
    "bb_bandwidth",
    "atr_ratio",
    "volatility_10d",
    "volatility_20d",
    "volume_ratio",
    "high_low_ratio",
    "close_open_ratio",
]


class TimeSeriesFeaturePipeline:
    """Builds strictly causal, anti-lookahead feature matrices from historical OHLCV data."""

    def __init__(self, target_horizon: int = 3, cost_hurdle_bps: float = 20.0):
        self.target_horizon = target_horizon
        self.cost_hurdle = cost_hurdle_bps / 10000.0  # e.g. 0.0020 for 20 bps

    def build_features_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-aligned technical features for a single stock dataframe.
        Input df must have: [date, open, high, low, close, volume]
        """
        if df.empty or len(df) < 55:
            return pd.DataFrame()

        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        close = df["close"].astype(float)
        open_p = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        feats = pd.DataFrame(index=df.index)
        feats["date"] = df["date"]
        feats["close"] = close

        # 1. Price Momentum / Return Features (strictly backward looking)
        feats["ret_1d"] = close.pct_change(1)
        feats["ret_3d"] = close.pct_change(3)
        feats["ret_5d"] = close.pct_change(5)
        feats["ret_10d"] = close.pct_change(10)

        # 2. Moving Average Divergence
        ema5 = TechnicalIndicators.ema(close, 5)
        ema20 = TechnicalIndicators.ema(close, 20)
        feats["ema_5_20_ratio"] = (ema5 - ema20) / (close + 1e-10)

        sma50 = TechnicalIndicators.sma(close, 50)
        feats["sma_50_ratio"] = (close - sma50) / (sma50 + 1e-10)

        # 3. Oscillators & Momentum
        rsi = TechnicalIndicators.rsi(close, 14)
        feats["rsi_14_norm"] = (rsi - 50.0) / 50.0  # [-1.0, 1.0]

        _, _, macd_hist = TechnicalIndicators.macd(close, 12, 26, 9)
        feats["macd_hist_ratio"] = macd_hist / (close + 1e-10)

        # 4. Volatility & Bands
        mid, up, low_b, pct_b = TechnicalIndicators.bollinger_bands(close, 20)
        feats["bb_percent_b"] = pct_b
        feats["bb_bandwidth"] = (up - low_b) / (mid.replace(0.0, np.nan) + 1e-10)

        atr = TechnicalIndicators.atr(high, low, close, 14)
        feats["atr_ratio"] = atr / (close + 1e-10)

        feats["volatility_10d"] = TechnicalIndicators.rolling_volatility(close, 10)
        feats["volatility_20d"] = TechnicalIndicators.rolling_volatility(close, 20)

        # 5. Volume & Intraday Structure
        vol_mean20 = volume.rolling(window=20, min_periods=1).mean()
        feats["volume_ratio"] = volume / (vol_mean20 + 1e-10)

        feats["high_low_ratio"] = (high - low) / (close + 1e-10)
        feats["close_open_ratio"] = (close - open_p) / (open_p + 1e-10)

        # 6. Target Construction (Future Horizon $H$)
        # Target: Forward return over next H days
        forward_price = close.shift(-self.target_horizon)
        forward_return = (forward_price - close) / close
        feats["forward_return"] = forward_return

        # Binary classification target: 1 if forward return exceeds hurdle, else 0
        feats["target"] = (forward_return > self.cost_hurdle).astype(int)

        # Drop warm-up rows (first 50 bars)
        feats = feats.iloc[50:].reset_index(drop=True)

        # Drop the last H rows where future target is not yet known
        feats = feats.iloc[:-self.target_horizon].reset_index(drop=True)

        # Causal handling only: never backward-fill from a future observation.
        # Warm-up rows are already removed; any remaining invalid feature row is
        # excluded rather than silently manufacturing a feature vector.
        feats[FEATURE_COLUMNS] = feats[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        feats = feats.dropna(subset=FEATURE_COLUMNS + ["forward_return", "target"]).reset_index(drop=True)
        return feats

    def build_multi_symbol_dataset(self, symbol_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine feature matrices from multiple stocks into a single training dataframe."""
        all_dfs = []
        for ticker, df in symbol_dfs.items():
            f_df = self.build_features_for_dataframe(df)
            if not f_df.empty:
                f_df["ticker"] = ticker
                all_dfs.append(f_df)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        # Sort chronologically by date to ensure strict temporal order
        combined = combined.sort_values("date").reset_index(drop=True)
        return combined

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        train_ratio: float = 0.65,
        val_ratio: float = 0.15,
        purge_days: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split on unique dates, not rows, and purge forward-label overlap.

        All tickers from a given date are assigned to the same partition. The
        final `purge_days` dates of each earlier partition are removed so their
        forward-return labels cannot overlap the next partition.
        """
        if df.empty or "date" not in df.columns:
            empty = df.iloc[0:0].copy()
            return empty, empty, empty
        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values(["date", "ticker"] if "ticker" in work.columns else ["date"])
        dates = pd.Index(work["date"].drop_duplicates().sort_values())
        if len(dates) < 3:
            return work.iloc[0:0].copy(), work.iloc[0:0].copy(), work.copy()
        train_end = max(1, min(len(dates)-2, int(len(dates) * train_ratio)))
        val_end = max(train_end+1, min(len(dates)-1, int(len(dates) * (train_ratio + val_ratio))))
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]
        purge = max(0, int(purge_days))
        if purge:
            train_dates = train_dates[:-purge] if len(train_dates) > purge else train_dates[:0]
            val_dates = val_dates[:-purge] if len(val_dates) > purge else val_dates[:0]
        train_df = work[work["date"].isin(train_dates)].copy().reset_index(drop=True)
        val_df = work[work["date"].isin(val_dates)].copy().reset_index(drop=True)
        test_df = work[work["date"].isin(test_dates)].copy().reset_index(drop=True)
        logger.info("Date-level purged split: Train=%d rows/%d dates, Val=%d rows/%d dates, Test=%d rows/%d dates",
                    len(train_df), len(train_dates), len(val_df), len(val_dates), len(test_df), len(test_dates))
        return train_df, val_df, test_df

    @staticmethod
    def walk_forward_splits(
        df: pd.DataFrame,
        n_splits: int = 3,
        min_train_ratio: float = 0.50,
        purge_days: int = 3,
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Expanding-window splits grouped by date with a purged label boundary."""
        if df.empty or n_splits < 1 or "date" not in df.columns:
            return
        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"])
        dates = pd.Index(work["date"].drop_duplicates().sort_values())
        min_train_len = max(1, int(len(dates) * min_train_ratio))
        remaining = len(dates) - min_train_len
        if remaining <= 0:
            return
        step = max(1, remaining // n_splits)
        for i in range(n_splits):
            train_end = min_train_len + i * step
            test_end = min(len(dates), train_end + step) if i < n_splits - 1 else len(dates)
            train_dates = dates[:max(0, train_end - max(0, purge_days))]
            test_dates = dates[train_end:test_end]
            train_slice = work[work["date"].isin(train_dates)].copy().reset_index(drop=True)
            test_slice = work[work["date"].isin(test_dates)].copy().reset_index(drop=True)
            if not train_slice.empty and not test_slice.empty:
                yield train_slice, test_slice

