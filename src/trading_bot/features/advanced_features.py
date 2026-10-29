"""Institutional feature engineering pipeline with cross-sectional ranks, market breadth, higher moments, and FinBERT sentiment."""
from __future__ import annotations
import math
import logging
from typing import Optional
import numpy as np
import pandas as pd

from trading_bot.features.indicators import TechnicalIndicators
from trading_bot.features.news_features import NewsSentimentAnalyzer

logger = logging.getLogger(__name__)

# Complete institutional feature columns
EXPANDED_FEATURE_COLUMNS = [
    # Core 16 Indicators
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
    # Advanced Volatility & Distributional Moments
    "realized_vol_parkinson_20d",
    "volume_zscore_20d",
    "overnight_gap",
    "intraday_range_ratio",
    "rolling_skew_20d",
    "rolling_kurt_20d",
    "dist_from_rolling_high_20d",
    "dist_from_rolling_low_20d",
    "stoch_position_20d",
    # Market-Wide Breadth
    "market_breadth_above_ema20",
    "market_mean_return_1d",
    # Cross-Sectional Percentile Ranks
    "cs_rank_ret_1d",
    "cs_rank_ret_5d",
    "cs_rank_rsi",
    "cs_rank_vol_zscore",
    "cs_rank_parkinson_vol",
    # FinBERT Sentiment
    "news_sentiment_1h",
    "news_sentiment_1d",
]


class AdvancedFeaturePipeline:
    """Computes full point-in-time institutional feature matrix with cross-sectional rankings and market breadth."""

    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """Parkinson Extreme Value Volatility estimator (5x more efficient than close-to-close std).
        Formula: sqrt( (1 / (4 * ln(2) * N)) * sum( ln(H / L)^2 ) )
        """
        hl_ratio = (high / (low.replace(0.0, np.nan) + 1e-10)).clip(lower=1.0)
        log_hl_sq = np.log(hl_ratio) ** 2
        factor = 1.0 / (4.0 * math.log(2.0))
        rolling_var = log_hl_sq.rolling(window=window, min_periods=2).mean() * factor
        return np.sqrt(rolling_var.clip(lower=0.0)).fillna(0.0)

    @classmethod
    def compute_single_symbol_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all time-series features for a single symbol strictly point-in-time (no future data)."""
        if df.empty or len(df) < 30:
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
        feats["open"] = open_p
        feats["high"] = high
        feats["low"] = low
        feats["volume"] = volume

        # 1. Core returns
        feats["ret_1d"] = close.pct_change(1)
        feats["ret_3d"] = close.pct_change(3)
        feats["ret_5d"] = close.pct_change(5)
        feats["ret_10d"] = close.pct_change(10)

        # 2. Moving average divergence
        ema5 = TechnicalIndicators.ema(close, 5)
        ema20 = TechnicalIndicators.ema(close, 20)
        feats["ema_5_20_ratio"] = (ema5 - ema20) / (close + 1e-10)
        feats["ema_20"] = ema20

        sma50 = TechnicalIndicators.sma(close, 50)
        feats["sma_50_ratio"] = (close - sma50) / (sma50 + 1e-10)

        # 3. Oscillators
        rsi = TechnicalIndicators.rsi(close, 14)
        feats["rsi_14_norm"] = (rsi - 50.0) / 50.0

        _, _, macd_hist = TechnicalIndicators.macd(close, 12, 26, 9)
        feats["macd_hist_ratio"] = macd_hist / (close + 1e-10)

        # 4. Volatility & Bands
        mid, up, low_b, pct_b = TechnicalIndicators.bollinger_bands(close, 20)
        feats["bb_percent_b"] = pct_b
        feats["bb_bandwidth"] = (up - low_b) / (mid.replace(0.0, np.nan) + 1e-10)

        atr = TechnicalIndicators.atr(high, low, close, 14)
        feats["atr_ratio"] = atr / (close + 1e-10)
        feats["atr_14"] = atr

        feats["volatility_10d"] = TechnicalIndicators.rolling_volatility(close, 10)
        feats["volatility_20d"] = TechnicalIndicators.rolling_volatility(close, 20)

        # 5. Volume ratios
        vol_mean20 = volume.rolling(window=20, min_periods=1).mean()
        feats["volume_ratio"] = volume / (vol_mean20 + 1e-10)
        vol_std20 = volume.rolling(window=20, min_periods=2).std().fillna(1.0)
        feats["volume_zscore_20d"] = ((volume - vol_mean20) / (vol_std20 + 1e-10)).clip(-3.0, 3.0)

        # 6. Intraday and overnight geometric features
        feats["high_low_ratio"] = (high - low) / (close + 1e-10)
        feats["close_open_ratio"] = (close - open_p) / (open_p + 1e-10)
        feats["intraday_range_ratio"] = (high - low) / (open_p + 1e-10)

        # Overnight gap: today's open vs yesterday's close (shift 1 close)
        prev_close = close.shift(1)
        feats["overnight_gap"] = ((open_p - prev_close) / (prev_close + 1e-10)).fillna(0.0)

        # 7. Realized Parkinson Volatility
        feats["realized_vol_parkinson_20d"] = cls.parkinson_volatility(high, low, 20)

        # 8. Rolling Skewness and Kurtosis of daily returns
        log_rets = np.log(close / close.shift(1)).fillna(0.0)
        feats["rolling_skew_20d"] = log_rets.rolling(window=20, min_periods=10).skew().fillna(0.0).clip(-3.0, 3.0)
        feats["rolling_kurt_20d"] = log_rets.rolling(window=20, min_periods=10).kurt().fillna(0.0).clip(-3.0, 10.0)

        # 9. Distance from rolling 20-day High / Low
        roll_high20 = high.rolling(window=20, min_periods=1).max()
        roll_low20 = low.rolling(window=20, min_periods=1).min()
        feats["dist_from_rolling_high_20d"] = (close / (roll_high20 + 1e-10)) - 1.0
        feats["dist_from_rolling_low_20d"] = (close / (roll_low20 + 1e-10)) - 1.0
        feats["stoch_position_20d"] = (close - roll_low20) / (roll_high20 - roll_low20 + 1e-10)

        # News placeholders (filled by news manager during live/replay)
        feats["news_sentiment_1h"] = 0.0
        feats["news_sentiment_1d"] = 0.0

        return feats

    @classmethod
    def compute_multi_symbol_universe_matrix(
        cls,
        symbol_dfs: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Compute full point-in-time universe matrix with cross-sectional ranks and market breadth."""
        all_symbol_feats = []

        for ticker, df in symbol_dfs.items():
            s_feats = cls.compute_single_symbol_features(df)
            if not s_feats.empty:
                s_feats["ticker"] = ticker
                all_symbol_feats.append(s_feats)

        if not all_symbol_feats:
            return pd.DataFrame()

        universe_df = pd.concat(all_symbol_feats, ignore_index=True)
        universe_df["date_str"] = universe_df["date"].astype(str).str[:10]

        # 1. Market Breadth per date: fraction of universe trading above their own EMA20
        universe_df["above_ema20"] = (universe_df["close"] > universe_df["ema_20"]).astype(float)
        breadth_by_date = universe_df.groupby("date_str")["above_ema20"].mean().to_dict()
        mean_ret_by_date = universe_df.groupby("date_str")["ret_1d"].mean().to_dict()

        universe_df["market_breadth_above_ema20"] = universe_df["date_str"].map(breadth_by_date).fillna(0.50)
        universe_df["market_mean_return_1d"] = universe_df["date_str"].map(mean_ret_by_date).fillna(0.0)

        # 2. Vectorized Cross-Sectional Percentile Rankings across tickers on each date
        for src_col, rank_col in [
            ("ret_1d", "cs_rank_ret_1d"),
            ("ret_5d", "cs_rank_ret_5d"),
            ("rsi_14_norm", "cs_rank_rsi"),
            ("volume_zscore_20d", "cs_rank_vol_zscore"),
            ("realized_vol_parkinson_20d", "cs_rank_parkinson_vol"),
        ]:
            universe_df[rank_col] = universe_df.groupby("date_str")[src_col].rank(pct=True).fillna(0.50)

        # Sort strictly chronologically by date
        universe_df = universe_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        # Fill any remaining NaNs cleanly
        universe_df[EXPANDED_FEATURE_COLUMNS] = universe_df[EXPANDED_FEATURE_COLUMNS].fillna(0.0)

        return universe_df
