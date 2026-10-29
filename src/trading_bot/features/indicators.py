"""Technical indicator calculations for quantitative strategy signals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


class TechnicalIndicators:
    """Vectorized and rolling technical indicator computation."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (0 to 100)."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        return rsi_series.fillna(50.0)

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Moving Average Convergence Divergence.
        Returns: (macd_line, signal_line, histogram)
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.
        Returns: (middle_band, upper_band, lower_band, percent_b)
        percent_b = (price - lower) / (upper - lower)
        """
        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std().fillna(0.0)
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        band_width = upper - lower
        percent_b = (series - lower) / (band_width.replace(0.0, np.nan))
        percent_b = percent_b.fillna(0.5)
        return middle, upper, lower, percent_b

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()

    @staticmethod
    def roc(series: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change (% price change over period)."""
        return series.pct_change(periods=period).fillna(0.0) * 100.0

    @staticmethod
    def rolling_volatility(series: pd.Series, period: int = 20) -> pd.Series:
        """Rolling daily standard deviation of log returns."""
        log_ret = np.log(series / series.shift(1)).fillna(0.0)
        return log_ret.rolling(window=period, min_periods=2).std().fillna(0.0)


class FeatureExtractor:
    """Computes all technical features for a ticker's historical dataframe."""

    @classmethod
    def compute_features(cls, df: pd.DataFrame, short_win: int = 5, med_win: int = 20, long_win: int = 50) -> dict[str, float]:
        """Extract latest feature snapshot for strategy evaluation.
        Requires dataframe with columns: close, open, high, low, volume.
        """
        if df.empty or len(df) < 2:
            return {
                "close": 0.0,
                "sma_short": 0.0,
                "sma_med": 0.0,
                "sma_long": 0.0,
                "ema_short": 0.0,
                "ema_med": 0.0,
                "rsi_14": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_hist": 0.0,
                "bb_percent_b": 0.5,
                "atr": 0.0,
                "roc_short": 0.0,
                "roc_med": 0.0,
                "volatility_20": 0.0,
                "z_score_price": 0.0,
                "volume_ratio": 1.0,
                "ret_1d": 0.0, "ret_3d": 0.0, "ret_5d": 0.0, "ret_10d": 0.0,
                "ema_5_20_ratio": 0.0, "sma_50_ratio": 0.0, "rsi_14_norm": 0.0,
                "macd_hist_ratio": 0.0, "bb_bandwidth": 0.0, "atr_ratio": 0.0,
                "volatility_10d": 0.0, "volatility_20d": 0.0,
                "high_low_ratio": 0.0, "close_open_ratio": 0.0,
            }

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        sma_s = TechnicalIndicators.sma(close, short_win).iloc[-1]
        sma_m = TechnicalIndicators.sma(close, med_win).iloc[-1]
        sma_l = TechnicalIndicators.sma(close, long_win).iloc[-1]

        ema_s = TechnicalIndicators.ema(close, short_win).iloc[-1]
        ema_m = TechnicalIndicators.ema(close, med_win).iloc[-1]

        rsi_val = TechnicalIndicators.rsi(close, 14).iloc[-1]
        macd_val, macd_sig, macd_h = TechnicalIndicators.macd(close, 12, 26, 9)
        _, _, _, bb_pct = TechnicalIndicators.bollinger_bands(close, med_win)
        atr_val = TechnicalIndicators.atr(high, low, close, 14).iloc[-1]

        roc_s = TechnicalIndicators.roc(close, short_win).iloc[-1]
        roc_m = TechnicalIndicators.roc(close, med_win).iloc[-1]
        vol_20 = TechnicalIndicators.rolling_volatility(close, med_win).iloc[-1]

        # Price distance z-score
        rolling_mean = close.rolling(window=med_win, min_periods=1).mean().iloc[-1]
        rolling_std = close.rolling(window=med_win, min_periods=1).std().iloc[-1]
        z_score = (close.iloc[-1] - rolling_mean) / (rolling_std + 1e-10)

        # Relative volume
        mean_vol = volume.rolling(window=med_win, min_periods=1).mean().iloc[-1]
        vol_ratio = (volume.iloc[-1] / (mean_vol + 1e-10)) if mean_vol > 0 else 1.0

        # ML features must use the exact names and normalization used by
        # TimeSeriesFeaturePipeline during training. Keep these causal and aligned.
        close_safe = close.replace(0.0, np.nan)
        ema5 = TechnicalIndicators.ema(close, 5)
        ema20 = TechnicalIndicators.ema(close, 20)
        sma50 = TechnicalIndicators.sma(close, 50)
        _, bb_up, bb_low, bb_pct20 = TechnicalIndicators.bollinger_bands(close, 20)
        bb_mid20 = TechnicalIndicators.sma(close, 20)
        atr14 = TechnicalIndicators.atr(high, low, close, 14)
        ret_1d = close.pct_change(1).iloc[-1]
        ret_3d = close.pct_change(3).iloc[-1]
        ret_5d = close.pct_change(5).iloc[-1]
        ret_10d = close.pct_change(10).iloc[-1]
        vol_mean20 = volume.rolling(window=20, min_periods=1).mean().iloc[-1]
        model_features = {
            "ret_1d": ret_1d,
            "ret_3d": ret_3d,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "ema_5_20_ratio": (ema5.iloc[-1] - ema20.iloc[-1]) / (close.iloc[-1] + 1e-10),
            "sma_50_ratio": (close.iloc[-1] - sma50.iloc[-1]) / (sma50.iloc[-1] + 1e-10),
            "rsi_14_norm": (rsi_val - 50.0) / 50.0,
            "macd_hist_ratio": macd_h.iloc[-1] / (close.iloc[-1] + 1e-10),
            "bb_percent_b": bb_pct20.iloc[-1],
            "bb_bandwidth": (bb_up.iloc[-1] - bb_low.iloc[-1]) / (bb_mid20.iloc[-1] + 1e-10),
            "atr_ratio": atr14.iloc[-1] / (close.iloc[-1] + 1e-10),
            "volatility_10d": TechnicalIndicators.rolling_volatility(close, 10).iloc[-1],
            "volatility_20d": TechnicalIndicators.rolling_volatility(close, 20).iloc[-1],
            "volume_ratio": volume.iloc[-1] / (vol_mean20 + 1e-10),
            "high_low_ratio": (high.iloc[-1] - low.iloc[-1]) / (close.iloc[-1] + 1e-10),
            "close_open_ratio": (close.iloc[-1] - df["open"].iloc[-1]) / (df["open"].iloc[-1] + 1e-10),
        }

        result = {
            "close": float(close.iloc[-1]),
            "sma_short": float(sma_s),
            "sma_med": float(sma_m),
            "sma_long": float(sma_l),
            "ema_short": float(ema_s),
            "ema_med": float(ema_m),
            "rsi_14": float(rsi_val),
            "macd": float(macd_val.iloc[-1]),
            "macd_signal": float(macd_sig.iloc[-1]),
            "macd_hist": float(macd_h.iloc[-1]),
            "bb_percent_b": float(bb_pct.iloc[-1]),
            "atr": float(atr_val),
            "roc_short": float(roc_s),
            "roc_med": float(roc_m),
            "volatility_20": float(vol_20),
            "z_score_price": float(z_score),
            "volume_ratio": float(vol_ratio),
        }
        for name, value in model_features.items():
            try:
                numeric = float(value)
                result[name] = numeric if np.isfinite(numeric) else 0.0
            except (TypeError, ValueError):
                result[name] = 0.0
        return result
