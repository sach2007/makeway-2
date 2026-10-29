"""Independent strategy signal generators producing normalized [-1.0, 1.0] conviction scores."""
from __future__ import annotations
import math
from typing import Any
from trading_bot.api.schemas import TradeAction


class SignalScore:
    """Represents a strategy's evaluated conviction on a ticker."""
    def __init__(self, ticker: str, score: float, strategy_name: str, rationale: str = ""):
        self.ticker = ticker
        self.score = max(-1.0, min(1.0, float(score)))  # Clamp to [-1.0, 1.0]
        self.strategy_name = strategy_name
        self.rationale = rationale

    @property
    def action(self) -> TradeAction:
        if self.score > 0.2:
            return TradeAction.BUY
        elif self.score < -0.2:
            return TradeAction.SHORT
        return TradeAction.SELL


class MomentumStrategy:
    """Evaluates short and medium-term price velocity confirmed by volume."""

    @classmethod
    def evaluate(cls, ticker: str, features: dict[str, float]) -> SignalScore:
        roc_s = features.get("roc_short", 0.0)
        roc_m = features.get("roc_med", 0.0)
        vol_ratio = features.get("volume_ratio", 1.0)

        blended_roc = (roc_s * 0.6) + (roc_m * 0.4)
        raw_score = math.tanh(blended_roc / 3.0)

        vol_mult = min(1.3, max(0.8, vol_ratio))
        score = max(-1.0, min(1.0, raw_score * vol_mult))

        rationale = f"ROC(s={roc_s:.1f}%, m={roc_m:.1f}%), VolRatio={vol_ratio:.2f}"
        return SignalScore(ticker=ticker, score=score, strategy_name="Momentum", rationale=rationale)


class TrendFollowingStrategy:
    """Uses EMA relationships and MACD momentum structure."""

    @classmethod
    def evaluate(cls, ticker: str, features: dict[str, float]) -> SignalScore:
        close = features.get("close", 0.0)
        if close <= 0:
            return SignalScore(ticker=ticker, score=0.0, strategy_name="Trend")

        ema_s = features.get("ema_short", close)
        ema_m = features.get("ema_med", close)
        macd_hist = features.get("macd_hist", 0.0)
        rsi = features.get("rsi_14", 50.0)

        ema_diff_pct = (ema_s - ema_m) / (close + 1e-10) * 100.0
        ema_score = math.tanh(ema_diff_pct / 2.0)
        macd_score = math.tanh(macd_hist / (close * 0.005 + 1e-10))
        rsi_bias = (rsi - 50.0) / 50.0

        blended = (ema_score * 0.5) + (macd_score * 0.35) + (rsi_bias * 0.15)
        score = max(-1.0, min(1.0, blended))

        rationale = f"EMA_diff={ema_diff_pct:.2f}%, MACD_h={macd_hist:.2f}, RSI={rsi:.1f}"
        return SignalScore(ticker=ticker, score=score, strategy_name="Trend", rationale=rationale)


class MeanReversionStrategy:
    """Detects overextended statistical deviations using Bollinger %B, RSI extremes, and Z-scores."""

    @classmethod
    def evaluate(cls, ticker: str, features: dict[str, float]) -> SignalScore:
        bb_pct = features.get("bb_percent_b", 0.5)
        rsi = features.get("rsi_14", 50.0)
        z_score = features.get("z_score_price", 0.0)

        score = 0.0
        rationale_parts = []

        if bb_pct < 0.15 or rsi < 32.0 or z_score < -1.8:
            oversold_intensity = max(
                (0.2 - bb_pct) * 3.0 if bb_pct < 0.2 else 0.0,
                (35.0 - rsi) / 20.0 if rsi < 35.0 else 0.0,
                abs(z_score + 1.5) / 1.5 if z_score < -1.5 else 0.0
            )
            score = min(1.0, oversold_intensity)
            rationale_parts.append(f"Oversold: BB%B={bb_pct:.2f}, RSI={rsi:.1f}, Z={z_score:.2f}")

        elif bb_pct > 0.85 or rsi > 68.0 or z_score > 1.8:
            overbought_intensity = max(
                (bb_pct - 0.8) * 3.0 if bb_pct > 0.8 else 0.0,
                (rsi - 65.0) / 20.0 if rsi > 65.0 else 0.0,
                (z_score - 1.5) / 1.5 if z_score > 1.5 else 0.0
            )
            score = -min(1.0, overbought_intensity)
            rationale_parts.append(f"Overbought: BB%B={bb_pct:.2f}, RSI={rsi:.1f}, Z={z_score:.2f}")
        else:
            score = 0.0
            rationale_parts.append(f"Neutral: BB%B={bb_pct:.2f}, RSI={rsi:.1f}")

        return SignalScore(ticker=ticker, score=score, strategy_name="MeanReversion", rationale="; ".join(rationale_parts))


class DipBuyingStrategy:
    """Specialized engine to detect high-probability dip buying and rally shorting setups."""

    @classmethod
    def evaluate(cls, ticker: str, features: dict[str, float], ml_conviction: float = 0.0) -> SignalScore:
        close = features.get("close", 0.0)
        if close <= 0:
            return SignalScore(ticker=ticker, score=0.0, strategy_name="DipBuyer")

        ema_m = features.get("ema_med", close)
        sma_l = features.get("sma_long", close)
        rsi = features.get("rsi_14", 50.0)
        bb_pct = features.get("bb_percent_b", 0.5)
        macd_hist = features.get("macd_hist", 0.0)

        # 1. Bullish Dip Buying: Asset in macro uptrend (close > sma_l or ema_m > sma_l) pulling back
        macro_uptrend = (close >= sma_l * 0.97) or (ema_m >= sma_l)
        is_dip = (rsi < 42.0) or (bb_pct < 0.25)

        if macro_uptrend and is_dip:
            # Dip depth score
            dip_score = max(0.40, (45.0 - rsi) / 25.0)
            if bb_pct < 0.15:
                dip_score += 0.25
            if ml_conviction > 0:
                dip_score += 0.20
            score = min(1.0, dip_score)
            rationale = f"BUY_THE_DIP: Macro Bull (Close {close:.2f} >= SMA {sma_l:.2f}), RSI={rsi:.1f}, BB%B={bb_pct:.2f}"
            return SignalScore(ticker=ticker, score=score, strategy_name="DipBuyer", rationale=rationale)

        # 2. Bearish Rally Shorting: Asset in macro downtrend (close < sma_l) overbought on bounce
        macro_downtrend = (close < sma_l * 0.98) and (ema_m < sma_l)
        is_rally = (rsi > 58.0) or (bb_pct > 0.75)

        if macro_downtrend and is_rally:
            short_score = max(0.40, (rsi - 55.0) / 25.0)
            if bb_pct > 0.85:
                short_score += 0.25
            if ml_conviction < 0:
                short_score += 0.20
            score = -min(1.0, short_score)
            rationale = f"SHORT_THE_RALLY: Macro Bear (Close {close:.2f} < SMA {sma_l:.2f}), RSI={rsi:.1f}, BB%B={bb_pct:.2f}"
            return SignalScore(ticker=ticker, score=score, strategy_name="DipBuyer", rationale=rationale)

        return SignalScore(ticker=ticker, score=0.0, strategy_name="DipBuyer", rationale="No dip/rally trigger")


class NewsSignalStrategy:
    """Translates real-time news sentiment into bounded signal convictions."""

    @classmethod
    def evaluate(cls, ticker: str, sentiment_score: float) -> SignalScore:
        score = max(-1.0, min(1.0, sentiment_score))
        rationale = f"NewsSentiment={sentiment_score:.2f}"
        return SignalScore(ticker=ticker, score=score, strategy_name="News", rationale=rationale)


class MLSignalStrategy:
    """Translates machine learning forward return predictions into bounded conviction."""

    @classmethod
    def evaluate(cls, ticker: str, ml_conviction: float) -> SignalScore:
        score = max(-1.0, min(1.0, ml_conviction))
        rationale = f"ML_Conviction={ml_conviction:+.2f}"
        return SignalScore(ticker=ticker, score=score, strategy_name="MachineLearning", rationale=rationale)
