"""Deterministic lightweight financial sentiment analyzer and news feature engine."""
from __future__ import annotations
import re
from typing import Optional
from trading_bot.api.schemas import NewsItem

# High-impact financial lexicon
POSITIVE_KEYWORDS = {
    "surge", "surges", "surging", "profit", "profits", "profitable",
    "record", "growth", "growing", "beat", "beats", "beating",
    "upgrade", "upgraded", "upgrades", "rally", "rallies", "dividend",
    "breakthrough", "acquisition", "expansion", "bullish", "outperform",
    "soar", "soars", "soaring", "gain", "gains", "revenue", "partnership",
    "patent", "contract", "approval", "approved", "milestone"
}

NEGATIVE_KEYWORDS = {
    "plunge", "plunges", "plunging", "slump", "slumps", "loss", "losses",
    "miss", "misses", "missing", "downgrade", "downgraded", "probe",
    "investigation", "lawsuit", "sued", "fraud", "scandal", "default",
    "bankruptcy", "bearish", "decline", "declining", "crash", "crashes",
    "penalty", "fine", "fined", "violation", "reconsider", "warning",
    "recalled", "cut", "layoff", "layoffs", "debt", "deficit"
}


class NewsSentimentAnalyzer:
    """Fast, local keyword sentiment scoring with negation and intensity handling."""

    @classmethod
    def score_text(cls, text: str) -> float:
        """Returns a normalized sentiment score in [-1.0, 1.0]."""
        if not text:
            return 0.0

        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return 0.0

        pos_score = 0.0
        neg_score = 0.0
        negated = False

        for i, word in enumerate(words):
            # Check negation within 2 preceding words
            if i > 0 and words[i-1] in {"not", "no", "never", "without", "hardly"}:
                negated = True
            elif i > 1 and words[i-2] in {"not", "no", "never"}:
                negated = True
            else:
                negated = False

            if word in POSITIVE_KEYWORDS:
                if negated:
                    neg_score += 1.0
                else:
                    pos_score += 1.0
            elif word in NEGATIVE_KEYWORDS:
                if negated:
                    pos_score += 1.0
                else:
                    neg_score += 1.0

        total = pos_score + neg_score
        if total == 0:
            return 0.0

        raw_score = (pos_score - neg_score) / total
        return max(-1.0, min(1.0, raw_score))


class NewsManager:
    """Maintains recent news sentiment per ticker with exponential decay."""

    def __init__(self, decay_rate: float = 0.85):
        self.decay_rate = decay_rate
        # Map: ticker -> latest decayed sentiment score
        self._ticker_sentiment: dict[str, float] = {}
        self._latest_news: dict[str, NewsItem] = {}

    def on_news(self, news: NewsItem) -> float:
        """Process incoming news item and update ticker sentiment score."""
        text = f"{news.headline} {news.description or ''}"
        score = NewsSentimentAnalyzer.score_text(text)
        news.sentiment_score = score

        ticker = (news.ticker or "").upper()
        if ticker:
            # Blend with previous sentiment using decay
            prev = self._ticker_sentiment.get(ticker, 0.0)
            blended = (prev * self.decay_rate) + (score * (1.0 - self.decay_rate * 0.5))
            self._ticker_sentiment[ticker] = max(-1.0, min(1.0, blended))
            self._latest_news[ticker] = news

        return score

    def on_tick_cycle(self) -> None:
        """Decay all news sentiments as time/ticks progress."""
        for ticker in list(self._ticker_sentiment.keys()):
            self._ticker_sentiment[ticker] *= self.decay_rate
            if abs(self._ticker_sentiment[ticker]) < 0.01:
                self._ticker_sentiment[ticker] = 0.0

    def get_sentiment(self, ticker: str) -> float:
        """Get current decaying sentiment score for ticker [-1.0 to +1.0]."""
        return self._ticker_sentiment.get(ticker.upper(), 0.0)

    def get_latest_headline(self, ticker: str) -> Optional[str]:
        item = self._latest_news.get(ticker.upper())
        return item.headline if item else None
