"""API Client, Authentication, and Schemas for Order Gateway."""

from trading_bot.api.client import AuthenticationError, TradingAPIClient
from trading_bot.api.schemas import (
    HeartbeatRequest,
    HeartbeatResponse,
    HistoryBar,
    MarketTick,
    NewsItem,
    OrderRequest,
    OrderResponse,
    Portfolio,
    PortfolioPosition,
    SimulationClock,
    StageInfo,
    StockInfo,
    TradeAction,
)

__all__ = [
    "TradingAPIClient",
    "AuthenticationError",
    "TradeAction",
    "OrderRequest",
    "OrderResponse",
    "StockInfo",
    "HistoryBar",
    "PortfolioPosition",
    "Portfolio",
    "SimulationClock",
    "StageInfo",
    "HeartbeatRequest",
    "HeartbeatResponse",
    "MarketTick",
    "NewsItem",
]
