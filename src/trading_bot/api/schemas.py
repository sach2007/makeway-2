"""API and Kafka request/response schemas matching the live MANTHAN platform."""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderRequest(BaseModel):
    """Payload for POST /api/v1/orders."""
    ticker: str
    action: TradeAction
    quantity: int = Field(gt=0)
    date: Optional[str] = None  # Market decision date (optional latency tracking)


class OrderResponse(BaseModel):
    """Response from POST /api/v1/orders."""
    order_id: Optional[str] = None
    status: str = "FILLED"
    ticker: str = ""
    action: str = ""
    quantity: int = 0
    execution_price: Optional[float] = None
    price: Optional[float] = None
    total_value: Optional[float] = None
    realized_pnl: Optional[float] = None
    capital_remaining: Optional[float] = None
    simulated_date: Optional[str] = None
    order_date: Optional[str] = None
    decision_delay: Optional[float] = None
    executed_at: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    model_config = {"extra": "allow"}


class StockInfo(BaseModel):
    """Item from GET /api/v1/stocks."""
    ticker: str
    symbol: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    model_config = {"extra": "allow"}


class HistoryBar(BaseModel):
    """OHLCV record from GET /api/v1/stocks/history."""
    date: str
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    model_config = {"extra": "allow"}


class PortfolioPosition(BaseModel):
    """Holding item from GET /api/v1/portfolio."""
    ticker: str
    quantity: int = 0
    average_buy_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    is_short: bool = False
    model_config = {"extra": "allow"}


class Portfolio(BaseModel):
    """Response from GET /api/v1/portfolio."""
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    initial_capital: float = 1_000_000.0
    capital_remaining: float = 1_000_000.0
    portfolio_market_value: float = 0.0
    total_wealth: float = 1_000_000.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions_count: int = 0
    orders_count: int = 0
    holdings: list[PortfolioPosition] = []
    model_config = {"extra": "allow"}


class CashInfo(BaseModel):
    """Response from GET /api/v1/portfolio/cash."""
    available_cash: float = 0.0
    capital_remaining: float = 0.0
    initial_capital: float = 1_000_000.0
    realized_pnl: float = 0.0
    margin_locked: float = 0.0
    currency: str = "INR"
    model_config = {"extra": "allow"}


class SimulationClock(BaseModel):
    """Response from GET /api/v1/simulation/clock."""
    current_day: int = 0
    total_days: int = 0
    remaining_days: int = 0
    seconds_per_day: float = 4.0
    status: str = "RUNNING"
    simulated_date: Optional[str] = None
    is_preflight: bool = False
    is_paused: bool = False
    model_config = {"extra": "allow"}


class StageInfo(BaseModel):
    """Response from GET /api/v1/stage."""
    stage: str = "development"
    status: str = "RUNNING"
    has_run: Optional[bool] = False
    simulation_id: Optional[str] = None
    interval_seconds: Optional[float] = 4.0
    total_days: Optional[int] = 0
    days_completed: Optional[int] = 0
    current_date: Optional[str] = None
    active_dataset: Optional[str] = None
    tickers: Optional[list[str]] = None
    ticker_count: Optional[int] = None
    model_config = {"extra": "allow"}


class HeartbeatRequest(BaseModel):
    """Payload for POST /api/v1/bot/heartbeat."""
    seq: int


class HeartbeatResponse(BaseModel):
    """Response from POST /api/v1/bot/heartbeat."""
    status: str = "ok"
    message: Optional[str] = None
    team_id: Optional[str] = None
    seq: Optional[int] = None
    model_config = {"extra": "allow"}


class MarketTick(BaseModel):
    """Normalized tick structure used internally by strategy engine."""
    ticker: str
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    simulated_date: Optional[str] = None
    timestamp: Optional[str] = None
    event_type: str = "price_tick"
    model_config = {"extra": "allow"}


class NewsItem(BaseModel):
    """Normalized news item used internally."""
    ticker: Optional[str] = None
    headline: str = ""
    description: Optional[str] = None
    date: Optional[str] = None
    simulated_date: Optional[str] = None
    sentiment_score: float = 0.0
    timestamp: Optional[str] = None
    model_config = {"extra": "allow"}
