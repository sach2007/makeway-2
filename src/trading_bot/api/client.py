"""HTTP API client for the MANTHAN trading simulation platform."""
import logging
import time
from typing import Any, Optional

import httpx

from trading_bot.api.schemas import (
    HeartbeatRequest, HeartbeatResponse, HistoryBar,
    OrderRequest, OrderResponse, Portfolio, SimulationClock,
    StageInfo, StockInfo,
)
from trading_bot.config import Settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter."""
    def __init__(self, max_per_second: int = 50):
        self.max_per_second = max_per_second
        self._tokens = float(max_per_second)
        self._last_refill = time.monotonic()
    
    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_per_second, self._tokens + elapsed * self.max_per_second)
        self._last_refill = now
        if self._tokens < 1.0:
            sleep_time = (1.0 - self._tokens) / self.max_per_second
            time.sleep(sleep_time)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0


class TradingAPIClient:
    """Client for the MANTHAN trading simulation API."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.api_base_url.rstrip('/')
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=settings.api_headers,
            timeout=httpx.Timeout(settings.order_timeout_seconds, connect=5.0),
        )
        self._rate_limiter = RateLimiter(settings.rate_limit_per_second)
        self._request_count = 0
        self._error_count = 0
        self._consecutive_5xx = 0
        self.is_emergency_halted = False
    
    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Make an API request with rate limiting, retries, jitter, and safety kill switch."""
        if self.is_emergency_halted:
            raise RuntimeError("API Client is halted by Emergency Kill Switch (3 consecutive 5xx errors).")

        self._rate_limiter.acquire()
        import random
        last_error = None
        
        for attempt in range(self.settings.max_order_retries):
            try:
                self._request_count += 1
                t0 = time.perf_counter()
                resp = self._client.request(method, path, **kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000
                
                logger.debug(f"{method} {path} -> {resp.status_code} ({latency_ms:.1f}ms)")
                
                if resp.status_code == 401:
                    logger.error("Authentication failed (401). Check API_TOKEN.")
                    raise AuthenticationError("Invalid or expired token")
                
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get('Retry-After', 1.0))
                    logger.warning(f"Rate limited (429). Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                
                # Fail-fast on 4xx client errors (do not retry bad order syntax/rejected trades)
                if 400 <= resp.status_code < 500:
                    self._consecutive_5xx = 0
                    return resp

                if resp.status_code >= 500:
                    self._consecutive_5xx += 1
                    logger.warning(f"Server error {resp.status_code} (consecutive: {self._consecutive_5xx}). Attempt {attempt+1}")
                    if self._consecutive_5xx >= 3:
                        self.is_emergency_halted = True
                        logger.critical("API returned 3 consecutive 5xx server errors. Tripping Emergency Kill Switch!")
                        raise RuntimeError("API Emergency Kill Switch: 3 consecutive 5xx errors")
                    jitter = random.uniform(0.05, 0.3)
                    time.sleep(min(2 ** attempt, 8) + jitter)
                    continue
                
                # Successful response (< 400)
                self._consecutive_5xx = 0
                return resp
                
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
                last_error = e
                self._error_count += 1
                logger.warning(f"Network error on {method} {path}: {e}. Attempt {attempt+1}")
                import random
                time.sleep(min(2 ** attempt, 8) + random.uniform(0.05, 0.3))
        
        raise ConnectionError(f"Failed after {self.settings.max_order_retries} attempts: {last_error}")
    
    # === Stock Universe ===
    def get_stocks(self) -> list[dict]:
        """Get available stock universe."""
        resp = self._request("GET", "/api/v1/stocks")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get('stocks', data.get('data', [data]))
        return []
    
    # === Historical Data ===
    def get_history(self, ticker: str, **params: Any) -> list[dict]:
        """Get historical OHLCV data for a ticker."""
        params['ticker'] = ticker
        resp = self._request("GET", "/api/v1/stocks/history", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get('prices', data.get('history', data.get('data', data.get('candles', []))))
        return []
    
    # === Portfolio ===
    def get_portfolio(self) -> dict:
        """Get current portfolio state."""
        resp = self._request("GET", "/api/v1/portfolio")
        resp.raise_for_status()
        return resp.json()
    
    def get_cash(self) -> float:
        """Get available cash."""
        resp = self._request("GET", "/api/v1/portfolio/cash")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, (int, float)):
            return float(data)
        if isinstance(data, dict):
            return float(data.get('available_cash', data.get('capital_remaining', data.get('cash_balance', data.get('cash', 0.0)))))
        return 0.0
    
    # === Orders ===
    def place_order(self, order: OrderRequest, client_order_id: Optional[str] = None) -> dict:
        """Submit an order with optional idempotency tracking. Returns the server response."""
        headers = {"X-Client-Order-ID": client_order_id} if client_order_id else {}
        resp = self._request("POST", "/api/v1/orders", json=order.model_dump(), headers=headers)
        result = resp.json() if resp.text else {}
        logger.info(f"Order response: {order.action} {order.quantity} {order.ticker} [id={client_order_id}] -> {resp.status_code}")
        return {"status_code": resp.status_code, **result} if isinstance(result, dict) else {"status_code": resp.status_code, "data": result}
    
    def get_orders(self) -> list[dict]:
        """Get order history."""
        resp = self._request("GET", "/api/v1/orders")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get('orders', data.get('data', []))
        return []
    
    # === Simulation Lifecycle ===
    def get_stage(self) -> dict:
        """Get current simulation stage."""
        resp = self._request("GET", "/api/v1/stage")
        resp.raise_for_status()
        return resp.json()
    
    def get_clock(self) -> dict:
        """Get simulation clock."""
        resp = self._request("GET", "/api/v1/simulation/clock")
        resp.raise_for_status()
        return resp.json()
    
    def send_heartbeat(self, seq: int) -> dict:
        """Send heartbeat with sequence number."""
        payload = HeartbeatRequest(seq=seq)
        resp = self._request("POST", "/api/v1/bot/heartbeat", json=payload.model_dump())
        return resp.json() if resp.text else {"status_code": resp.status_code}
    
    # === Health ===
    def check_connectivity(self) -> dict[str, Any]:
        """Test API connectivity without exposing credentials."""
        results = {}
        for name, path in [("stage", "/api/v1/stage"), ("stocks", "/api/v1/stocks")]:
            try:
                t0 = time.perf_counter()
                resp = self._client.get(path)
                ms = (time.perf_counter() - t0) * 1000
                results[name] = {"status": resp.status_code, "latency_ms": round(ms, 1)}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        results["token_configured"] = bool(self.settings.api_token)
        return results
    
    @property
    def stats(self) -> dict:
        return {"requests": self._request_count, "errors": self._error_count}
    
    def close(self) -> None:
        self._client.close()


class AuthenticationError(Exception):
    pass
