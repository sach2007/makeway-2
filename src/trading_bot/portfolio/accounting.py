"""Portfolio accounting, mark-to-market valuation, and fill recording."""
from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING

from trading_bot.api.schemas import Portfolio, PortfolioPosition, TradeAction

if TYPE_CHECKING:
    from trading_bot.api.client import TradingAPIClient

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Maintains local mark-to-market portfolio state and syncs with official gateway."""

    def __init__(self, initial_capital: float = 1_000_000.0):
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        # Map: ticker -> dict with keys: quantity (int, negative for short), avg_price, current_price
        self._positions: dict[str, dict] = {}
        self.realized_pnl: float = 0.0

    def sync_from_gateway(self, client: TradingAPIClient) -> bool:
        """Fetch authoritative portfolio and cash from the official API."""
        try:
            port_dict = client.get_portfolio()
            if isinstance(port_dict, dict):
                self.cash = float(port_dict.get("capital_remaining", port_dict.get("cash", self.cash)))
                self.realized_pnl = float(port_dict.get("realized_pnl", self.realized_pnl))
                
                holdings = port_dict.get("holdings", port_dict.get("positions", []))
                self._positions.clear()
                for h in holdings:
                    ticker = str(h.get("ticker", "")).upper()
                    if ticker:
                        self._positions[ticker] = {
                            "quantity": int(h.get("quantity", 0)),
                            "avg_price": float(h.get("average_buy_price", h.get("average_price", 0.0))),
                            "current_price": float(h.get("current_price", 0.0)),
                            "is_short": bool(h.get("is_short", int(h.get("quantity", 0)) < 0)),
                        }
                logger.info(f"Synchronized portfolio: cash={self.cash:.2f}, positions={len(self._positions)}")
                return True
        except Exception as e:
            logger.warning(f"Could not sync portfolio from gateway: {e}")
        return False

    def on_price_update(self, ticker: str, current_price: float) -> None:
        """Update current mark price for an open position."""
        ticker = ticker.upper()
        if ticker in self._positions:
            self._positions[ticker]["current_price"] = current_price

    def record_fill(self, ticker: str, action: TradeAction, quantity: int, price: float) -> None:
        """Update local book immediately upon confirmed order fill."""
        ticker = ticker.upper()
        pos = self._positions.setdefault(ticker, {
            "quantity": 0, "avg_price": 0.0, "current_price": price, "is_short": False
        })
        current_qty = pos["quantity"]
        pos["current_price"] = price

        if action == TradeAction.BUY:
            # Long purchase: reduces cash, increases quantity
            cost = quantity * price
            self.cash -= cost
            new_qty = current_qty + quantity
            if new_qty > 0:
                pos["avg_price"] = ((current_qty * pos["avg_price"]) + cost) / new_qty
            pos["quantity"] = new_qty
            pos["is_short"] = False

        elif action == TradeAction.SELL:
            # Long sale: adds cash, decreases quantity, realizes PnL
            revenue = quantity * price
            self.cash += revenue
            pnl = (price - pos["avg_price"]) * quantity
            self.realized_pnl += pnl
            pos["quantity"] = max(0, current_qty - quantity)
            if pos["quantity"] == 0:
                pos["avg_price"] = 0.0

        elif action == TradeAction.SHORT:
            # Short sale: borrows and sells shares, crediting cash with negative quantity
            proceeds = quantity * price
            self.cash += proceeds
            new_short_qty = abs(current_qty) + quantity
            # Average short price
            pos["avg_price"] = ((abs(current_qty) * pos["avg_price"]) + proceeds) / new_short_qty
            pos["quantity"] = -new_short_qty
            pos["is_short"] = True

        elif action == TradeAction.COVER:
            # Cover short: buys back shares, debits cash, realizes PnL: (Short Price - Cover Price) * Shares
            buyback_cost = quantity * price
            self.cash -= buyback_cost
            pnl = (pos["avg_price"] - price) * quantity
            self.realized_pnl += pnl
            new_short_qty = max(0, abs(current_qty) - quantity)
            pos["quantity"] = -new_short_qty
            if new_short_qty == 0:
                pos["avg_price"] = 0.0
                pos["is_short"] = False

    def get_portfolio_snapshot(self) -> Portfolio:
        """Compute full mark-to-market portfolio snapshot."""
        holdings = []
        total_market_value = 0.0
        total_unrealized_pnl = 0.0

        for ticker, pos in self._positions.items():
            qty = pos["quantity"]
            if qty == 0:
                continue

            avg_p = pos["avg_price"]
            curr_p = pos.get("current_price") or avg_p

            if qty > 0:  # Long
                mkt_val = qty * curr_p
                unrealized = (curr_p - avg_p) * qty
                unrealized_pct = ((curr_p - avg_p) / avg_p * 100.0) if avg_p > 0 else 0.0
                is_short = False
            else:  # Short (negative quantity)
                mkt_val = qty * curr_p  # Negative liability
                unrealized = (avg_p - curr_p) * abs(qty)
                unrealized_pct = ((avg_p - curr_p) / avg_p * 100.0) if avg_p > 0 else 0.0
                is_short = True

            total_market_value += mkt_val
            total_unrealized_pnl += unrealized

            holdings.append(PortfolioPosition(
                ticker=ticker,
                quantity=qty,
                average_buy_price=avg_p,
                current_price=curr_p,
                market_value=mkt_val,
                unrealized_pnl=unrealized,
                unrealized_pnl_pct=unrealized_pct,
                is_short=is_short
            ))

        total_wealth = self.cash + total_market_value
        total_pnl = self.realized_pnl + total_unrealized_pnl
        total_pnl_pct = ((total_wealth - self.initial_capital) / self.initial_capital * 100.0)

        return Portfolio(
            capital_remaining=self.cash,
            portfolio_market_value=total_market_value,
            total_wealth=total_wealth,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=total_unrealized_pnl,
            positions_count=len(holdings),
            holdings=holdings
        )

    def get_cash_snapshot(self) -> float:
        return self.cash

    def get_holding_quantity(self, ticker: str) -> int:
        return self._positions.get(ticker.upper(), {}).get("quantity", 0)
