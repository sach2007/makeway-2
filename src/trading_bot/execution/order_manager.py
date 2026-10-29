"""Order Management System (OMS) with state machine, reconciliation, and retry safety."""
from __future__ import annotations
import logging
import time
import uuid
from enum import Enum
from typing import Optional, TYPE_CHECKING

from trading_bot.api.schemas import OrderRequest, OrderResponse, TradeAction
from trading_bot.risk.limits import PreTradeRiskEngine

if TYPE_CHECKING:
    from trading_bot.api.client import TradingAPIClient
    from trading_bot.portfolio.accounting import PortfolioTracker

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"


class ManagedOrder:
    """Internal record tracking an order through its full lifecycle."""
    def __init__(self, request: OrderRequest, intent_id: Optional[str] = None):
        self.intent_id = intent_id or str(uuid.uuid4())
        self.request = request
        self.state = OrderState.CREATED
        self.order_id: Optional[str] = None
        self.execution_price: Optional[float] = None
        self.rejection_reason: Optional[str] = None
        self.created_at = time.time()
        self.submitted_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.roundtrip_ms: float = 0.0


class OrderManager:
    """Handles order validation, submission, response handling, and post-trade safety."""

    def __init__(
        self,
        api_client: TradingAPIClient,
        risk_engine: PreTradeRiskEngine,
        portfolio_tracker: PortfolioTracker
    ):
        self.client = api_client
        self.risk_engine = risk_engine
        self.portfolio_tracker = portfolio_tracker
        self.order_history: list[ManagedOrder] = []
        self._total_filled = 0
        self._total_rejected = 0

    def execute_decision(
        self,
        ticker: str,
        action: TradeAction,
        quantity: int,
        estimated_price: float,
        simulated_date: Optional[str] = None
    ) -> Optional[ManagedOrder]:
        """Validate, size, submit, and reconcile an order."""
        if quantity <= 0:
            return None

        order_req = OrderRequest(
            ticker=ticker,
            action=action,
            quantity=quantity,
            date=simulated_date
        )
        managed = ManagedOrder(order_req)
        self.order_history.append(managed)
        # Stable per-intent ID is sent to the gateway where supported to reduce
        # duplicate submissions after an ambiguous network outcome.
        client_order_id = managed.intent_id

        # 1. Pre-trade Risk Check
        portfolio = self.portfolio_tracker.get_portfolio_snapshot()
        available_cash = self.portfolio_tracker.get_cash_snapshot()

        is_approved, reject_msg = self.risk_engine.validate_order(
            order=order_req,
            price=estimated_price,
            portfolio=portfolio,
            available_cash=available_cash
        )

        if not is_approved:
            managed.state = OrderState.REJECTED
            managed.rejection_reason = f"Pre-trade risk: {reject_msg}"
            self._total_rejected += 1
            logger.info(f"Order REJECTED by risk engine: {action} {quantity} {ticker} ({reject_msg})")
            return managed

        managed.state = OrderState.VALIDATED

        # 2. Submit to API Gateway
        managed.state = OrderState.SUBMITTED
        managed.submitted_at = time.time()

        t0 = time.perf_counter()
        try:
            resp_dict = self.client.place_order(order_req, client_order_id=client_order_id)
            managed.roundtrip_ms = (time.perf_counter() - t0) * 1000.0
            managed.completed_at = time.time()

            status_code = resp_dict.get("status_code", 200)

            # Check HTTP and business status
            if status_code in (200, 201) and resp_dict.get("status") in ("FILLED", "SUCCESS", "ACCEPTED", None):
                managed.state = OrderState.FILLED
                managed.order_id = resp_dict.get("order_id")
                managed.execution_price = float(resp_dict.get("execution_price", resp_dict.get("price", estimated_price)))
                self._total_filled += 1
                logger.info(
                    f"Order FILLED: {action} {quantity} {ticker} @ {managed.execution_price:.2f} "
                    f"({managed.roundtrip_ms:.1f}ms)"
                )

                # Update local portfolio state immediately
                self.portfolio_tracker.record_fill(
                    ticker=ticker,
                    action=action,
                    quantity=quantity,
                    price=managed.execution_price
                )
            else:
                managed.state = OrderState.REJECTED
                managed.rejection_reason = resp_dict.get("message") or resp_dict.get("error") or f"HTTP {status_code}"
                self._total_rejected += 1
                logger.warning(
                    f"Order REJECTED by server: {action} {quantity} {ticker} -> {managed.rejection_reason}"
                )

        except Exception as e:
            managed.state = OrderState.UNKNOWN_OUTCOME
            managed.rejection_reason = f"Network/Timeout error: {e}"
            logger.error(f"Uncertain order outcome for {action} {quantity} {ticker}: {e}")
            # Try to safely reconcile with server order history
            self._reconcile_unknown_order(managed)

        return managed

    def _reconcile_unknown_order(self, managed: ManagedOrder) -> None:
        """Query official order history to check if the uncertain order actually executed."""
        try:
            time.sleep(1.0)
            orders = self.client.get_orders()
            # Never infer a fill from ticker/action/quantity alone: that can
            # accidentally match an older order. Prefer an exact idempotency ID.
            exact = [o for o in orders if str(o.get("client_order_id", o.get("clientOrderId", ""))) == managed.intent_id]
            if exact:
                o = exact[0]
                status = str(o.get("status", "")).upper()
                if status in {"FILLED", "SUCCESS", "EXECUTED", "COMPLETED"}:
                    managed.state = OrderState.FILLED
                    managed.order_id = o.get("order_id", o.get("id"))
                    managed.execution_price = float(o.get("execution_price", o.get("price", 0.0)) or 0.0)
                    self._total_filled += 1
                    self.portfolio_tracker.record_fill(ticker=managed.request.ticker, action=managed.request.action, quantity=managed.request.quantity, price=managed.execution_price)
                    logger.info("Reconciled unknown order using exact client ID %s", managed.intent_id)
                    return
                if status in {"REJECTED", "CANCELLED", "CANCELED", "FAILED"}:
                    managed.state = OrderState.REJECTED
                    managed.rejection_reason = f"Server reports order status {status}"
                    return
            # If gateway does not expose client IDs, keep outcome UNKNOWN rather
            # than risk double-counting a historical order or retrying blindly.
            managed.state = OrderState.UNKNOWN_OUTCOME
            managed.rejection_reason = "Could not safely reconcile: gateway order history did not expose matching client_order_id; manual/periodic sync required"
        except Exception as rec_err:
            logger.warning(f"Failed to reconcile unknown order: {rec_err}")

    @property
    def stats(self) -> dict:
        return {
            "total_submitted": len(self.order_history),
            "filled": self._total_filled,
            "rejected": self._total_rejected,
        }
