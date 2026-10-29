"""Resilient Kafka consumer for market ticks, news feeds, and pre-flight heartbeat ACKs."""
from __future__ import annotations
import json
import logging
import time
from typing import Callable, Optional, TYPE_CHECKING
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from trading_bot.api.schemas import MarketTick, NewsItem
from trading_bot.config import Settings
from trading_bot.market_data.normalizer import EventNormalizer

if TYPE_CHECKING:
    from trading_bot.api.client import TradingAPIClient

logger = logging.getLogger(__name__)


class MarketDataConsumer:
    """Consumes Kafka feeds, handles pre-flight heartbeat sequence ACKs, and passes normalized events."""

    def __init__(
        self,
        settings: Settings,
        api_client: TradingAPIClient,
        on_tick_callback: Optional[Callable[[MarketTick], None]] = None,
        on_news_callback: Optional[Callable[[NewsItem], None]] = None,
        on_preflight_callback: Optional[Callable[[int], None]] = None,
    ):
        self.settings = settings
        self.client = api_client
        self.on_tick_callback = on_tick_callback
        self.on_news_callback = on_news_callback
        self.on_preflight_callback = on_preflight_callback

        self._consumer: Optional[KafkaConsumer] = None
        self._is_running = False
        self._last_heartbeat_seq: Optional[int] = None
        self._messages_processed = 0

    def connect(self) -> bool:
        """Establish connection to Kafka broker."""
        broker = self.settings.kafka_bootstrap_servers
        group_id = self.settings.kafka_group_id
        topics = [self.settings.kafka_tick_topic, self.settings.kafka_news_topic]

        logger.info(f"Connecting KafkaConsumer to {broker} on topics {topics} (group={group_id})")
        try:
            self._consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=[broker],
                group_id=group_id,
                auto_offset_reset="latest",
                enable_auto_commit=False,
                consumer_timeout_ms=1000,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            logger.info("Kafka consumer connected successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect Kafka consumer: {e}")
            self._consumer = None
            return False

    def poll_once(self) -> int:
        """Poll once for messages and dispatch to registered callbacks."""
        if not self._consumer:
            if not self.connect():
                time.sleep(2.0)
                return 0

        try:
            records = self._consumer.poll(timeout_ms=500, max_records=50)
            if not records:
                return 0

            count = 0
            for topic_partition, msgs in records.items():
                for msg in msgs:
                    count += 1
                    self._messages_processed += 1
                    event = msg.value

                    if not isinstance(event, dict):
                        continue

                    event_type = event.get("event_type") or event.get("event")

                    # 1. Pre-flight heartbeat standby ping
                    if event_type == "preflight_tick":
                        seq = event.get("seq")
                        if seq is not None and seq != self._last_heartbeat_seq:
                            self._last_heartbeat_seq = seq
                            logger.info(f"Received pre-flight standby ping seq={seq}. Sending ACK...")
                            try:
                                ack_resp = self.client.send_heartbeat(seq=int(seq))
                                logger.info(f"Heartbeat ACK response: {ack_resp}")
                                if self.on_preflight_callback:
                                    self.on_preflight_callback(int(seq))
                            except Exception as hb_err:
                                logger.warning(f"Failed to ACK heartbeat seq={seq}: {hb_err}")

                    # 2. Price tick event
                    elif msg.topic == self.settings.kafka_tick_topic or event_type == "price_tick":
                        tick = EventNormalizer.normalize_tick_event(event)
                        if tick and self.on_tick_callback:
                            self.on_tick_callback(tick)

                    # 3. News flash event
                    elif msg.topic == self.settings.kafka_news_topic or event_type == "news_flash":
                        news = EventNormalizer.normalize_news_event(event)
                        if news and self.on_news_callback:
                            self.on_news_callback(news)

            return count

        except Exception as e:
            logger.warning(f"Error polling Kafka: {e}")
            return 0

    def run_loop(self) -> None:
        """Blocking loop continuously polling messages until stopped."""
        self._is_running = True
        logger.info("Starting market data consumer loop...")
        while self._is_running:
            self.poll_once()

    def stop(self) -> None:
        """Stop consumer loop and close Kafka connection."""
        self._is_running = False
        if self._consumer:
            try:
                self._consumer.close()
            except Exception:
                pass
            self._consumer = None
        logger.info("Kafka consumer stopped.")

    @property
    def messages_processed(self) -> int:
        return self._messages_processed
