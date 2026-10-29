"""Structured logging configuration."""
import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """JSON log formatter that redacts sensitive fields."""
    SENSITIVE_KEYS = {'authorization', 'token', 'password', 'secret', 'api_token', 'jwt'}
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = str(record.exc_info[1])
        if hasattr(record, 'extra_data'):
            log_entry["data"] = self._redact(record.extra_data)
        return json.dumps(log_entry, default=str)
    
    def _redact(self, data: dict) -> dict:
        redacted = {}
        for k, v in data.items():
            if k.lower() in self.SENSITIVE_KEYS:
                redacted[k] = '****REDACTED****'
            elif isinstance(v, dict):
                redacted[k] = self._redact(v)
            else:
                redacted[k] = v
        return redacted


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.handlers = [handler]
    
    # Quiet noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('kafka').setLevel(logging.WARNING)
