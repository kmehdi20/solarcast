"""Structured logging.

Deliberately built on the standard library: no extra dependency, and
records stay consumable by any collector (journald, Loki, CloudWatch).

Two formats:

* ``text``  — readable during development;
* ``json``  — one JSON line per record, for production.

Extra fields go through ``extra={"context": {...}}`` and are serialized in
both formats.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_CONFIGURED = False

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def _extract_context(record: logging.LogRecord) -> dict[str, Any]:
    """Collect the free-form fields attached to the record."""
    context = dict(getattr(record, "context", {}) or {})
    for key, value in record.__dict__.items():
        if key not in _RESERVED and key != "context":
            context[key] = value
    return context


class JsonFormatter(logging.Formatter):
    """Serializes each record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = _extract_context(record)
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format, with context appended at the end of the line."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)-28s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = _extract_context(record)
        if context:
            rendered = " ".join(f"{k}={v}" for k, v in sorted(context.items()))
            base = f"{base} | {rendered}"
        return base


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Install the root handler. Idempotent."""
    global _CONFIGURED

    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter() if json_format else TextFormatter())

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # HTTP clients are noisy at DEBUG.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Shortcut to get a logger named after a module."""
    return logging.getLogger(name)
