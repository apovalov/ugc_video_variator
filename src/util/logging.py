"""Structured logging utilities for the creative_variants CLI."""

from __future__ import annotations

import logging
import os
from typing import Any

_LOG_LEVEL = os.getenv("CREATIVE_VARIANTS_LOG_LEVEL", "INFO").upper()


def _configure_root_logger() -> None:
    """Idempotently configure the root logger with a consistent formatter."""
    if getattr(_configure_root_logger, "_configured", False):
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(_LOG_LEVEL)
    root.handlers = [handler]

    _configure_root_logger._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger with standard configuration applied."""
    _configure_root_logger()
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    return logger


def emit_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log a structured event line with key=value pairs for downstream parsing."""
    kv_pairs = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s %s", event, kv_pairs.strip())
