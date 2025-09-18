"""Centralized logging helpers for Quantisity Capital.

Provides structured JSON logging with contextual adapters so that every
subsystem emits consistent, machine-parseable records.
"""

from __future__ import annotations

import json
import logging
import logging.config
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


_RESERVED_ATTRS: Iterable[str] = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}

_RESERVED_RENAMES = {
    "module": "module_name",
    "filename": "file_name",
    "funcName": "function_name",
}


def _sanitize_extra(extra: Mapping[str, Any]) -> Dict[str, Any]:
    """Rename or prefix keys that would collide with LogRecord attributes."""

    sanitized: Dict[str, Any] = {}
    for key, value in extra.items():
        if key in _RESERVED_ATTRS:
            new_key = _RESERVED_RENAMES.get(key, f"extra_{key}")
            sanitized[new_key] = value
        else:
            sanitized[key] = value
    return sanitized


class StructuredFormatter(logging.Formatter):
    """Render log records as JSON with consistent fields."""

    def __init__(self, environment: str = "development", default_fields: Optional[Mapping[str, Any]] = None):
        super().__init__()
        self.environment = environment
        self.default_fields = dict(default_fields or {})

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring not needed
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "file": record.filename,
            "line": record.lineno,
            "environment": self.environment,
            "message": record.getMessage(),
        }

        payload.update(self.default_fields)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_ATTRS and not key.startswith("_")
        }
        if extras:
            payload.update(extras)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=self._serialize)

    @staticmethod
    def _serialize(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            if isinstance(value, datetime):
                return value.isoformat()
            return str(value)


class HumanReadableFormatter(logging.Formatter):
    """Compact key/value formatter for console readability."""

    def __init__(self, *, show_environment: bool = True):
        super().__init__()
        self.show_environment = show_environment

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring not needed
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        component = getattr(record, "component", None)
        subsystem = getattr(record, "subsystem", None)
        context = ".".join(part for part in (component, subsystem) if part)
        message = record.getMessage()

        parts = [f"{timestamp}", f"{level:<5}"]
        if context:
            parts.append(context)
        parts.append(message)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_ATTRS and key not in {"component", "subsystem"}
        }

        if not self.show_environment:
            extras.pop("environment", None)

        if extras:
            formatted_extras = " ".join(
                f"{key}={self._stringify(value)}" for key, value in sorted(extras.items())
            )
            parts.append(formatted_extras)

        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        return " | ".join(parts)

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, default=str)
            except TypeError:
                return str(value)
        return str(value)


@dataclass
class LoggingContext:
    """Default metadata applied to every record produced by a logger adapter."""

    component: str
    subsystem: Optional[str] = None
    environment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"component": self.component}
        if self.subsystem:
            data["subsystem"] = self.subsystem
        if self.environment:
            data["environment"] = self.environment
        return data


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Injects consistent contextual metadata into every log record."""

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]):  # noqa: D401 - inherited docstring not needed
        kwargs = dict(kwargs)
        extra = dict(self.extra)
        provided_extra = kwargs.get("extra") or {}
        extra.update(provided_extra)
        kwargs["extra"] = _sanitize_extra(extra)
        return msg, kwargs


def setup_logging(config: Mapping[str, Any], *, environment: str = "development") -> None:
    """Configure the logging subsystem using app configuration.

    This function is safe to call multiple times; later calls override earlier
    configurations (useful when the full configuration is not yet available
    during bootstrap).
    """

    logging_config = dict(config.get("logging", {})) if config else {}

    log_level = logging_config.get("level", "INFO").upper()
    file_path = logging_config.get("file_path", "logs/quantisity_capital.log")
    max_bytes = int(logging_config.get("max_bytes", 10 * 1024 * 1024))
    backup_count = int(logging_config.get("backup_count", 5))
    console_enabled = bool(logging_config.get("console", True))

    log_file = Path(file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: Dict[str, Dict[str, Any]] = {
        "app_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
            "formatter": "structured",
            "level": log_level,
        }
    }

    if console_enabled:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "console",
            "level": log_level,
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "()": "logging_utils.StructuredFormatter",
                    "environment": environment,
                },
                "console": {
                    "()": "logging_utils.HumanReadableFormatter",
                    "show_environment": False,
                },
            },
            "handlers": handlers,
            "root": {
                "level": log_level,
                "handlers": list(handlers.keys()),
            },
        }
    )


def get_logger(name: str, *, component: str, subsystem: Optional[str] = None, environment: Optional[str] = None):
    """Return a logger adapter that injects the standard structured context."""

    base_logger = logging.getLogger(name)
    context = LoggingContext(component=component, subsystem=subsystem, environment=environment)
    return ContextLoggerAdapter(base_logger, context.to_dict())
