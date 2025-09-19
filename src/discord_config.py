"""Utility helpers for loading Discord webhook secrets.

Secrets are stored in a separate YAML file (default: ``config/discord_webhooks.yaml``)
to keep deployment credentials outside of the main configuration that is
committed to source control. This module centralises the loading logic so
consumers can rely on a common schema and error handling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_WEBHOOKS_FILE = Path("config/discord_webhooks.yaml")


class DiscordWebhookConfigError(RuntimeError):
    """Raised when the webhook configuration cannot be loaded or parsed."""


def load_discord_webhooks(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the Discord webhook secrets file.

    Args:
        path: Optional override path. When not provided, the ``DISCORD_WEBHOOKS_FILE``
            environment variable is consulted and falls back to
            ``config/discord_webhooks.yaml``.

    Returns:
        Parsed webhook mapping. The expected structure is ``{"webhooks": {…}}``.

    Raises:
        DiscordWebhookConfigError: When the file exists but cannot be parsed as YAML.
    """

    resolved_path = _resolve_path(path)
    if resolved_path is None:
        return {}

    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive logging
        raise DiscordWebhookConfigError(f"Failed to parse Discord webhook YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise DiscordWebhookConfigError("Discord webhook configuration must be a mapping")

    webhooks = data.get("webhooks")
    if webhooks is None:
        return {}
    if not isinstance(webhooks, dict):
        raise DiscordWebhookConfigError("'webhooks' section must be a mapping")

    return webhooks


def _resolve_path(path: Optional[str]) -> Optional[Path]:
    """Resolve the webhook configuration path.

    Returns ``None`` when no file can be found or access is not configured. This
    allows services to run in environments where Discord publishing is optional.
    """

    if path:
        candidate = Path(path)
    else:
        env_path = os.getenv("DISCORD_WEBHOOKS_FILE")
        candidate = Path(env_path) if env_path else DEFAULT_WEBHOOKS_FILE

    if not candidate.exists():
        return None

    return candidate


__all__ = ["load_discord_webhooks", "DiscordWebhookConfigError", "DEFAULT_WEBHOOKS_FILE"]

