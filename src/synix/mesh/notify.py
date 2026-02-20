"""Webhook notifications for mesh events — best-effort, never raises."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


async def send_notification(webhook_url: str, source: str, event: str, detail: dict) -> None:
    """Send a notification via webhook. Best-effort — logs failures, never raises.

    Posts JSON payload ``{"source": source, "event": event, "detail": detail}``
    to *webhook_url*.  Empty *webhook_url* is a no-op.
    """
    if not webhook_url:
        logger.debug("No webhook URL configured, skipping notification")
        return

    payload = {"source": source, "event": event, "detail": detail}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
            logger.debug("Notification sent: event=%s status=%d", event, resp.status_code)
    except Exception:
        logger.warning(
            "Failed to send notification to %s (event=%s)",
            webhook_url,
            event,
            exc_info=True,
        )
