"""Tests for mesh webhook notifications."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from synix.mesh.notify import send_notification


class TestSendNotification:
    @pytest.mark.asyncio
    async def test_successful_webhook_call(self):
        """Webhook POST succeeds and sends correct payload."""
        mock_response = httpx.Response(200, request=httpx.Request("POST", "http://hook"))

        with patch("synix.mesh.notify.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await send_notification("http://hook/notify", "test-mesh", "build_complete", {"build_id": "abc"})

            mock_client.post.assert_called_once_with(
                "http://hook/notify",
                json={
                    "source": "test-mesh",
                    "event": "build_complete",
                    "detail": {"build_id": "abc"},
                },
            )

    @pytest.mark.asyncio
    async def test_network_failure_logged_not_raised(self, caplog):
        """Network error is logged as warning, never raised."""
        with patch("synix.mesh.notify.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with caplog.at_level(logging.WARNING):
                await send_notification("http://unreachable/hook", "src", "evt", {})

            assert "Failed to send notification" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_webhook_url_returns_immediately(self, caplog):
        """Empty webhook URL skips sending entirely."""
        with caplog.at_level(logging.DEBUG):
            await send_notification("", "src", "evt", {"key": "val"})

        assert "No webhook URL configured" in caplog.text

    @pytest.mark.asyncio
    async def test_payload_format(self):
        """Verify the JSON payload matches expected schema."""
        captured_payload = {}
        mock_response = httpx.Response(200, request=httpx.Request("POST", "http://hook"))

        with patch("synix.mesh.notify.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def capture_post(url, json=None):
                captured_payload.update(json)
                return mock_response

            mock_client.post = capture_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await send_notification("http://hook/notify", "my-mesh", "deploy_done", {"version": "1.0"})

        assert captured_payload == {
            "source": "my-mesh",
            "event": "deploy_done",
            "detail": {"version": "1.0"},
        }

    @pytest.mark.asyncio
    async def test_http_error_status_logged(self, caplog):
        """Non-2xx HTTP response is logged as warning, not raised."""
        mock_response = httpx.Response(
            500,
            request=httpx.Request("POST", "http://hook"),
        )

        with patch("synix.mesh.notify.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with caplog.at_level(logging.WARNING):
                await send_notification("http://hook/notify", "src", "evt", {})

            assert "Failed to send notification" in caplog.text
