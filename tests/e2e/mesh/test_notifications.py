"""E2E: Notification delivery (best-effort)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synix.mesh.notify import send_notification


class TestNotifications:
    @pytest.mark.asyncio
    async def test_notification_sends_payload(self):
        """Notification sends correct event and detail payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("synix.mesh.notify.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await send_notification(
                webhook_url="https://hooks.example.com/test",
                source="test-mesh",
                event="build_complete",
                detail={"build_number": 1},
            )

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["event"] == "build_complete"

    @pytest.mark.asyncio
    async def test_notification_failure_does_not_raise(self):
        """Network failure in notification is logged, not raised."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("network down"))

        with patch("synix.mesh.notify.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            # Should not raise
            await send_notification(
                webhook_url="https://hooks.example.com/test",
                source="test-mesh",
                event="build_failed",
                detail={"error": "test"},
            )

    @pytest.mark.asyncio
    async def test_notification_empty_webhook_skips(self):
        """Empty webhook URL should skip notification."""
        # Should not raise or make any network calls
        await send_notification(
            webhook_url="",
            source="test-mesh",
            event="build_complete",
            detail={},
        )
