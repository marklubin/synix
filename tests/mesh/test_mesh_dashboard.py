"""Tests for synix.mesh.dashboard — Rich dashboard rendering and log tailing."""

from __future__ import annotations

import json

import pytest

from synix.mesh.dashboard import MeshDashboard, tail_log_file

pytestmark = pytest.mark.mesh


@pytest.fixture
def mock_mesh_dir(tmp_path):
    """Create a minimal mesh directory with state.json and log file."""
    mesh_dir = tmp_path / "test-mesh"
    mesh_dir.mkdir()
    (mesh_dir / "logs").mkdir()

    # State file
    state = {
        "role": "server",
        "my_hostname": "obispo",
        "server_url": "http://obispo:7433",
        "term": {"counter": 3, "leader_id": "obispo"},
        "config_hash": "abc123",
    }
    (mesh_dir / "state.json").write_text(json.dumps(state))

    # Log file with JSON Lines
    entries = [
        {
            "ts": "2026-02-19T14:30:01+00:00",
            "level": "INFO",
            "logger": "synix.mesh.server",
            "msg": "Build #7 completed: built=12 cached=5",
            "event": "build_completed",
        },
        {
            "ts": "2026-02-19T14:30:02+00:00",
            "level": "INFO",
            "logger": "synix.mesh.server",
            "msg": "Bundle created: build.tar.gz",
            "event": "bundle_created",
        },
        {
            "ts": "2026-02-19T14:35:00+00:00",
            "level": "DEBUG",
            "logger": "synix.mesh.server",
            "msg": "Heartbeat from salinas",
            "event": "heartbeat_received",
        },
    ]
    log_path = mesh_dir / "logs" / "server.log"
    log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    return mesh_dir


class TestTailLogFile:
    def test_returns_parsed_entries(self, mock_mesh_dir):
        log_path = mock_mesh_dir / "logs" / "server.log"
        entries = tail_log_file(log_path, lines=10)
        assert len(entries) == 3
        assert entries[0]["event"] == "build_completed"

    def test_limits_lines(self, mock_mesh_dir):
        log_path = mock_mesh_dir / "logs" / "server.log"
        entries = tail_log_file(log_path, lines=2)
        assert len(entries) == 2
        # Should be the last 2 entries
        assert entries[0]["event"] == "bundle_created"

    def test_nonexistent_file(self, tmp_path):
        entries = tail_log_file(tmp_path / "nonexistent.log")
        assert entries == []

    def test_malformed_lines_skipped(self, tmp_path):
        log_path = tmp_path / "bad.log"
        log_path.write_text('{"valid": true}\nnot json\n{"also_valid": true}\n')
        entries = tail_log_file(log_path)
        assert len(entries) == 2

    def test_empty_file(self, tmp_path):
        log_path = tmp_path / "empty.log"
        log_path.write_text("")
        entries = tail_log_file(log_path)
        assert entries == []


class TestMeshDashboard:
    def test_creates_without_error(self, mock_mesh_dir):
        dash = MeshDashboard(
            mesh_dir=mock_mesh_dir,
            name="test-mesh",
        )
        assert dash.name == "test-mesh"

    def test_local_state_loaded(self, mock_mesh_dir):
        dash = MeshDashboard(mesh_dir=mock_mesh_dir, name="test-mesh")
        state = dash._get_local_state()
        assert state["role"] == "server"
        assert state["my_hostname"] == "obispo"

    def test_health_panel_without_server(self, mock_mesh_dir):
        dash = MeshDashboard(mesh_dir=mock_mesh_dir, name="test-mesh")
        # Simulate a failed refresh to trigger "server unreachable"
        dash.server_url = "http://localhost:19999"
        dash.refresh()
        panel = dash._build_health_panel()
        assert panel.title == "Health"
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=120)
        c.print(panel)
        output = buf.getvalue()
        assert "obispo" in output
        assert "server unreachable" in output

    def test_activity_panel_shows_log_entries(self, mock_mesh_dir):
        dash = MeshDashboard(mesh_dir=mock_mesh_dir, name="test-mesh", log_lines=10)
        panel = dash._build_activity_panel()
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=120)
        c.print(panel)
        output = buf.getvalue()
        assert "Build #7" in output
        assert "Bundle created" in output

    def test_members_panel_without_server(self, mock_mesh_dir):
        dash = MeshDashboard(mesh_dir=mock_mesh_dir, name="test-mesh")
        panel = dash._build_members_panel()
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=120)
        c.print(panel)
        output = buf.getvalue()
        assert "server unreachable" in output

    def test_rich_console_protocol(self, mock_mesh_dir):
        """Dashboard implements __rich_console__ for Live rendering."""
        dash = MeshDashboard(mesh_dir=mock_mesh_dir, name="test-mesh")
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=120)
        c.print(dash)
        output = buf.getvalue()
        assert "Synix Mesh: test-mesh" in output

    def test_no_state_file(self, tmp_path):
        """Dashboard handles missing state.json gracefully."""
        mesh_dir = tmp_path / "empty-mesh"
        mesh_dir.mkdir()
        (mesh_dir / "logs").mkdir()
        dash = MeshDashboard(mesh_dir=mesh_dir, name="empty")
        state = dash._get_local_state()
        assert state == {}

    def test_no_log_file(self, tmp_path):
        """Dashboard handles missing log files gracefully."""
        mesh_dir = tmp_path / "no-logs"
        mesh_dir.mkdir()
        (mesh_dir / "logs").mkdir()
        state = {"role": "client", "my_hostname": "test", "term": {"counter": 0, "leader_id": ""}, "server_url": ""}
        (mesh_dir / "state.json").write_text(json.dumps(state))
        dash = MeshDashboard(mesh_dir=mesh_dir, name="no-logs", log_lines=10)
        panel = dash._build_activity_panel()
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=120)
        c.print(panel)
        assert "No log entries" in buf.getvalue()
