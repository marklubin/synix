"""Tests for MeshClient incremental scanning logic."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock

import pytest

from synix.mesh.client import FileTracker, MeshClient, SubsessionBoundary
from synix.mesh.config import ClientConfig, MeshConfig, SourceConfig

pytestmark = pytest.mark.mesh


def _make_jsonl_lines(pairs: int = 4) -> bytes:
    """Create JSONL content with N user/assistant pairs."""
    lines = []
    for i in range(pairs):
        lines.append(json.dumps({"role": "user", "content": f"msg {i}"}))
        lines.append(json.dumps({"role": "assistant", "content": f"reply {i}"}))
    return ("\n".join(lines) + "\n").encode()


class TestFileTracker:
    def test_defaults(self):
        t = FileTracker()
        assert t.byte_offset == 0
        assert t.subsession_seq == 0
        assert t.pending_turns == 0
        assert t.boundaries == []

    def test_subsession_boundary(self):
        b = SubsessionBoundary(seq=0, byte_start=0, byte_end=100)
        assert b.seq == 0
        assert b.byte_start == 0
        assert b.byte_end == 100


class TestComputePrefixHash:
    def test_stable(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_bytes(b"x" * 8192)
        h1, s1 = MeshClient._compute_prefix_hash(f)
        h2, s2 = MeshClient._compute_prefix_hash(f)
        assert h1 == h2
        assert s1 == s2 == 4096  # capped at 4KB
        assert len(h1) == 64  # SHA256 hex

    def test_different_prefix_different_hash(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_bytes(b"aaa" * 2000)
        h1, _ = MeshClient._compute_prefix_hash(f)
        f.write_bytes(b"bbb" * 2000)
        h2, _ = MeshClient._compute_prefix_hash(f)
        assert h1 != h2

    def test_missing_file_returns_empty(self, tmp_path):
        f = tmp_path / "nonexistent.jsonl"
        h, s = MeshClient._compute_prefix_hash(f)
        assert h == ""
        assert s == 0

    def test_small_file_uses_actual_size(self, tmp_path):
        f = tmp_path / "tiny.jsonl"
        f.write_bytes(b"small")
        h, s = MeshClient._compute_prefix_hash(f)
        assert s == 5
        assert len(h) == 64


class TestHardResetTracker:
    def test_resets_all_fields(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_bytes(b"hello")

        tracker = FileTracker(
            byte_offset=500,
            subsession_seq=3,
            pending_turns=2,
            last_activity=time.time(),
            boundaries=[SubsessionBoundary(0, 0, 100)],
        )

        config = MeshConfig(name="test")
        client = MeshClient(config)
        client._hard_reset_tracker(tracker, f)

        assert tracker.byte_offset == 0
        assert tracker.subsession_seq == 0
        assert tracker.pending_turns == 0
        assert tracker.last_activity == 0.0
        assert tracker.boundaries == []
        assert tracker.prefix_hash != ""
        assert tracker.inode != 0


class TestIncrementalScan:
    """Test _scan_incremental logic using direct method calls."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(
            name="test",
            source=SourceConfig(
                watch_dir=str(tmp_path / "watch"),
                incremental=True,
                min_turns=4,
                idle_timeout=120,
            ),
            client=ClientConfig(scan_interval=5),
        )
        c = MeshClient(config)
        c._http = AsyncMock()
        c.server_url = "http://localhost:7433"
        return c

    @pytest.fixture
    def watch_dir(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir(parents=True)
        return d

    @pytest.mark.asyncio
    async def test_first_scan_creates_tracker(self, client, watch_dir):
        f = watch_dir / "session.jsonl"
        # Only 2 turns — below min_turns=4, no flush
        f.write_bytes(_make_jsonl_lines(pairs=2))

        await client._scan_incremental(f, watch_dir, time.time())

        path_str = str(f)
        assert path_str in client._file_trackers
        tracker = client._file_trackers[path_str]
        assert tracker.byte_offset > 0
        assert tracker.pending_turns == 2
        assert tracker.subsession_seq == 0  # not flushed

    @pytest.mark.asyncio
    async def test_flush_on_min_turns(self, client, watch_dir):
        f = watch_dir / "session.jsonl"
        f.write_bytes(_make_jsonl_lines(pairs=5))

        # Mock the HTTP submit to succeed
        mock_resp = AsyncMock()
        mock_resp.status_code = 201
        client._http.post = AsyncMock(return_value=mock_resp)

        await client._scan_incremental(f, watch_dir, time.time())

        tracker = client._file_trackers[str(f)]
        assert tracker.subsession_seq == 1  # advanced after flush
        assert tracker.pending_turns == 0
        assert len(tracker.boundaries) == 1
        assert tracker.boundaries[0].seq == 0

    @pytest.mark.asyncio
    async def test_no_flush_below_threshold(self, client, watch_dir):
        f = watch_dir / "session.jsonl"
        f.write_bytes(_make_jsonl_lines(pairs=2))

        await client._scan_incremental(f, watch_dir, time.time())

        tracker = client._file_trackers[str(f)]
        assert tracker.subsession_seq == 0
        assert len(tracker.boundaries) == 0

    @pytest.mark.asyncio
    async def test_incremental_growth(self, client, watch_dir):
        f = watch_dir / "session.jsonl"
        # First write: 2 turns
        content1 = _make_jsonl_lines(pairs=2)
        f.write_bytes(content1)

        await client._scan_incremental(f, watch_dir, time.time())
        tracker = client._file_trackers[str(f)]
        assert tracker.pending_turns == 2
        assert tracker.subsession_seq == 0

        # Append: 3 more turns (total 5 >= min_turns=4)
        content2 = _make_jsonl_lines(pairs=3)
        with open(f, "ab") as fh:
            fh.write(content2)

        mock_resp = AsyncMock()
        mock_resp.status_code = 201
        client._http.post = AsyncMock(return_value=mock_resp)

        await client._scan_incremental(f, watch_dir, time.time())
        assert tracker.subsession_seq == 1
        assert tracker.pending_turns == 0

    @pytest.mark.asyncio
    async def test_no_rescan_if_unchanged(self, client, watch_dir):
        f = watch_dir / "session.jsonl"
        f.write_bytes(_make_jsonl_lines(pairs=2))

        await client._scan_incremental(f, watch_dir, time.time())
        initial_offset = client._file_trackers[str(f)].byte_offset

        # Second scan — no growth
        await client._scan_incremental(f, watch_dir, time.time())
        assert client._file_trackers[str(f)].byte_offset == initial_offset

    @pytest.mark.asyncio
    async def test_truncation_resets_tracker(self, client, watch_dir):
        f = watch_dir / "session.jsonl"
        f.write_bytes(_make_jsonl_lines(pairs=5))

        # Mock a successful submit so byte_offset advances past the flush
        mock_response = AsyncMock()
        mock_response.status_code = 200
        client._http.post = AsyncMock(return_value=mock_response)

        await client._scan_incremental(f, watch_dir, time.time())
        assert client._file_trackers[str(f)].byte_offset > 0

        # Truncate
        f.write_bytes(b"short\n")

        await client._scan_incremental(f, watch_dir, time.time())
        tracker = client._file_trackers[str(f)]
        assert tracker.byte_offset == 0
        assert tracker.boundaries == []

    @pytest.mark.asyncio
    async def test_prefix_rewrite_resets_tracker(self, client, watch_dir):
        f = watch_dir / "session.jsonl"
        original = _make_jsonl_lines(pairs=3)
        f.write_bytes(original)

        await client._scan_incremental(f, watch_dir, time.time())
        old_prefix = client._file_trackers[str(f)].prefix_hash

        # Rewrite file with different prefix but same length
        different = b"x" * len(original)
        f.write_bytes(different)

        await client._scan_incremental(f, watch_dir, time.time())
        tracker = client._file_trackers[str(f)]
        assert tracker.byte_offset == 0  # reset
        assert tracker.prefix_hash != old_prefix


class TestIdleTimeoutFlush:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(
            name="test",
            source=SourceConfig(
                watch_dir=str(tmp_path / "watch"),
                incremental=True,
                min_turns=4,
                idle_timeout=10,  # Short for testing
            ),
        )
        c = MeshClient(config)
        c._http = AsyncMock()
        c.server_url = "http://localhost:7433"
        return c

    @pytest.fixture
    def watch_dir(self, tmp_path):
        d = tmp_path / "watch"
        d.mkdir(parents=True)
        return d

    @pytest.mark.asyncio
    async def test_idle_flush(self, client, watch_dir):
        f = watch_dir / "session.jsonl"
        f.write_bytes(_make_jsonl_lines(pairs=2))

        # Scan — picks up 2 turns, below threshold
        now = time.time()
        await client._scan_incremental(f, watch_dir, now)
        tracker = client._file_trackers[str(f)]
        assert tracker.pending_turns == 2
        assert tracker.subsession_seq == 0

        # Simulate idle timeout by setting last_activity far in the past
        tracker.last_activity = now - 20  # 20s ago, exceeds idle_timeout=10

        mock_resp = AsyncMock()
        mock_resp.status_code = 201
        client._http.post = AsyncMock(return_value=mock_resp)

        # Full scan_and_submit checks idle timeout
        await client._scan_and_submit()

        assert tracker.subsession_seq == 1
        assert tracker.pending_turns == 0


class TestStatePersistence:
    def test_save_and_load_trackers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="test")

        # Save
        c1 = MeshClient(config)
        c1._file_trackers["/path/to/file.jsonl"] = FileTracker(
            byte_offset=500,
            subsession_seq=2,
            last_activity=1234567890.0,
            pending_turns=1,
            prefix_hash="abc123",
            inode=99999,
            boundaries=[
                SubsessionBoundary(seq=0, byte_start=0, byte_end=200),
                SubsessionBoundary(seq=1, byte_start=200, byte_end=500),
            ],
        )
        c1._submitted = {"hash1", "hash2"}
        c1._save_submitted_state()

        # Load
        c2 = MeshClient(config)
        c2._load_submitted_state()

        assert c2._submitted == {"hash1", "hash2"}
        assert "/path/to/file.jsonl" in c2._file_trackers

        t = c2._file_trackers["/path/to/file.jsonl"]
        assert t.byte_offset == 500
        assert t.subsession_seq == 2
        assert t.pending_turns == 1
        assert t.prefix_hash == "abc123"
        assert t.inode == 99999
        assert len(t.boundaries) == 2
        assert t.boundaries[0].byte_end == 200

    def test_load_legacy_state_without_trackers(self, tmp_path, monkeypatch):
        """Old state files without file_trackers still load correctly."""
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="test")

        state_file = config.mesh_dir / "client" / "submitted_sessions.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"sha256_hashes": ["old_hash"]}))

        c = MeshClient(config)
        c._load_submitted_state()
        assert c._submitted == {"old_hash"}
        assert c._file_trackers == {}


class TestReplayTrackersForFailover:
    def test_replay_preserves_boundaries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="test")

        f = tmp_path / "file.jsonl"
        f.write_bytes(b"x" * 1000)

        client = MeshClient(config)
        prefix_hash, prefix_size = MeshClient._compute_prefix_hash(f)

        client._file_trackers[str(f)] = FileTracker(
            byte_offset=500,
            subsession_seq=2,
            pending_turns=1,
            prefix_hash=prefix_hash,
            prefix_size=prefix_size,
            inode=f.stat().st_ino,
            boundaries=[
                SubsessionBoundary(seq=0, byte_start=0, byte_end=250),
                SubsessionBoundary(seq=1, byte_start=250, byte_end=500),
            ],
        )

        client._replay_trackers_for_failover()

        t = client._file_trackers[str(f)]
        assert t.byte_offset == 0  # Reset for replay
        assert t.subsession_seq == 0
        assert t.pending_turns == 0
        assert len(t.boundaries) == 2  # Preserved

    def test_replay_resets_on_changed_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="test")

        f = tmp_path / "file.jsonl"
        f.write_bytes(b"original" * 500)

        client = MeshClient(config)

        client._file_trackers[str(f)] = FileTracker(
            byte_offset=500,
            subsession_seq=2,
            prefix_hash="wrong_hash",  # Doesn't match current file
            inode=f.stat().st_ino,
            boundaries=[SubsessionBoundary(seq=0, byte_start=0, byte_end=250)],
        )

        client._replay_trackers_for_failover()

        t = client._file_trackers[str(f)]
        assert t.byte_offset == 0
        assert t.boundaries == []  # Cleared

    def test_replay_removes_deleted_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNIX_MESH_ROOT", str(tmp_path))
        config = MeshConfig(name="test")

        client = MeshClient(config)
        client._file_trackers["/nonexistent/file.jsonl"] = FileTracker(
            byte_offset=100,
        )

        client._replay_trackers_for_failover()

        assert "/nonexistent/file.jsonl" not in client._file_trackers
