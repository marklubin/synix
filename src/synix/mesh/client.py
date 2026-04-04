"""Mesh client daemon — watches, pulls, heartbeats."""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import httpx

from synix.mesh.auth import auth_headers
from synix.mesh.cluster import (
    ClusterState,
    cluster_config_hash,
    elect_leader,
)
from synix.mesh.config import MeshConfig
from synix.mesh.deploy import run_deploy_hooks
from synix.mesh.logging import mesh_event
from synix.mesh.package import extract_bundle

logger = logging.getLogger(__name__)


# --- Incremental tracking dataclasses ---


@dataclass
class SubsessionBoundary:
    """Record of a submitted subsession's byte range in the source file."""

    seq: int
    byte_start: int
    byte_end: int


@dataclass
class FileTracker:
    """Tracks incremental scan state for a single JSONL file."""

    byte_offset: int = 0
    subsession_seq: int = 0
    last_activity: float = 0.0
    pending_turns: int = 0
    prefix_hash: str = ""
    prefix_size: int = 0  # Bytes used for prefix_hash (for stable comparison)
    inode: int = 0
    boundaries: list[SubsessionBoundary] = field(default_factory=list)


class MeshClient:
    """Async client daemon with watcher, puller, and heartbeat loops."""

    def __init__(self, config: MeshConfig):
        self.config = config
        self.server_url = ""  # Set from state.json or discovery
        self.state_path = config.mesh_dir / "state.json"
        self.cluster_state = ClusterState.load(self.state_path)
        self.server_url = self.cluster_state.server_url
        self._submitted: set[str] = set()  # sha256 hashes of submitted files
        self._file_trackers: dict[str, FileTracker] = {}  # path -> tracker
        self._current_etag = ""
        self._archive_etag = ""  # ETag for sessions manifest
        self._consecutive_heartbeat_failures = 0
        self._running = False
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Start all three async loops."""
        self._running = True
        self._http = httpx.AsyncClient(timeout=30)
        self._load_submitted_state()

        try:
            await asyncio.gather(
                self._watcher_loop(),
                self._puller_loop(),
                self._heartbeat_loop(),
            )
        finally:
            self._running = False
            if self._http:
                await self._http.aclose()

    async def stop(self) -> None:
        """Signal all loops to stop."""
        self._running = False

    # --- Watcher loop ---
    async def _watcher_loop(self):
        """Scan watch_dir for new files and submit to server."""
        consecutive_failures = 0
        while self._running:
            try:
                await self._scan_and_submit()
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                logger.warning("Watcher scan failed", exc_info=True)
            backoff = min(self.config.client.scan_interval * (2 ** min(consecutive_failures, 5)), 300)
            await asyncio.sleep(backoff)

    async def _scan_and_submit(self):
        """Scan watch_dir, dispatch to incremental or whole-file per file."""
        watch_dir = Path(os.path.expanduser(self.config.source.watch_dir))
        if not watch_dir.exists():
            return

        incremental = self.config.source.incremental
        now = time.time()

        for pattern in self.config.source.patterns:
            for file_path in watch_dir.glob(pattern):
                if not file_path.is_file():
                    continue
                # Check exclude patterns
                rel = str(file_path.relative_to(watch_dir))
                if any(fnmatch(rel, exc) for exc in self.config.source.exclude):
                    continue

                if incremental:
                    await self._scan_incremental(file_path, watch_dir, now)
                else:
                    await self._scan_whole_file(file_path, watch_dir)

        # Idle timeout flush — check all tracked files for pending turns
        if incremental:
            idle_timeout = self.config.source.idle_timeout
            for path_str, tracker in list(self._file_trackers.items()):
                if (
                    tracker.pending_turns > 0
                    and tracker.last_activity > 0
                    and (now - tracker.last_activity) >= idle_timeout
                ):
                    file_path = Path(path_str)
                    if file_path.exists():
                        await self._flush_subsession(file_path, watch_dir, tracker)

    async def _scan_whole_file(self, file_path: Path, watch_dir: Path):
        """Original whole-file hash-and-submit logic."""
        content = file_path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if sha in self._submitted:
            return
        await self._submit_file(file_path, content, sha)

    async def _scan_incremental(self, file_path: Path, watch_dir: Path, now: float):
        """Incremental offset-tracked delta extraction for a single file.

        1. stat() — skip if size unchanged
        2. Safety checks (truncation, inode change, prefix rewrite)
        3. Read new bytes from byte_offset
        4. Count user/assistant turns
        5. Flush if threshold reached
        """
        path_str = str(file_path)
        tracker = self._file_trackers.get(path_str)

        try:
            st = file_path.stat()
        except OSError:
            logger.warning("Cannot stat %s, skipping", file_path, exc_info=True)
            return

        file_size = st.st_size
        file_inode = st.st_ino

        # First time seeing this file
        if tracker is None:
            prefix_hash, prefix_size = self._compute_prefix_hash(file_path)
            tracker = FileTracker(
                byte_offset=0,
                subsession_seq=0,
                last_activity=now,
                pending_turns=0,
                prefix_hash=prefix_hash,
                prefix_size=prefix_size,
                inode=file_inode,
            )
            self._file_trackers[path_str] = tracker

        # Safety checks — run before the quick-return so rewrites are detected
        if file_size < tracker.byte_offset:
            logger.warning(
                "File truncated: %s (was %d bytes, now %d). Resetting tracker.",
                file_path.name,
                tracker.byte_offset,
                file_size,
            )
            self._hard_reset_tracker(tracker, file_path)
            return

        if file_inode != tracker.inode:
            logger.warning(
                "File replaced (inode changed): %s. Resetting tracker.",
                file_path.name,
            )
            self._hard_reset_tracker(tracker, file_path)
            return

        # Prefix integrity check — read same number of bytes as original
        current_prefix, _ = self._compute_prefix_hash(file_path, max_bytes=tracker.prefix_size or 4096)
        if current_prefix != tracker.prefix_hash:
            logger.warning(
                "File content rewritten (prefix hash changed): %s. Resetting tracker.",
                file_path.name,
            )
            self._hard_reset_tracker(tracker, file_path)
            return

        # No growth since last scan — file passed all safety checks
        if file_size == tracker.byte_offset:
            return

        # Read new bytes
        try:
            with open(file_path, "rb") as f:
                f.seek(tracker.byte_offset)
                new_bytes = f.read()
        except OSError:
            logger.warning("Failed to read %s at offset %d", file_path, tracker.byte_offset, exc_info=True)
            return

        if not new_bytes:
            return

        # Split on newlines, discard trailing partial line
        lines = new_bytes.split(b"\n")
        if not new_bytes.endswith(b"\n"):
            # Last chunk is incomplete — put it back
            incomplete = lines.pop()
            consumed = len(new_bytes) - len(incomplete)
        else:
            consumed = len(new_bytes)
            # Remove trailing empty string from split
            if lines and lines[-1] == b"":
                lines.pop()

        if not lines:
            return

        # Count user/assistant turns (lightweight JSONL scan)
        new_turns = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                role = obj.get("role", obj.get("type", ""))
                if role in ("user", "human"):
                    new_turns += 1
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON — count as a turn conservatively
                pass

        tracker.pending_turns += new_turns
        tracker.last_activity = now

        # Advance byte_offset past the consumed complete lines
        new_offset = tracker.byte_offset + consumed

        # Check flush condition: turn threshold
        if tracker.pending_turns >= self.config.source.min_turns:
            # Set offset tentatively for flush to read the right byte range,
            # but roll back if flush fails so data isn't skipped on retry.
            tracker.byte_offset = new_offset
            success = await self._flush_subsession(file_path, watch_dir, tracker)
            if not success:
                # Revert offset so these bytes are retried on next scan
                tracker.byte_offset = new_offset - consumed if consumed <= new_offset else 0
        else:
            # Just advance offset tracking but don't flush yet
            tracker.byte_offset = new_offset

        self._save_submitted_state()

    async def _flush_subsession(self, file_path: Path, watch_dir: Path, tracker: FileTracker) -> bool:
        """Package and submit the pending subsession content. Returns True on success."""
        # Determine byte range for this subsession
        if tracker.boundaries:
            byte_start = tracker.boundaries[-1].byte_end
        else:
            byte_start = 0
        byte_end = tracker.byte_offset

        if byte_end <= byte_start:
            return True  # Nothing to flush — not a failure

        # Read the subsession content
        try:
            with open(file_path, "rb") as f:
                f.seek(byte_start)
                content = f.read(byte_end - byte_start)
        except OSError:
            logger.warning("Failed to read subsession from %s", file_path, exc_info=True)
            return False

        if not content:
            return True

        seq = tracker.subsession_seq
        flushed_turns = tracker.pending_turns

        # Submit
        rel_path = file_path.relative_to(watch_dir)
        project_dir = str(rel_path.parent) if rel_path.parent != Path(".") else "default"
        session_id = file_path.stem

        success = await self._submit_subsession(
            session_id=session_id,
            project_dir=project_dir,
            content=content,
            subsession_seq=seq,
            file_name=file_path.name,
        )

        if success:
            boundary = SubsessionBoundary(seq=seq, byte_start=byte_start, byte_end=byte_end)
            tracker.boundaries.append(boundary)
            tracker.subsession_seq = seq + 1
            tracker.pending_turns = 0
            self._save_submitted_state()

            mesh_event(
                logger,
                logging.INFO,
                f"Flushed subsession {file_path.name} seq={seq} ({byte_end - byte_start} bytes)",
                "subsession_flushed",
                {
                    "file_name": file_path.name,
                    "seq": seq,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "turns": flushed_turns,
                },
            )

        return success

    async def _submit_subsession(
        self,
        session_id: str,
        project_dir: str,
        content: bytes,
        subsession_seq: int,
        file_name: str,
    ) -> bool:
        """Submit a subsession to the server. Returns True on success."""
        encoded = base64.b64encode(content).decode()
        sha256 = hashlib.sha256(content).hexdigest()

        payload = {
            "session_id": session_id,
            "project_dir": project_dir,
            "subsession_seq": subsession_seq,
            "content": encoded,
            "sha256": sha256,
        }

        try:
            headers = auth_headers(
                self.config.token,
                self.cluster_state.my_hostname,
                term_counter=self.cluster_state.term.counter,
            )
            resp = await self._http.post(
                f"{self.server_url}/api/v1/sessions",
                json=payload,
                headers=headers,
            )
            if resp.status_code in (200, 201):
                self._submitted.add(sha256)
                mesh_event(
                    logger,
                    logging.INFO,
                    f"Submitted {file_name} seq={subsession_seq}",
                    "file_submitted",
                    {
                        "file_name": file_name,
                        "subsession_seq": subsession_seq,
                        "sha256": sha256,
                    },
                )
                return True
            else:
                logger.warning(
                    "Submit failed for %s seq=%d: %s %s",
                    file_name,
                    subsession_seq,
                    resp.status_code,
                    resp.text,
                )
                return False
        except Exception:
            logger.warning("Failed to submit %s seq=%d", file_name, subsession_seq, exc_info=True)
            return False

    async def _submit_file(self, file_path: Path, content: bytes, sha256: str) -> None:
        """Submit a single file to the server (whole-file mode)."""
        encoded = base64.b64encode(content).decode()

        # Derive project_dir and session_id from path
        watch_dir = Path(os.path.expanduser(self.config.source.watch_dir))
        rel_path = file_path.relative_to(watch_dir)
        project_dir = str(rel_path.parent) if rel_path.parent != Path(".") else "default"
        session_id = file_path.stem

        payload = {
            "session_id": session_id,
            "project_dir": project_dir,
            "content": encoded,
            "sha256": sha256,
        }

        try:
            headers = auth_headers(
                self.config.token,
                self.cluster_state.my_hostname,
                term_counter=self.cluster_state.term.counter,
            )
            resp = await self._http.post(
                f"{self.server_url}/api/v1/sessions",
                json=payload,
                headers=headers,
            )
            if resp.status_code in (200, 201):
                self._submitted.add(sha256)
                self._save_submitted_state()
                mesh_event(
                    logger,
                    logging.INFO,
                    f"Submitted {file_path.name}",
                    "file_submitted",
                    {
                        "file_name": file_path.name,
                        "sha256": sha256,
                    },
                )
            else:
                logger.warning(
                    "Submit failed for %s: %s %s",
                    file_path.name,
                    resp.status_code,
                    resp.text,
                )
        except Exception:
            logger.warning("Failed to submit %s", file_path.name, exc_info=True)

    def _hard_reset_tracker(self, tracker: FileTracker, file_path: Path) -> None:
        """Reset a tracker after detecting file corruption/replacement."""
        tracker.byte_offset = 0
        tracker.subsession_seq = 0
        tracker.pending_turns = 0
        tracker.last_activity = 0.0
        tracker.boundaries.clear()
        prefix_hash, prefix_size = self._compute_prefix_hash(file_path)
        tracker.prefix_hash = prefix_hash
        tracker.prefix_size = prefix_size
        try:
            tracker.inode = file_path.stat().st_ino
        except OSError:
            tracker.inode = 0

    @staticmethod
    def _compute_prefix_hash(file_path: Path, max_bytes: int = 4096) -> tuple[str, int]:
        """SHA256 of first N bytes of a file for integrity checking.

        Returns (hash_hex, bytes_read) so callers can compare the same
        prefix size on subsequent checks.
        """
        try:
            with open(file_path, "rb") as f:
                prefix = f.read(max_bytes)
            return hashlib.sha256(prefix).hexdigest(), len(prefix)
        except OSError:
            return "", 0

    @staticmethod
    def _count_turns_in_lines(lines: list[bytes]) -> int:
        """Count user/assistant turns in JSONL lines."""
        turns = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                role = obj.get("role", obj.get("type", ""))
                if role in ("user", "human"):
                    turns += 1
            except (json.JSONDecodeError, ValueError):
                pass
        return turns

    # --- Puller loop ---
    async def _puller_loop(self):
        """Periodically check for new artifact bundles, then sync sessions."""
        consecutive_failures = 0
        while self._running:
            try:
                await self._pull_artifacts()
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                logger.warning("Pull failed", exc_info=True)
            # Session sync — lower priority, runs after each pull cycle
            try:
                await self._sync_sessions()
            except Exception:
                logger.warning("Session sync failed", exc_info=True)
            backoff = min(self.config.client.pull_interval * (2 ** min(consecutive_failures, 5)), 600)
            await asyncio.sleep(backoff)

    async def _pull_artifacts(self):
        """ETag check -> download bundle -> extract -> deploy."""
        headers = auth_headers(self.config.token, self.cluster_state.my_hostname)
        if self._current_etag:
            headers["If-None-Match"] = self._current_etag

        resp = await self._http.get(
            f"{self.server_url}/api/v1/artifacts/bundle",
            headers=headers,
        )

        if resp.status_code == 304:
            return  # No changes

        if resp.status_code == 200:
            # Save bundle and extract
            bundle_path = self.config.mesh_dir / "client" / "bundle.tar.gz"
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            bundle_path.write_bytes(resp.content)

            build_dir = self.config.mesh_dir / "client" / "build"
            extract_bundle(bundle_path, build_dir)

            self._current_etag = resp.headers.get("ETag", "")

            # Run client deploy hooks
            if self.config.deploy.client_commands:
                try:
                    run_deploy_hooks(self.config.deploy.client_commands, build_dir)
                except RuntimeError:
                    logger.warning("Client deploy hooks failed", exc_info=True)

            mesh_event(
                logger,
                logging.INFO,
                f"Artifacts pulled (etag={self._current_etag[:12]})",
                "artifacts_pulled",
                {
                    "etag": self._current_etag,
                    "bundle_size": len(resp.content),
                },
            )
        elif resp.status_code == 404:
            logger.debug("No bundle available yet")
        else:
            logger.warning("Bundle pull returned %s", resp.status_code)

    # --- Heartbeat loop ---
    async def _heartbeat_loop(self):
        """Send heartbeats to server, trigger election on failures."""
        while self._running:
            try:
                success = await self._send_heartbeat()
                if success:
                    self._consecutive_heartbeat_failures = 0
                else:
                    self._consecutive_heartbeat_failures += 1
                    if self._consecutive_heartbeat_failures >= 3:
                        mesh_event(
                            logger,
                            logging.WARNING,
                            "3 consecutive heartbeat failures — triggering election",
                            "election_triggered",
                            {"consecutive_failures": self._consecutive_heartbeat_failures},
                        )
                        await self._trigger_election()
                        self._consecutive_heartbeat_failures = 0
            except Exception:
                logger.warning("Heartbeat failed", exc_info=True)
                self._consecutive_heartbeat_failures += 1
                if self._consecutive_heartbeat_failures >= 3:
                    await self._trigger_election()
                    self._consecutive_heartbeat_failures = 0
            await asyncio.sleep(self.config.client.heartbeat_interval)

    async def _send_heartbeat(self) -> bool:
        """Send heartbeat to server. Returns True on success."""
        headers = auth_headers(self.config.token, self.cluster_state.my_hostname)
        config_hash = cluster_config_hash(self.config.cluster.leader_candidates)
        payload = {
            "hostname": self.cluster_state.my_hostname,
            "term": self.cluster_state.term.to_dict(),
            "config_hash": config_hash,
        }
        try:
            resp = await self._http.post(
                f"{self.server_url}/api/v1/heartbeat",
                json=payload,
                headers=headers,
            )
            return resp.status_code == 200
        except Exception:
            logger.warning("Heartbeat request failed", exc_info=True)
            return False

    async def _trigger_election(self):
        """Run leader election when heartbeats fail."""

        def ping_fn(host: str) -> bool:
            # Simple HTTP health check
            try:
                r = httpx.get(
                    f"http://{host}:{self.config.server.port}/api/v1/health",
                    timeout=5,
                )
                return r.status_code == 200
            except Exception:
                return False

        # elect_leader uses sync ping_fn — run in thread to avoid blocking event loop
        winner = await asyncio.to_thread(
            elect_leader,
            self.config.cluster.leader_candidates,
            ping_fn,
            self.cluster_state.my_hostname,
        )

        if winner == self.cluster_state.my_hostname:
            mesh_event(
                logger,
                logging.INFO,
                f"Election result: {winner} (self)",
                "election_result",
                {
                    "winner": winner,
                    "new_term": self.cluster_state.term.counter + 1,
                },
            )
            # Bootstrap server store from local session archive before promoting
            archive_dir = self.config.mesh_dir / "client" / "sessions-archive"
            if archive_dir.exists() and any(archive_dir.rglob("*.jsonl.gz")):
                from synix.mesh.store import SessionStore

                server_dir = self.config.mesh_dir / "server"
                server_dir.mkdir(parents=True, exist_ok=True)
                imported = SessionStore.bootstrap_from_archive(
                    db_path=server_dir / "sessions.db",
                    sessions_dir=server_dir / "sessions",
                    archive_dir=archive_dir,
                )
                logger.info("Bootstrapped %d sessions from archive for new leader", imported)

            self.cluster_state.term.counter += 1
            self.cluster_state.term.leader_id = self.cluster_state.my_hostname
            self.cluster_state.role = "server"
            self.cluster_state.save(self.state_path)
            # Server startup handled by caller/systemd
        elif winner:
            mesh_event(
                logger,
                logging.INFO,
                f"Election result: {winner}",
                "election_result",
                {
                    "winner": winner,
                    "new_term": self.cluster_state.term.counter + 1,
                },
            )
            self.server_url = f"http://{winner}:{self.config.server.port}"
            self.cluster_state.server_url = self.server_url
            self.cluster_state.term.counter += 1
            self.cluster_state.term.leader_id = winner
            self.cluster_state.save(self.state_path)
            # Re-submit all files to new leader — reset trackers but keep boundaries
            self._submitted.clear()
            self._replay_trackers_for_failover()
            self._save_submitted_state()
        else:
            logger.error("Election failed — no reachable candidates")

    def _replay_trackers_for_failover(self) -> None:
        """Reset trackers for leader failover, preserving boundaries for replay.

        Boundaries are kept intact so re-scan can replay identical subsessions
        (producing the same SHA256 per subsession for stable artifact labels).
        If a file's prefix changed during the failover, hard-reset everything.
        """
        for path_str, tracker in list(self._file_trackers.items()):
            file_path = Path(path_str)
            if not file_path.exists():
                del self._file_trackers[path_str]
                continue

            current_prefix, _ = self._compute_prefix_hash(file_path, max_bytes=tracker.prefix_size or 4096)
            if current_prefix != tracker.prefix_hash:
                # Content changed during failover — boundaries are meaningless
                logger.warning(
                    "File %s changed during failover, full reset",
                    file_path.name,
                )
                self._hard_reset_tracker(tracker, file_path)
            else:
                # Keep boundaries, reset offset to 0 for replay
                tracker.byte_offset = 0
                tracker.subsession_seq = 0
                tracker.pending_turns = 0

    # --- Session sync ---
    async def _sync_sessions(self):
        """Sync session archive from server. Downloads missing sessions."""
        headers = auth_headers(self.config.token, self.cluster_state.my_hostname)
        if self._archive_etag:
            headers["If-None-Match"] = self._archive_etag

        try:
            resp = await self._http.get(
                f"{self.server_url}/api/v1/sessions/manifest",
                headers=headers,
            )
        except Exception:
            logger.debug("Session manifest fetch failed", exc_info=True)
            return

        if resp.status_code == 304:
            return  # No changes

        if resp.status_code != 200:
            logger.debug("Session manifest returned %s", resp.status_code)
            return

        self._archive_etag = resp.headers.get("ETag", "")
        manifest = resp.json()
        server_sessions = manifest.get("sessions", [])

        # Load local manifest
        archive_dir = self.config.mesh_dir / "client" / "sessions-archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        local_manifest = self._load_archive_manifest()

        # Find missing sessions
        downloaded = 0
        for session in server_sessions:
            sid = session["session_id"]
            pdir = session["project_dir"]
            sha256 = session["jsonl_sha256"]
            seq = session.get("subsession_seq", 0)
            key = f"{sid}:{pdir}:{seq}"

            if key in local_manifest and local_manifest[key]["sha256"] == sha256:
                continue  # Already have this one

            # Download individual session file
            try:
                dl_headers = auth_headers(self.config.token, self.cluster_state.my_hostname)
                dl_resp = await self._http.get(
                    f"{self.server_url}/api/v1/sessions/{sid}/file",
                    params={"project_dir": pdir, "subsession_seq": str(seq)},
                    headers=dl_headers,
                )
                if dl_resp.status_code != 200:
                    logger.warning("Failed to download session %s seq=%d: %s", sid, seq, dl_resp.status_code)
                    continue

                # Verify SHA-256 of decompressed content
                gz_bytes = dl_resp.content
                try:
                    decompressed = gzip.decompress(gz_bytes)
                except Exception:
                    logger.warning("Failed to decompress session %s seq=%d", sid, seq, exc_info=True)
                    continue

                actual_sha = hashlib.sha256(decompressed).hexdigest()
                if actual_sha != sha256:
                    logger.warning(
                        "SHA-256 mismatch for session %s seq=%d: expected %s got %s",
                        sid,
                        seq,
                        sha256[:12],
                        actual_sha[:12],
                    )
                    continue

                # Write to archive
                session_dir = archive_dir / pdir
                session_dir.mkdir(parents=True, exist_ok=True)
                from synix.mesh.store import SessionStore

                fname = SessionStore._file_name(sid, seq)
                (session_dir / fname).write_bytes(gz_bytes)

                local_manifest[key] = {"project_dir": pdir, "sha256": sha256, "subsession_seq": seq}
                downloaded += 1

            except Exception:
                logger.warning("Failed to download session %s seq=%d", sid, seq, exc_info=True)

        if downloaded > 0:
            self._save_archive_manifest(local_manifest)
            mesh_event(
                logger,
                logging.INFO,
                f"Session sync: downloaded {downloaded}/{len(server_sessions)}",
                "session_sync",
                {"downloaded_count": downloaded, "total_count": len(server_sessions)},
            )

    def _load_archive_manifest(self) -> dict[str, dict]:
        """Load local archive manifest. Returns {key: {project_dir, sha256}}."""
        manifest_path = self.config.mesh_dir / "client" / "sessions-archive" / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text())
            return data.get("sessions", {})
        except Exception:
            logger.warning("Failed to load archive manifest", exc_info=True)
            return {}

    def _save_archive_manifest(self, sessions: dict[str, dict]) -> None:
        """Save local archive manifest."""
        manifest_path = self.config.mesh_dir / "client" / "sessions-archive" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"etag": self._archive_etag, "sessions": sessions}
        manifest_path.write_text(json.dumps(data, indent=2))

    # --- State persistence ---
    def _load_submitted_state(self):
        """Load submitted session hashes and file trackers from disk."""
        state_file = self.config.mesh_dir / "client" / "submitted_sessions.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self._submitted = set(data.get("sha256_hashes", []))
                # Load file trackers
                for path_str, tracker_data in data.get("file_trackers", {}).items():
                    boundaries = [
                        SubsessionBoundary(
                            seq=b["seq"],
                            byte_start=b["byte_start"],
                            byte_end=b["byte_end"],
                        )
                        for b in tracker_data.get("boundaries", [])
                    ]
                    self._file_trackers[path_str] = FileTracker(
                        byte_offset=tracker_data.get("byte_offset", 0),
                        subsession_seq=tracker_data.get("subsession_seq", 0),
                        last_activity=tracker_data.get("last_activity", 0.0),
                        pending_turns=tracker_data.get("pending_turns", 0),
                        prefix_hash=tracker_data.get("prefix_hash", ""),
                        prefix_size=tracker_data.get("prefix_size", 0),
                        inode=tracker_data.get("inode", 0),
                        boundaries=boundaries,
                    )
            except Exception:
                logger.warning("Failed to load submitted state", exc_info=True)
                self._submitted = set()
                self._file_trackers = {}

    def _save_submitted_state(self):
        """Save submitted session hashes and file trackers to disk."""
        state_file = self.config.mesh_dir / "client" / "submitted_sessions.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        trackers_data = {}
        for path_str, tracker in self._file_trackers.items():
            trackers_data[path_str] = {
                "byte_offset": tracker.byte_offset,
                "subsession_seq": tracker.subsession_seq,
                "last_activity": tracker.last_activity,
                "pending_turns": tracker.pending_turns,
                "prefix_hash": tracker.prefix_hash,
                "prefix_size": tracker.prefix_size,
                "inode": tracker.inode,
                "boundaries": [
                    {
                        "seq": b.seq,
                        "byte_start": b.byte_start,
                        "byte_end": b.byte_end,
                    }
                    for b in tracker.boundaries
                ],
            }

        state_file.write_text(
            json.dumps(
                {
                    "sha256_hashes": sorted(self._submitted),
                    "file_trackers": trackers_data,
                },
                indent=2,
            )
        )
