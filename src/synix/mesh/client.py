"""Mesh client daemon — watches, pulls, heartbeats."""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import logging
import os
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
from synix.mesh.package import extract_bundle

logger = logging.getLogger(__name__)


class MeshClient:
    """Async client daemon with watcher, puller, and heartbeat loops."""

    def __init__(self, config: MeshConfig):
        self.config = config
        self.server_url = ""  # Set from state.json or discovery
        self.state_path = config.mesh_dir / "state.json"
        self.cluster_state = ClusterState.load(self.state_path)
        self.server_url = self.cluster_state.server_url
        self._submitted: set[str] = set()  # sha256 hashes of submitted files
        self._current_etag = ""
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
        while self._running:
            try:
                await self._scan_and_submit()
            except Exception:
                logger.warning("Watcher scan failed", exc_info=True)
            await asyncio.sleep(self.config.client.scan_interval)

    async def _scan_and_submit(self):
        """Scan watch_dir, compute sha256, submit new files."""
        watch_dir = Path(os.path.expanduser(self.config.source.watch_dir))
        if not watch_dir.exists():
            return

        for pattern in self.config.source.patterns:
            for file_path in watch_dir.glob(pattern):
                if not file_path.is_file():
                    continue
                # Check exclude patterns
                rel = str(file_path.relative_to(watch_dir))
                if any(fnmatch(rel, exc) for exc in self.config.source.exclude):
                    continue

                content = file_path.read_bytes()
                sha = hashlib.sha256(content).hexdigest()
                if sha in self._submitted:
                    continue

                await self._submit_file(file_path, content, sha)

    async def _submit_file(self, file_path: Path, content: bytes, sha256: str) -> None:
        """Submit a single file to the server."""
        compressed = gzip.compress(content)
        encoded = base64.b64encode(compressed).decode()

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
            headers = auth_headers(self.config.token, self.cluster_state.my_hostname)
            resp = await self._http.post(
                f"{self.server_url}/api/v1/sessions",
                json=payload,
                headers=headers,
            )
            if resp.status_code == 200:
                self._submitted.add(sha256)
                self._save_submitted_state()
                logger.info("Submitted %s (sha256=%s...)", file_path.name, sha256[:12])
            else:
                logger.warning(
                    "Submit failed for %s: %s %s",
                    file_path.name,
                    resp.status_code,
                    resp.text,
                )
        except Exception:
            logger.warning("Failed to submit %s", file_path.name, exc_info=True)

    # --- Puller loop ---
    async def _puller_loop(self):
        """Periodically check for new artifact bundles."""
        while self._running:
            try:
                await self._pull_artifacts()
            except Exception:
                logger.warning("Pull failed", exc_info=True)
            await asyncio.sleep(self.config.client.pull_interval)

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

            logger.info(
                "Pulled and deployed new artifacts (etag=%s)",
                self._current_etag[:12] if self._current_etag else "none",
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
                        logger.warning("3 consecutive heartbeat failures — triggering election")
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

        winner = elect_leader(
            self.config.cluster.leader_candidates,
            ping_fn,
            self.cluster_state.my_hostname,
        )

        if winner == self.cluster_state.my_hostname:
            logger.info("Won election — promoting to server")
            self.cluster_state.term.counter += 1
            self.cluster_state.term.leader_id = self.cluster_state.my_hostname
            self.cluster_state.role = "server"
            self.cluster_state.save(self.state_path)
            # Server startup handled by caller/systemd
        elif winner:
            logger.info("Election winner: %s — updating server URL", winner)
            self.server_url = f"http://{winner}:{self.config.server.port}"
            self.cluster_state.server_url = self.server_url
            self.cluster_state.term.counter += 1
            self.cluster_state.term.leader_id = winner
            self.cluster_state.save(self.state_path)
            # Re-submit all files to new leader
            self._submitted.clear()
            self._save_submitted_state()
        else:
            logger.error("Election failed — no reachable candidates")

    # --- State persistence ---
    def _load_submitted_state(self):
        """Load submitted session hashes from disk."""
        state_file = self.config.mesh_dir / "client" / "submitted_sessions.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self._submitted = set(data.get("sha256_hashes", []))
            except Exception:
                logger.warning("Failed to load submitted state", exc_info=True)
                self._submitted = set()

    def _save_submitted_state(self):
        """Save submitted session hashes to disk."""
        state_file = self.config.mesh_dir / "client" / "submitted_sessions.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"sha256_hashes": sorted(self._submitted)}))
