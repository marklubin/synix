"""Health monitoring for mesh cluster members."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemberState:
    """State of a single cluster member."""

    hostname: str
    last_heartbeat: float = 0.0
    missed_beats: int = 0
    status: str = "unknown"  # "healthy", "suspect", "dead", "unknown"
    term_counter: int = 0
    config_hash: str = ""


class HealthMonitor:
    """Tracks heartbeats and detects member failures.

    A member is:
    - "healthy": received heartbeat within the interval
    - "suspect": missed 1-2 heartbeats
    - "dead": missed 3+ heartbeats
    """

    def __init__(self, heartbeat_interval: int = 120, miss_threshold: int = 3):
        self.heartbeat_interval = heartbeat_interval
        self.miss_threshold = miss_threshold
        self.members: dict[str, MemberState] = {}

    def record_heartbeat(self, hostname: str, term_counter: int = 0, config_hash: str = "") -> None:
        """Record a heartbeat from a member."""
        if hostname not in self.members:
            self.members[hostname] = MemberState(hostname=hostname)
        member = self.members[hostname]
        member.last_heartbeat = time.time()
        member.missed_beats = 0
        member.status = "healthy"
        member.term_counter = term_counter
        member.config_hash = config_hash

    def check_health(self) -> list[MemberState]:
        """Check all members and update their status. Returns list of dead members."""
        now = time.time()
        dead = []
        for member in self.members.values():
            if member.last_heartbeat == 0:
                continue
            elapsed = now - member.last_heartbeat
            expected_beats = int(elapsed / self.heartbeat_interval)
            member.missed_beats = max(0, expected_beats)
            if member.missed_beats >= self.miss_threshold:
                member.status = "dead"
                dead.append(member)
            elif member.missed_beats >= 1:
                member.status = "suspect"
            else:
                member.status = "healthy"
        return dead

    def get_member_states(self) -> list[dict]:
        """Return all member states as dicts."""
        return [
            {
                "hostname": m.hostname,
                "status": m.status,
                "last_heartbeat": m.last_heartbeat,
                "missed_beats": m.missed_beats,
                "term_counter": m.term_counter,
                "config_hash": m.config_hash,
            }
            for m in self.members.values()
        ]

    def remove_member(self, hostname: str) -> None:
        """Remove a member from tracking."""
        self.members.pop(hostname, None)
