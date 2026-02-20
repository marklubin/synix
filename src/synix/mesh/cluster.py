"""Leader election and cluster state management.

Pure logic module — no server/client imports. Network operations
(ping, HTTP) are injected as callables for testability.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def cluster_config_hash(candidates: list[str]) -> str:
    """SHA-256 of ordered candidate list, joined by null bytes."""
    return hashlib.sha256("\0".join(candidates).encode()).hexdigest()


@dataclass
class Term:
    """Election term: monotonically increasing counter + leader identity."""

    counter: int = 0
    leader_id: str = ""

    def to_dict(self) -> dict:
        return {"counter": self.counter, "leader_id": self.leader_id}

    @classmethod
    def from_dict(cls, d: dict) -> Term:
        return cls(counter=d.get("counter", 0), leader_id=d.get("leader_id", ""))


@dataclass
class ClusterState:
    """Full cluster state persisted to state.json."""

    term: Term = field(default_factory=Term)
    candidates: list[str] = field(default_factory=list)
    config_hash: str = ""
    role: str = "client"  # "server" or "client"
    server_url: str = ""
    my_hostname: str = ""

    def save(self, path: Path) -> None:
        """Persist state to JSON file."""
        data = {
            "term": self.term.to_dict(),
            "candidates": self.candidates,
            "config_hash": self.config_hash,
            "role": self.role,
            "server_url": self.server_url,
            "my_hostname": self.my_hostname,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ClusterState:
        """Load state from JSON file. Returns default if file doesn't exist."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            term=Term.from_dict(data.get("term", {})),
            candidates=data.get("candidates", []),
            config_hash=data.get("config_hash", ""),
            role=data.get("role", "client"),
            server_url=data.get("server_url", ""),
            my_hostname=data.get("my_hostname", ""),
        )


def validate_term(request_term: Term, server_term: Term) -> tuple[bool, str]:
    """Validate a request's term against the server's current term.

    Returns (valid, reason). Rejects if:
    - request counter < server counter (stale term)
    - request counter == server counter AND request leader_id != server leader_id
    """
    if request_term.counter < server_term.counter:
        return False, f"stale term: request={request_term.counter} < server={server_term.counter}"
    if request_term.counter == server_term.counter and request_term.leader_id != server_term.leader_id:
        return False, f"term conflict: same counter={request_term.counter} but leader_id mismatch"
    return True, ""


def elect_leader(
    candidates: list[str],
    ping_fn: Callable[[str], bool],
    my_hostname: str,
) -> str | None:
    """Run leader election. Iterate candidates in priority order, first alive wins.

    Args:
        candidates: Ordered list of hostnames (index 0 = highest priority)
        ping_fn: Callable that returns True if host is reachable
        my_hostname: This node's hostname

    Returns the winning hostname, or None if no candidate is reachable.
    """
    for candidate in candidates:
        if candidate == my_hostname:
            return candidate  # I'm alive, I win at this priority
        if ping_fn(candidate):
            return candidate  # Higher-priority candidate is alive
    return None


def resolve_equal_term_tiebreak(
    term_counter: int,
    leader_a: str,
    leader_b: str,
    candidates: list[str],
) -> str:
    """Deterministic tie-breaking: lower candidate index wins.

    When two candidates both claim the same term counter,
    the one with the lower index in candidates is the valid leader.
    Raises ValueError if neither leader is in candidates.
    """
    idx_a = candidates.index(leader_a) if leader_a in candidates else len(candidates)
    idx_b = candidates.index(leader_b) if leader_b in candidates else len(candidates)
    if idx_a == len(candidates) and idx_b == len(candidates):
        raise ValueError(f"Neither {leader_a} nor {leader_b} found in candidates")
    return leader_a if idx_a <= idx_b else leader_b


def should_step_down(
    my_hostname: str,
    my_term: Term,
    request_term: Term,
    candidates: list[str],
) -> bool:
    """Check if the current server should step down.

    Step down if:
    - Request has higher term counter
    - Request has same term counter but a higher-priority leader_id
    """
    if request_term.counter > my_term.counter:
        return True
    if request_term.counter == my_term.counter:
        if request_term.leader_id != my_hostname:
            # Check if request's leader has higher priority
            my_idx = candidates.index(my_hostname) if my_hostname in candidates else len(candidates)
            req_idx = (
                candidates.index(request_term.leader_id) if request_term.leader_id in candidates else len(candidates)
            )
            if req_idx < my_idx:
                return True
    return False


def leader_self_check(
    my_hostname: str,
    my_term: Term,
    candidates: list[str],
    ping_fn: Callable[[str], bool],
) -> bool:
    """Periodic self-check for the leader.

    Probes higher-priority candidates. If a higher-priority node is alive,
    fails closed (returns False = should step down).

    Returns True if this node should remain leader, False if it should step down.
    """
    my_idx = candidates.index(my_hostname) if my_hostname in candidates else len(candidates)

    for i, candidate in enumerate(candidates):
        if i >= my_idx:
            break  # Only check higher-priority candidates
        if ping_fn(candidate):
            logger.warning(
                "Leader self-check: higher-priority candidate %s (index %d) is alive, stepping down (my index: %d)",
                candidate,
                i,
                my_idx,
            )
            return False  # Higher-priority node is alive, step down

    return True  # No higher-priority node is alive, remain leader
