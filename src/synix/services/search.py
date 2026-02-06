"""FTS search operations (data plane)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class SearchHit:
    """A search result with relevance info."""

    record_id: str
    content: str
    step_name: str
    rank: float
    snippet: str | None = None
    metadata: dict[str, Any] | None = None


def search_fts(
    session: Session,
    query: str,
    step: str | None = None,
    branch: str = "main",
    limit: int = 10,
) -> list[SearchHit]:
    """Search records using FTS5 full-text search.

    Args:
        session: Database session.
        query: Search query (FTS5 syntax supported).
        step: Optional step name to filter by.
        branch: Branch name (default: main).
        limit: Maximum results to return.

    Returns:
        List of SearchHit objects sorted by relevance.
    """
    # Build the FTS query
    # Using MATCH for FTS5 and bm25() for ranking
    # Join through record_fts_map to get record IDs
    if step:
        sql = """
            SELECT
                r.id,
                r.content,
                r.step_name,
                bm25(record_fts) as rank,
                snippet(record_fts, 0, '<mark>', '</mark>', '...', 32) as snippet,
                r.metadata_json
            FROM record_fts
            JOIN record_fts_map m ON record_fts.rowid = m.rowid
            JOIN records r ON m.record_id = r.id
            WHERE record_fts MATCH :query
              AND r.step_name = :step
              AND r.branch = :branch
            ORDER BY rank
            LIMIT :limit
        """
        params = {"query": query, "step": step, "branch": branch, "limit": limit}
    else:
        sql = """
            SELECT
                r.id,
                r.content,
                r.step_name,
                bm25(record_fts) as rank,
                snippet(record_fts, 0, '<mark>', '</mark>', '...', 32) as snippet,
                r.metadata_json
            FROM record_fts
            JOIN record_fts_map m ON record_fts.rowid = m.rowid
            JOIN records r ON m.record_id = r.id
            WHERE record_fts MATCH :query
              AND r.branch = :branch
            ORDER BY rank
            LIMIT :limit
        """
        params = {"query": query, "branch": branch, "limit": limit}

    result = session.execute(text(sql), params)
    rows = result.fetchall()

    hits = []
    for row in rows:
        import json

        metadata = None
        if row[5]:
            try:
                metadata = json.loads(row[5])
            except (json.JSONDecodeError, TypeError):
                pass

        hits.append(
            SearchHit(
                record_id=str(row[0]),
                content=row[1],
                step_name=row[2],
                rank=float(row[3]),
                snippet=row[4],
                metadata=metadata,
            )
        )

    return hits


def search_prefix(
    session: Session,
    prefix: str,
    step: str | None = None,
    branch: str = "main",
    limit: int = 10,
) -> list[SearchHit]:
    """Search records using prefix matching.

    Adds a wildcard to the query for prefix matching.

    Args:
        session: Database session.
        prefix: Prefix to search for.
        step: Optional step name to filter by.
        branch: Branch name (default: main).
        limit: Maximum results to return.

    Returns:
        List of SearchHit objects.
    """
    # Add wildcard for prefix matching
    query = f'"{prefix}"*'
    return search_fts(session, query, step=step, branch=branch, limit=limit)


def count_matches(
    session: Session,
    query: str,
    step: str | None = None,
    branch: str = "main",
) -> int:
    """Count matching records for a query.

    Args:
        session: Database session.
        query: Search query.
        step: Optional step name to filter by.
        branch: Branch name.

    Returns:
        Number of matching records.
    """
    if step:
        sql = """
            SELECT COUNT(*)
            FROM record_fts
            JOIN record_fts_map m ON record_fts.rowid = m.rowid
            JOIN records r ON m.record_id = r.id
            WHERE record_fts MATCH :query
              AND r.step_name = :step
              AND r.branch = :branch
        """
        params = {"query": query, "step": step, "branch": branch}
    else:
        sql = """
            SELECT COUNT(*)
            FROM record_fts
            JOIN record_fts_map m ON record_fts.rowid = m.rowid
            JOIN records r ON m.record_id = r.id
            WHERE record_fts MATCH :query
              AND r.branch = :branch
        """
        params = {"query": query, "branch": branch}

    result = session.execute(text(sql), params)
    return result.scalar() or 0
