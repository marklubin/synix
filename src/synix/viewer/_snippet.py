"""Search snippet extraction helper."""

from __future__ import annotations

import re


def make_snippet(content: str, query: str, max_chars: int = 200) -> str:
    """Extract a context snippet around the first matching query term.

    - Finds the first occurrence of any query term (case-insensitive).
    - Returns ~max_chars of surrounding context with matched terms
      wrapped in ``<mark>`` tags.
    - If no match is found, returns the beginning of *content* truncated
      to *max_chars*.
    - Multi-word queries are split on whitespace; any term can match.
    """
    if not content:
        return ""

    terms = query.split()
    if not terms:
        return content[:max_chars]

    # Build a pattern that matches any term (longest first to avoid partial
    # matches when one term is a prefix of another).
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_terms]
    pattern = re.compile("|".join(escaped), re.IGNORECASE)

    match = pattern.search(content)
    if match is None:
        return content[:max_chars]

    # Centre the window around the first match.
    mid = match.start()
    half = max_chars // 2
    start = max(0, mid - half)
    end = min(len(content), start + max_chars)
    # Re-adjust start if we hit the end of content early.
    start = max(0, end - max_chars)

    window = content[start:end]

    # Wrap every occurrence inside the window.
    highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", window)
    return highlighted
