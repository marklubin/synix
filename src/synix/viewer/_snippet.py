"""Search snippet extraction helper."""

from __future__ import annotations

import html
import re


def make_snippet(content: str, query: str, max_chars: int = 200) -> str:
    """Extract a context snippet around the first matching query term.

    - Finds the first occurrence of any query term (case-insensitive).
    - Returns ~max_chars of surrounding context with matched terms
      wrapped in ``<mark>`` tags.
    - Content is HTML-escaped to prevent XSS when rendered in the browser.
    - If no match is found, returns the beginning of *content* truncated
      to *max_chars* (HTML-escaped).
    - Multi-word queries are split on whitespace; any term can match.
    """
    if not content:
        return ""

    terms = query.split()
    if not terms:
        return html.escape(content[:max_chars])

    # Build a pattern that matches any term (longest first to avoid partial
    # matches when one term is a prefix of another).
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_terms]
    pattern = re.compile("|".join(escaped), re.IGNORECASE)

    match = pattern.search(content)
    if match is None:
        return html.escape(content[:max_chars])

    # Centre the window around the first match.
    mid = match.start()
    half = max_chars // 2
    start = max(0, mid - half)
    end = min(len(content), start + max_chars)
    # Re-adjust start if we hit the end of content early.
    start = max(0, end - max_chars)

    window = content[start:end]

    # HTML-escape the window first to prevent XSS, then highlight matches.
    safe_window = html.escape(window)

    # Build a new pattern that matches the HTML-escaped versions of the terms
    # so highlighting works correctly on the escaped text.
    safe_escaped = [re.escape(html.escape(t)) for t in sorted_terms]
    safe_pattern = re.compile("|".join(safe_escaped), re.IGNORECASE)

    highlighted = safe_pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe_window)
    return highlighted
