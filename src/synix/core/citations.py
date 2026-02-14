"""Citation URI scheme — parse, render, and extract synix:// citation URIs.

The ``synix://`` scheme makes citations unambiguous and machine-parseable
in any text format.  The default output format is markdown links:
``[display text](synix://label)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SYNIX_SCHEME = "synix"

# Matches synix://label where label is word chars + hyphens
_URI_RE = re.compile(r"synix://([\w-]+)")


@dataclass
class Citation:
    """A parsed citation reference."""

    uri: str  # "synix://intel-acme-analytics"
    scheme: str  # "synix"
    ref: str  # "intel-acme-analytics" (the label)


def parse_uri(uri: str) -> Citation:
    """Parse a ``synix://label`` URI into a Citation."""
    m = _URI_RE.fullmatch(uri)
    if not m:
        raise ValueError(f"Invalid synix URI: {uri!r}")
    return Citation(uri=uri, scheme=SYNIX_SCHEME, ref=m.group(1))


def make_uri(label: str) -> str:
    """Build a ``synix://label`` URI from an artifact label."""
    return f"{SYNIX_SCHEME}://{label}"


def render_markdown(uri: str, display_text: str | None = None) -> str:
    """Render a citation as a markdown link ``[text](synix://label)``."""
    ref = parse_uri(uri).ref
    return f"[{display_text or ref}]({uri})"


def extract_citations(content: str) -> list[Citation]:
    """Extract all ``synix://`` URIs from *content* regardless of format."""
    return [Citation(uri=f"synix://{m}", scheme=SYNIX_SCHEME, ref=m) for m in _URI_RE.findall(content)]
