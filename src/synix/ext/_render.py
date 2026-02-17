"""Safe prompt template rendering for ext transforms.

Escapes placeholder tokens in substitution values so user content
(artifact text, labels, etc.) cannot accidentally trigger further
substitutions.
"""

from __future__ import annotations

# All placeholder tokens used across ext transforms
_PLACEHOLDERS = (
    "{artifact}",
    "{artifacts}",
    "{label}",
    "{artifact_type}",
    "{group_key}",
    "{count}",
    "{accumulated}",
    "{step}",
    "{total}",
)

# Escape sequences — Unicode zero-width joiner inside braces
# so they won't match any placeholder pattern
_ESCAPE_MAP = {p: p[0] + "\u200b" + p[1:] for p in _PLACEHOLDERS}
_UNESCAPE_MAP = {v: k for k, v in _ESCAPE_MAP.items()}


def _escape_value(value: str) -> str:
    """Escape placeholder tokens in a substitution value."""
    result = value
    for token, escaped in _ESCAPE_MAP.items():
        result = result.replace(token, escaped)
    return result


def render_template(template: str, **kwargs: str) -> str:
    """Render a prompt template with safe placeholder substitution.

    Values are escaped before substitution so that user content containing
    placeholder tokens (e.g., ``{artifact}`` in artifact text) won't be
    accidentally replaced by subsequent substitutions.

    After all substitutions, escape sequences are removed so the final
    prompt contains clean text.
    """
    result = template
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, _escape_value(value))

    # Remove escape sequences from final output
    for escaped, original in _UNESCAPE_MAP.items():
        result = result.replace(escaped, original)

    return result
