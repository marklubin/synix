"""Blind PR review — posts an LLM review using only docs + diff as context."""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_DIFF_LINES = 10_000
WEBSITE_URL = "https://synix.dev"
LLM_TIMEOUT = 120  # seconds

CLAUDE_MODEL = "claude-opus-4-6"
OPENAI_MODEL = "o3"

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def fetch_diff(repo: str, pr_number: str, token: str) -> str:
    """Fetch PR diff via GitHub API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_website(url: str) -> str:
    """Fetch website content as plain text. Best-effort."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "synix-blind-review"})
        resp.raise_for_status()
        text = re.sub(r"<script[^>]*>.*?</script>", "", resp.text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]
    except Exception:
        return "(website content unavailable)"


def truncate_diff(diff: str) -> tuple[str, int, int]:
    """Truncate diff to MAX_DIFF_LINES. Returns (diff, shown_lines, total_lines)."""
    lines = diff.splitlines()
    total = len(lines)
    if total <= MAX_DIFF_LINES:
        return diff, total, total
    truncated = "\n".join(lines[:MAX_DIFF_LINES])
    truncated += f"\n\n... [diff truncated: showing first {MAX_DIFF_LINES:,} of {total:,} lines]"
    return truncated, MAX_DIFF_LINES, total


def load_prompt(provider: str) -> str:
    """Load the review prompt for the given provider."""
    prompt_file = PROMPTS_DIR / f"{provider}_review.md"
    return prompt_file.read_text()


def read_file(name: str) -> str:
    """Read a file from the workspace."""
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", "."))
    path = workspace / name
    if path.exists():
        return path.read_text()
    return f"({name} not found)"


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def build_user_message(readme: str, design_doc: str, website: str, diff: str) -> str:
    """Compose the user message with all four context sections."""
    return f"""## README.md

{readme}

---

## DESIGN.md

{design_doc}

---

## Website content (synix.dev)

{website}

---

## PR Diff

```diff
{diff}
```"""


def review_claude(system_prompt: str, user_message: str) -> tuple[str, str]:
    """Call Claude API with extended thinking. Returns (review_text, model_name)."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        temperature=1,  # required for extended thinking
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    # Extract the text block (skip thinking blocks)
    for block in response.content:
        if block.type == "text":
            return block.text, CLAUDE_MODEL
    return "(no text in response)", CLAUDE_MODEL


def review_openai(system_prompt: str, user_message: str) -> tuple[str, str]:
    """Call OpenAI API with o3. Returns (review_text, model_name)."""
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content, OPENAI_MODEL


# ---------------------------------------------------------------------------
# GitHub interaction
# ---------------------------------------------------------------------------


def post_comment(repo: str, pr_number: str, token: str, body: str) -> None:
    """Post a comment on the PR via GitHub API."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    resp = requests.post(url, headers=headers, json={"body": body}, timeout=15)
    resp.raise_for_status()


def format_comment(
    review: str,
    provider: str,
    model: str,
    diff_shown: int,
    diff_total: int,
    prompt_file: str,
) -> str:
    """Format the review as a PR comment."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if provider == "claude":
        header = f"**Architectural Review** — Claude Opus | Blind review (docs + diff only)"
    else:
        header = f"**Red Team Review** — OpenAI o3 | Adversarial review (docs + diff only)"

    diff_note = f"{diff_shown:,} lines"
    if diff_shown < diff_total:
        diff_note += f" (of {diff_total:,} total — truncated)"

    return f"""> [!NOTE]
> {header}

{review}

<details>
<summary>Review parameters</summary>

- **Model**: `{model}`
- **Context**: README.md, DESIGN.md, synix.dev, PR diff
- **Diff size**: {diff_note}
- **Prompt**: `{prompt_file}`
- **Timestamp**: {timestamp}

</details>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    provider = os.environ["PROVIDER"]  # "claude" or "openai"
    token = os.environ["GITHUB_TOKEN"]
    pr_number = os.environ["PR_NUMBER"]
    repo = os.environ["REPO"]

    # 1. Gather context
    print(f"Fetching diff for {repo}#{pr_number}...")
    raw_diff = fetch_diff(repo, pr_number, token)
    diff, diff_shown, diff_total = truncate_diff(raw_diff)
    print(f"Diff: {diff_total:,} lines ({diff_shown:,} shown)")

    readme = read_file("README.md")
    design_doc = read_file("docs/DESIGN.md")

    print("Fetching website content...")
    website = fetch_website(WEBSITE_URL)

    # 2. Build prompt and call LLM
    system_prompt = load_prompt(provider)
    user_message = build_user_message(readme, design_doc, website, diff)
    prompt_file = f".github/prompts/{provider}_review.md"

    print(f"Calling {provider} API ({CLAUDE_MODEL if provider == 'claude' else OPENAI_MODEL})...")
    try:
        if provider == "claude":
            review, model = review_claude(system_prompt, user_message)
        elif provider == "openai":
            review, model = review_openai(system_prompt, user_message)
        else:
            print(f"Unknown provider: {provider}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        error_type = type(exc).__name__
        fail_body = (
            f"> [!WARNING]\n"
            f"> **AI Review failed** — {provider} | `{error_type}`\n\n"
            f"The {provider} blind review could not be completed. "
            f"This does not affect the PR.\n"
        )
        print(f"LLM call failed ({error_type}): {exc}", file=sys.stderr)
        post_comment(repo, pr_number, token, fail_body)
        sys.exit(0)  # don't red-X the PR

    # 3. Post review comment
    print(f"Posting review ({len(review)} chars)...")
    body = format_comment(review, provider, model, diff_shown, diff_total, prompt_file)
    post_comment(repo, pr_number, token, body)
    print("Done.")


if __name__ == "__main__":
    main()
