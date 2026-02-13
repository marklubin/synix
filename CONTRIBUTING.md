# Contributing to Synix

Thanks for your interest in contributing.

## Development Setup

1. Clone the repo.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Run tests:
   ```bash
   uv run pytest
   ```

## Development Workflow

1. Create a feature branch.
2. Make focused changes.
3. Add or update tests.
4. Run checks before opening a PR:
   ```bash
   uv run release
   ```

## Pull Request Guidelines

- Keep PRs small and scoped.
- Include a clear description of what changed and why.
- Include test updates for behavior changes.
- Link related issues when applicable.

## Reporting Bugs

Please open an issue with:

- expected behavior
- actual behavior
- reproduction steps
- environment details

## Questions

If you are unsure about an approach, open an issue first to discuss.
