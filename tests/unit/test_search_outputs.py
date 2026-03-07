"""Unit tests for local search output discovery and resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synix.build.search_outputs import (
    SearchOutputResolutionError,
    SearchOutputSpec,
    list_search_outputs,
    resolve_search_output,
)


def _write_projection_cache(build_dir: Path, payload: dict) -> None:
    (build_dir / ".projection_cache.json").write_text(json.dumps(payload))


def test_resolve_search_output_returns_none_when_build_has_no_outputs(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    assert list_search_outputs(build_dir) == []
    assert resolve_search_output(build_dir) is None


def test_list_search_outputs_falls_back_to_legacy_search_db(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    legacy_db = build_dir / "search.db"
    legacy_db.write_bytes(b"")

    assert list_search_outputs(build_dir) == [
        SearchOutputSpec(
            name="search",
            projection_type="search_index",
            db_path=legacy_db.resolve(strict=False),
        )
    ]


def test_list_search_outputs_reads_projection_metadata_and_skips_invalid_entries(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    outputs_dir = build_dir / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "memory.db").write_bytes(b"")
    (outputs_dir / "archive.db").write_bytes(b"")

    _write_projection_cache(
        build_dir,
        {
            "archive": {
                "projection_type": "search_index",
                "db_path": "outputs/archive.db",
            },
            "broken": {
                "projection_type": "synix_search",
            },
            "search": {
                "projection_type": "synix_search",
                "db_path": "outputs/memory.db",
            },
            "ignored": {
                "projection_type": "flat_file",
                "db_path": "context.md",
            },
        },
    )

    assert list_search_outputs(build_dir) == [
        SearchOutputSpec(
            name="archive",
            projection_type="search_index",
            db_path=(outputs_dir / "archive.db").resolve(strict=False),
        ),
        SearchOutputSpec(
            name="search",
            projection_type="synix_search",
            db_path=(outputs_dir / "memory.db").resolve(strict=False),
        ),
    ]


def test_list_search_outputs_rejects_paths_outside_build_dir(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    _write_projection_cache(
        build_dir,
        {
            "search": {
                "projection_type": "synix_search",
                "db_path": "../escape.db",
            }
        },
    )

    with pytest.raises(SearchOutputResolutionError, match="escapes the build directory"):
        list_search_outputs(build_dir)


def test_resolve_search_output_returns_the_only_metadata_output(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    outputs_dir = build_dir / "outputs"
    outputs_dir.mkdir()
    db_path = outputs_dir / "memory.db"
    db_path.write_bytes(b"")
    _write_projection_cache(
        build_dir,
        {
            "memory": {
                "projection_type": "synix_search",
                "db_path": "outputs/memory.db",
            }
        },
    )

    resolved = resolve_search_output(build_dir)
    assert resolved == SearchOutputSpec(
        name="memory",
        projection_type="synix_search",
        db_path=db_path.resolve(strict=False),
    )


def test_resolve_search_output_honors_explicit_projection_name(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    outputs_dir = build_dir / "outputs"
    outputs_dir.mkdir()
    archive_db = outputs_dir / "archive.db"
    current_db = outputs_dir / "current.db"
    archive_db.write_bytes(b"")
    current_db.write_bytes(b"")
    _write_projection_cache(
        build_dir,
        {
            "archive": {
                "projection_type": "search_index",
                "db_path": "outputs/archive.db",
            },
            "current": {
                "projection_type": "synix_search",
                "db_path": "outputs/current.db",
            },
        },
    )

    resolved = resolve_search_output(build_dir, projection_name="archive")
    assert resolved == SearchOutputSpec(
        name="archive",
        projection_type="search_index",
        db_path=archive_db.resolve(strict=False),
    )


def test_resolve_search_output_prefers_named_search_synix_output(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    outputs_dir = build_dir / "outputs"
    outputs_dir.mkdir()
    search_db = outputs_dir / "search.db"
    archive_db = outputs_dir / "archive.db"
    search_db.write_bytes(b"")
    archive_db.write_bytes(b"")
    _write_projection_cache(
        build_dir,
        {
            "archive": {
                "projection_type": "search_index",
                "db_path": "outputs/archive.db",
            },
            "search": {
                "projection_type": "synix_search",
                "db_path": "outputs/search.db",
            },
        },
    )

    resolved = resolve_search_output(build_dir)
    assert resolved == SearchOutputSpec(
        name="search",
        projection_type="synix_search",
        db_path=search_db.resolve(strict=False),
    )


def test_resolve_search_output_prefers_the_only_synix_output_when_multiple_exist(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    outputs_dir = build_dir / "outputs"
    outputs_dir.mkdir()
    current_db = outputs_dir / "current.db"
    legacy_db = outputs_dir / "legacy.db"
    current_db.write_bytes(b"")
    legacy_db.write_bytes(b"")
    _write_projection_cache(
        build_dir,
        {
            "current": {
                "projection_type": "synix_search",
                "db_path": "outputs/current.db",
            },
            "legacy": {
                "projection_type": "search_index",
                "db_path": "outputs/legacy.db",
            },
        },
    )

    resolved = resolve_search_output(build_dir)
    assert resolved == SearchOutputSpec(
        name="current",
        projection_type="synix_search",
        db_path=current_db.resolve(strict=False),
    )


def test_resolve_search_output_requires_projection_when_outputs_are_ambiguous(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    outputs_dir = build_dir / "outputs"
    outputs_dir.mkdir()
    for filename in ("alpha.db", "beta.db"):
        (outputs_dir / filename).write_bytes(b"")
    _write_projection_cache(
        build_dir,
        {
            "alpha": {
                "projection_type": "search_index",
                "db_path": "outputs/alpha.db",
            },
            "beta": {
                "projection_type": "search_index",
                "db_path": "outputs/beta.db",
            },
        },
    )

    with pytest.raises(SearchOutputResolutionError, match="Re-run with --projection <name>"):
        resolve_search_output(build_dir)


def test_resolve_search_output_reports_available_projection_names(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    outputs_dir = build_dir / "outputs"
    outputs_dir.mkdir()
    for filename in ("archive.db", "search.db"):
        (outputs_dir / filename).write_bytes(b"")
    _write_projection_cache(
        build_dir,
        {
            "archive": {
                "projection_type": "search_index",
                "db_path": "outputs/archive.db",
            },
            "search": {
                "projection_type": "synix_search",
                "db_path": "outputs/search.db",
            },
        },
    )

    with pytest.raises(SearchOutputResolutionError, match="Available: archive, search"):
        resolve_search_output(build_dir, projection_name="missing")
