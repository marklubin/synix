"""Tests for the versioned prompt store."""

from __future__ import annotations

from pathlib import Path

import pytest

from synix.server.prompt_store import PromptStore


@pytest.fixture
def store(tmp_path: Path) -> PromptStore:
    return PromptStore(tmp_path / "prompts.db")


@pytest.fixture
def seeded_store(store: PromptStore) -> PromptStore:
    """Store with a few prompts pre-loaded."""
    store.put("greeting", "Hello {name}, welcome!")
    store.put("farewell", "Goodbye {name}.")
    store.put("summary", "Summarize the following:\n{text}")
    return store


# --- Basic CRUD ---


def test_put_and_get(store: PromptStore) -> None:
    store.put("test", "Hello world")
    assert store.get("test") == "Hello world"


def test_get_nonexistent_returns_none(store: PromptStore) -> None:
    assert store.get("nonexistent") is None


def test_get_with_meta(store: PromptStore) -> None:
    result = store.put("test", "content here")
    meta = store.get_with_meta("test")
    assert meta is not None
    assert meta["key"] == "test"
    assert meta["version"] == 1
    assert meta["content"] == "content here"
    assert meta["content_hash"] == result["content_hash"]
    assert meta["created_at"] is not None


def test_get_with_meta_nonexistent(store: PromptStore) -> None:
    assert store.get_with_meta("nope") is None


# --- Versioning ---


def test_put_creates_versions(store: PromptStore) -> None:
    store.put("prompt", "version 1")
    store.put("prompt", "version 2")
    store.put("prompt", "version 3")
    assert store.get("prompt") == "version 3"


def test_get_specific_version(store: PromptStore) -> None:
    store.put("prompt", "version 1")
    store.put("prompt", "version 2")
    assert store.get("prompt", version=1) == "version 1"
    assert store.get("prompt", version=2) == "version 2"


def test_get_with_meta_specific_version(store: PromptStore) -> None:
    store.put("prompt", "v1 content")
    store.put("prompt", "v2 content")
    meta = store.get_with_meta("prompt", version=1)
    assert meta is not None
    assert meta["version"] == 1
    assert meta["content"] == "v1 content"


def test_get_nonexistent_version_returns_none(store: PromptStore) -> None:
    store.put("prompt", "v1")
    assert store.get("prompt", version=99) is None


# --- Dedup ---


def test_put_dedup_same_content(store: PromptStore) -> None:
    """Putting identical content doesn't create a new version."""
    r1 = store.put("prompt", "same content")
    r2 = store.put("prompt", "same content")
    assert r1["version"] == r2["version"]
    assert r1["content_hash"] == r2["content_hash"]
    assert len(store.history("prompt")) == 1


def test_put_different_content_creates_version(store: PromptStore) -> None:
    store.put("prompt", "content A")
    store.put("prompt", "content B")
    assert len(store.history("prompt")) == 2


# --- Listing ---


def test_list_keys_empty(store: PromptStore) -> None:
    assert store.list_keys() == []


def test_list_keys(seeded_store: PromptStore) -> None:
    keys = seeded_store.list_keys()
    assert keys == ["farewell", "greeting", "summary"]  # alphabetical


def test_list_keys_no_duplicates(store: PromptStore) -> None:
    """Multiple versions of same key show up once."""
    store.put("prompt", "v1")
    store.put("prompt", "v2")
    assert store.list_keys() == ["prompt"]


# --- History ---


def test_history_empty_key(store: PromptStore) -> None:
    assert store.history("nonexistent") == []


def test_history_order(store: PromptStore) -> None:
    store.put("prompt", "first")
    store.put("prompt", "second")
    store.put("prompt", "third")
    hist = store.history("prompt")
    assert len(hist) == 3
    # Newest first
    assert hist[0]["version"] == 3
    assert hist[1]["version"] == 2
    assert hist[2]["version"] == 1


def test_history_contains_hash(store: PromptStore) -> None:
    result = store.put("prompt", "test content")
    hist = store.history("prompt")
    assert hist[0]["content_hash"] == result["content_hash"]


# --- Content hash ---


def test_content_hash(store: PromptStore) -> None:
    result = store.put("prompt", "test")
    assert store.content_hash("prompt") == result["content_hash"]


def test_content_hash_nonexistent(store: PromptStore) -> None:
    assert store.content_hash("nope") is None


# --- Seeding from files ---


def test_seed_from_files(store: PromptStore, tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "greeting.txt").write_text("Hello {name}!")
    (prompts_dir / "summary.txt").write_text("Summarize:\n{text}")
    (prompts_dir / "not_a_prompt.md").write_text("ignored")

    count = store.seed_from_files(prompts_dir)
    assert count == 2
    assert store.get("greeting") == "Hello {name}!"
    assert store.get("summary") == "Summarize:\n{text}"


def test_seed_skips_existing(store: PromptStore, tmp_path: Path) -> None:
    """Existing keys are not overwritten by seeding."""
    store.put("greeting", "Custom greeting")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "greeting.txt").write_text("Default greeting")
    (prompts_dir / "other.txt").write_text("Other prompt")

    count = store.seed_from_files(prompts_dir)
    assert count == 1  # only "other" imported
    assert store.get("greeting") == "Custom greeting"  # unchanged
    assert store.get("other") == "Other prompt"


def test_seed_nonexistent_dir(store: PromptStore, tmp_path: Path) -> None:
    count = store.seed_from_files(tmp_path / "does_not_exist")
    assert count == 0


def test_seed_empty_dir(store: PromptStore, tmp_path: Path) -> None:
    prompts_dir = tmp_path / "empty_prompts"
    prompts_dir.mkdir()
    count = store.seed_from_files(prompts_dir)
    assert count == 0


# --- Return value of put ---


def test_put_returns_metadata(store: PromptStore) -> None:
    result = store.put("key", "content")
    assert result["key"] == "key"
    assert result["version"] == 1
    assert "content_hash" in result
    assert "created_at" in result


def test_put_version_increments(store: PromptStore) -> None:
    r1 = store.put("key", "v1")
    r2 = store.put("key", "v2")
    assert r1["version"] == 1
    assert r2["version"] == 2


# --- Multiple keys ---


def test_independent_keys(store: PromptStore) -> None:
    """Versions are tracked per-key."""
    store.put("alpha", "a1")
    store.put("alpha", "a2")
    store.put("beta", "b1")

    assert store.get("alpha") == "a2"
    assert store.get("beta") == "b1"
    assert store.get_with_meta("alpha")["version"] == 2
    assert store.get_with_meta("beta")["version"] == 1


# --- Close ---


def test_close(store: PromptStore) -> None:
    store.put("test", "value")
    store.close()
    # Re-open and verify data persisted
    store2 = PromptStore(store._db_path)
    assert store2.get("test") == "value"
    store2.close()
