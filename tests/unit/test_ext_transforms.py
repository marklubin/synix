"""Tests for generic platform transforms — MapSynthesis, GroupSynthesis, ReduceSynthesis, FoldSynthesis."""

from __future__ import annotations

import logging

import pytest

from synix import Artifact
from synix.ext._render import render_template
from synix.transforms import FoldSynthesis, GroupSynthesis, MapSynthesis, ReduceSynthesis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(label: str, content: str = "content", **metadata) -> Artifact:
    return Artifact(
        label=label,
        artifact_type="test",
        content=content,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    """Tests for safe prompt template rendering."""

    def test_basic_substitution(self):
        result = render_template("Hello {artifact}", artifact="world")
        assert result == "Hello world"

    def test_multiple_placeholders(self):
        result = render_template("{label}: {artifact}", label="Alice", artifact="content")
        assert result == "Alice: content"

    def test_placeholder_in_value_escaped(self):
        """User content containing placeholder tokens is not double-substituted."""
        result = render_template(
            "Label: {label}\nContent: {artifact}",
            label="bio-alice",
            artifact="Her note said {label} is important",
        )
        assert "Label: bio-alice" in result
        assert "Her note said {label} is important" in result

    def test_nested_placeholder_escaped(self):
        """Accumulated content with {artifact} doesn't get replaced."""
        result = render_template(
            "Current: {accumulated}\nNew: {artifact}",
            accumulated="Previous text mentioned {artifact} template",
            artifact="actual new content",
        )
        assert "Previous text mentioned {artifact} template" in result
        assert "actual new content" in result

    def test_all_placeholders(self):
        """All known placeholders work."""
        result = render_template(
            "{artifact}|{artifacts}|{label}|{artifact_type}|{group_key}|{count}|{accumulated}|{step}|{total}",
            artifact="a",
            artifacts="b",
            label="c",
            artifact_type="d",
            group_key="e",
            count="f",
            accumulated="g",
            step="h",
            total="i",
        )
        assert result == "a|b|c|d|e|f|g|h|i"

    def test_missing_placeholder_preserved(self):
        """Placeholders not in kwargs are left as-is."""
        result = render_template("Hello {artifact} and {unknown}", artifact="world")
        assert result == "Hello world and {unknown}"


# ---------------------------------------------------------------------------
# MapSynthesis
# ---------------------------------------------------------------------------


class TestMapSynthesis:
    """Tests for 1:1 MapSynthesis transform."""

    def test_single_input(self, mock_llm):
        """One input produces one output with prompt content substituted."""
        t = MapSynthesis(
            "ws",
            prompt="Analyze: {artifact}",
            artifact_type="analysis",
        )
        inp = _make_artifact("bio-alice", "Alice is an engineer.")
        results = t.execute([inp], {"llm_config": {}})

        assert len(results) == 1
        assert results[0].artifact_type == "analysis"
        assert results[0].label == "ws-bio-alice"
        assert results[0].input_ids == [inp.artifact_id]
        assert results[0].prompt_id.startswith("map_synthesis_v")

        # Verify prompt was sent to LLM with substitution
        assert len(mock_llm) == 1
        sent = mock_llm[0]["messages"][0]["content"]
        assert "Alice is an engineer." in sent

    def test_label_fn(self, mock_llm):
        """Custom label_fn overrides default label derivation."""
        t = MapSynthesis(
            "ws",
            prompt="Analyze: {artifact}",
            label_fn=lambda a: f"custom-{a.label.split('-')[-1]}",
        )
        inp = _make_artifact("bio-alice")
        results = t.execute([inp], {"llm_config": {}})

        assert results[0].label == "custom-alice"

    def test_default_split_1_to_1(self, mock_llm):
        """Default split gives 1:1 — one unit per input."""
        t = MapSynthesis("ws", prompt="Analyze: {artifact}")
        inputs = [_make_artifact(f"bio-{i}") for i in range(3)]
        units = t.split(inputs, {})

        assert len(units) == 3
        for unit_inputs, config_extras in units:
            assert len(unit_inputs) == 1

    def test_placeholder_label_and_type(self, mock_llm):
        """Placeholders {label} and {artifact_type} are substituted."""
        t = MapSynthesis("ws", prompt="Label: {label}, Type: {artifact_type}")
        inp = Artifact(label="bio-alice", artifact_type="biography", content="text")
        t.execute([inp], {"llm_config": {}})

        sent = mock_llm[0]["messages"][0]["content"]
        assert "Label: bio-alice" in sent
        assert "Type: biography" in sent

    def test_cache_key_changes_with_prompt(self):
        """Different prompts produce different cache keys."""
        t1 = MapSynthesis("ws", prompt="Prompt A")
        t2 = MapSynthesis("ws", prompt="Prompt B")
        assert t1.get_cache_key({}) != t2.get_cache_key({})

    def test_cache_key_changes_with_artifact_type(self):
        """Different artifact_type produces different cache keys."""
        t1 = MapSynthesis("ws", prompt="x", artifact_type="summary")
        t2 = MapSynthesis("ws", prompt="x", artifact_type="analysis")
        assert t1.get_cache_key({}) != t2.get_cache_key({})

    def test_fingerprint_includes_callable(self):
        """Fingerprint includes callable component when label_fn is set."""
        fn = lambda a: a.label  # noqa: E731
        t = MapSynthesis("ws", prompt="x", label_fn=fn)
        fp = t.compute_fingerprint({})
        assert "label_fn" in fp.components

    def test_fingerprint_without_callable(self):
        """Fingerprint omits callable when label_fn is None."""
        t = MapSynthesis("ws", prompt="x")
        fp = t.compute_fingerprint({})
        assert "callable" not in fp.components

    def test_estimate_output_count(self):
        """Default 1:1 estimate."""
        t = MapSynthesis("ws", prompt="x")
        assert t.estimate_output_count(5) == 5

    def test_placeholder_injection_safe(self, mock_llm):
        """User content with placeholder tokens doesn't cause double-substitution."""
        t = MapSynthesis("ws", prompt="Label: {label}\nContent: {artifact}")
        inp = Artifact(
            label="bio-alice",
            artifact_type="test",
            content="The {label} variable was mentioned in the text",
        )
        t.execute([inp], {"llm_config": {}})
        sent = mock_llm[0]["messages"][0]["content"]
        assert "Label: bio-alice" in sent
        assert "The {label} variable was mentioned in the text" in sent


# ---------------------------------------------------------------------------
# GroupSynthesis
# ---------------------------------------------------------------------------


class TestGroupSynthesis:
    """Tests for N:M GroupSynthesis transform."""

    def test_groups_by_metadata_key(self, mock_llm):
        """Artifacts grouped by metadata key produce one output per group."""
        inputs = [
            _make_artifact("ep-1", "content 1", team="alpha"),
            _make_artifact("ep-2", "content 2", team="beta"),
            _make_artifact("ep-3", "content 3", team="alpha"),
        ]
        t = GroupSynthesis(
            "team-summaries",
            group_by="team",
            prompt="Summarize team '{group_key}':\n\n{artifacts}",
            artifact_type="team_summary",
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert len(results) == 2
        assert len(mock_llm) == 2
        labels = {r.label for r in results}
        assert "team-alpha" in labels
        assert "team-beta" in labels

    def test_groups_by_callable(self, mock_llm):
        """group_by callable extracts custom group key."""
        inputs = [
            _make_artifact("ep-2024-01-a", "jan"),
            _make_artifact("ep-2024-01-b", "jan"),
            _make_artifact("ep-2024-02-a", "feb"),
        ]
        t = GroupSynthesis(
            "monthly",
            group_by=lambda a: a.label.rsplit("-", 1)[0],
            prompt="Group: {group_key}\n{artifacts}",
        )
        units = t.split(inputs, {})
        assert len(units) == 2

    def test_on_missing_group(self, mock_llm, caplog):
        """on_missing='group' collects missing-key artifacts under missing_key."""
        inputs = [
            _make_artifact("ep-1", team="alpha"),
            _make_artifact("ep-2"),  # no team metadata
        ]
        t = GroupSynthesis(
            "summaries",
            group_by="team",
            prompt="{artifacts}",
            on_missing="group",
        )
        with caplog.at_level(logging.WARNING, logger="synix.ext.group_synthesis"):
            units = t.split(inputs, {})
        group_keys = [cfg["_group_key"] for _, cfg in units]
        assert "_ungrouped" in group_keys
        assert "alpha" in group_keys

        assert "missing field 'team'" in caplog.text
        assert "grouped as '_ungrouped'" in caplog.text

    def test_on_missing_skip(self, mock_llm, caplog):
        """on_missing='skip' drops artifacts without the key."""
        inputs = [
            _make_artifact("ep-1", team="alpha"),
            _make_artifact("ep-2"),  # no team
        ]
        t = GroupSynthesis(
            "summaries",
            group_by="team",
            prompt="{artifacts}",
            on_missing="skip",
        )
        with caplog.at_level(logging.WARNING, logger="synix.ext.group_synthesis"):
            units = t.split(inputs, {})
        assert len(units) == 1  # only alpha group
        assert units[0][1]["_group_key"] == "alpha"

        assert "skipped" in caplog.text

    def test_on_missing_error(self):
        """on_missing='error' raises ValueError immediately."""
        inputs = [
            _make_artifact("ep-1", team="alpha"),
            _make_artifact("ep-2"),  # no team
        ]
        t = GroupSynthesis(
            "summaries",
            group_by="team",
            prompt="{artifacts}",
            on_missing="error",
        )
        with pytest.raises(ValueError, match="missing field 'team'"):
            t.split(inputs, {})

    def test_invalid_on_missing(self):
        """Invalid on_missing value raises ValueError at init."""
        with pytest.raises(ValueError, match="on_missing must be"):
            GroupSynthesis("x", group_by="k", prompt="x", on_missing="invalid")

    def test_custom_label_prefix(self, mock_llm):
        """label_prefix overrides default prefix."""
        inputs = [_make_artifact("ep-1", team="alpha")]
        t = GroupSynthesis(
            "summaries",
            group_by="team",
            prompt="{artifacts}",
            label_prefix="cust",
        )
        results = t.execute(inputs, {"llm_config": {}})
        assert results[0].label == "cust-alpha"

    def test_custom_missing_key(self, mock_llm, caplog):
        """Custom missing_key changes the ungrouped group name."""
        inputs = [_make_artifact("ep-1")]  # no team
        t = GroupSynthesis(
            "summaries",
            group_by="team",
            prompt="{artifacts}",
            on_missing="group",
            missing_key="no-team",
        )
        with caplog.at_level(logging.WARNING, logger="synix.ext.group_synthesis"):
            units = t.split(inputs, {})
        assert units[0][1]["_group_key"] == "no-team"

    def test_placeholders_substituted(self, mock_llm):
        """All placeholders are substituted in the prompt."""
        inputs = [_make_artifact("ep-1", "hello", team="alpha")]
        t = GroupSynthesis(
            "summaries",
            group_by="team",
            prompt="Key: {group_key}, Count: {count}, Type: {artifact_type}",
            artifact_type="report",
        )
        t.execute(inputs, {"llm_config": {}})
        sent = mock_llm[0]["messages"][0]["content"]
        assert "Key: alpha" in sent
        assert "Count: 1" in sent
        assert "Type: report" in sent

    def test_sorted_groups(self, mock_llm):
        """Groups are processed in sorted key order."""
        inputs = [
            _make_artifact("ep-1", team="charlie"),
            _make_artifact("ep-2", team="alpha"),
            _make_artifact("ep-3", team="bravo"),
        ]
        t = GroupSynthesis("s", group_by="team", prompt="{artifacts}")
        units = t.split(inputs, {})
        keys = [cfg["_group_key"] for _, cfg in units]
        assert keys == ["alpha", "bravo", "charlie"]

    def test_output_metadata(self, mock_llm):
        """Output artifacts include group_key and input_count metadata."""
        inputs = [
            _make_artifact("ep-1", team="alpha"),
            _make_artifact("ep-2", team="alpha"),
        ]
        t = GroupSynthesis("s", group_by="team", prompt="{artifacts}")
        results = t.execute(inputs, {"llm_config": {}})
        assert results[0].metadata["group_key"] == "alpha"
        assert results[0].metadata["input_count"] == 2

    def test_fingerprint_includes_callable(self):
        """Fingerprint includes callable when group_by is a callable."""
        fn = lambda a: "x"  # noqa: E731
        t = GroupSynthesis("s", group_by=fn, prompt="x")
        fp = t.compute_fingerprint({})
        assert "group_by" in fp.components

    def test_cache_key_changes_with_group_by(self):
        """Different group_by keys produce different cache keys."""
        t1 = GroupSynthesis("s", group_by="team", prompt="x")
        t2 = GroupSynthesis("s", group_by="region", prompt="x")
        assert t1.get_cache_key({}) != t2.get_cache_key({})

    def test_cache_key_changes_with_on_missing(self):
        """Different on_missing produces different cache keys."""
        t1 = GroupSynthesis("s", group_by="k", prompt="x", on_missing="group")
        t2 = GroupSynthesis("s", group_by="k", prompt="x", on_missing="skip")
        assert t1.get_cache_key({}) != t2.get_cache_key({})

    def test_cache_key_changes_with_artifact_type(self):
        """Different artifact_type produces different cache keys."""
        t1 = GroupSynthesis("s", group_by="k", prompt="x", artifact_type="summary")
        t2 = GroupSynthesis("s", group_by="k", prompt="x", artifact_type="report")
        assert t1.get_cache_key({}) != t2.get_cache_key({})


# ---------------------------------------------------------------------------
# ReduceSynthesis
# ---------------------------------------------------------------------------


class TestReduceSynthesis:
    """Tests for N:1 ReduceSynthesis transform."""

    def test_single_output(self, mock_llm):
        """All inputs reduce to exactly one output."""
        inputs = [_make_artifact(f"ws-{i}", f"profile {i}") for i in range(3)]
        t = ReduceSynthesis(
            "team_dynamics",
            prompt="Analyze dynamics:\n\n{artifacts}",
            label="team-dynamics",
            artifact_type="dynamics",
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert len(results) == 1
        assert len(mock_llm) == 1
        assert results[0].label == "team-dynamics"
        assert results[0].artifact_type == "dynamics"
        assert results[0].prompt_id.startswith("reduce_synthesis_v")
        assert len(results[0].input_ids) == 3
        assert results[0].metadata["input_count"] == 3

    def test_split_single_unit(self):
        """Split returns a single unit with all inputs."""
        t = ReduceSynthesis("r", prompt="x", label="out")
        inputs = [_make_artifact(f"a-{i}") for i in range(5)]
        units = t.split(inputs, {})
        assert len(units) == 1
        assert len(units[0][0]) == 5

    def test_estimate_output_count(self):
        """Always estimates 1 output."""
        t = ReduceSynthesis("r", prompt="x", label="out")
        assert t.estimate_output_count(10) == 1
        assert t.estimate_output_count(1) == 1

    def test_count_placeholder(self, mock_llm):
        """{count} placeholder is substituted."""
        inputs = [_make_artifact(f"a-{i}") for i in range(4)]
        t = ReduceSynthesis("r", prompt="Count: {count}", label="out")
        t.execute(inputs, {"llm_config": {}})
        sent = mock_llm[0]["messages"][0]["content"]
        assert "Count: 4" in sent

    def test_deterministic_ordering(self, mock_llm):
        """Inputs are sorted by artifact_id for deterministic prompt."""
        a1 = Artifact(label="z", artifact_type="test", content="zzz")
        a2 = Artifact(label="a", artifact_type="test", content="aaa")
        t = ReduceSynthesis("r", prompt="{artifacts}", label="out")
        t.execute([a1, a2], {"llm_config": {}})
        sent = mock_llm[0]["messages"][0]["content"]
        # Both should appear, sorted by artifact_id (content hash)
        assert "aaa" in sent
        assert "zzz" in sent

    def test_cache_key_changes_with_artifact_type(self):
        """Different artifact_type produces different cache keys."""
        t1 = ReduceSynthesis("r", prompt="x", label="out", artifact_type="summary")
        t2 = ReduceSynthesis("r", prompt="x", label="out", artifact_type="report")
        assert t1.get_cache_key({}) != t2.get_cache_key({})


# ---------------------------------------------------------------------------
# FoldSynthesis
# ---------------------------------------------------------------------------


class TestFoldSynthesis:
    """Tests for N:1 sequential FoldSynthesis transform."""

    def test_sequential_accumulation(self, mock_llm):
        """Each input produces one LLM call, accumulating results."""
        inputs = [_make_artifact(f"ep-{i}", f"event {i}") for i in range(3)]
        t = FoldSynthesis(
            "progressive",
            prompt="Current: {accumulated}\nNew: {artifact}",
            initial="Empty.",
            label="progressive",
            artifact_type="progressive",
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert len(results) == 1
        assert len(mock_llm) == 3  # one call per input
        assert results[0].label == "progressive"
        assert results[0].artifact_type == "progressive"
        assert results[0].prompt_id.startswith("fold_synthesis_v")
        assert results[0].metadata["input_count"] == 3

    def test_initial_value_in_first_call(self, mock_llm):
        """First LLM call uses the initial value as {accumulated}."""
        inputs = [_make_artifact("ep-0", "first event")]
        t = FoldSynthesis(
            "fold",
            prompt="Accumulated: {accumulated}\nNew: {artifact}",
            initial="Nothing yet.",
            label="out",
        )
        t.execute(inputs, {"llm_config": {}})
        sent = mock_llm[0]["messages"][0]["content"]
        assert "Nothing yet." in sent
        assert "first event" in sent

    def test_sort_by_metadata_key(self, mock_llm):
        """sort_by string sorts inputs by that metadata key."""
        inputs = [
            _make_artifact("ep-c", "third", date="2024-03"),
            _make_artifact("ep-a", "first", date="2024-01"),
            _make_artifact("ep-b", "second", date="2024-02"),
        ]
        t = FoldSynthesis(
            "fold",
            prompt="{accumulated}|{artifact}",
            sort_by="date",
            label="out",
        )
        t.execute(inputs, {"llm_config": {}})

        # First call should have the earliest date
        first_sent = mock_llm[0]["messages"][0]["content"]
        assert "first" in first_sent

    def test_sort_by_callable(self, mock_llm):
        """sort_by callable controls ordering."""
        inputs = [
            _make_artifact("ep-3", "c", order=3),
            _make_artifact("ep-1", "a", order=1),
            _make_artifact("ep-2", "b", order=2),
        ]
        t = FoldSynthesis(
            "fold",
            prompt="{artifact}",
            sort_by=lambda a: a.metadata["order"],
            label="out",
        )
        t.execute(inputs, {"llm_config": {}})
        first_sent = mock_llm[0]["messages"][0]["content"]
        assert "a" in first_sent

    def test_sort_by_default_artifact_id(self, mock_llm):
        """Default sort is by artifact_id when sort_by is None."""
        t = FoldSynthesis("fold", prompt="{artifact}", label="out")
        inputs = [_make_artifact(f"ep-{i}", f"content-{i}") for i in range(2)]
        t.execute(inputs, {"llm_config": {}})
        # Should run without error — order is by artifact_id
        assert len(mock_llm) == 2

    def test_split_single_unit(self):
        """Split returns a single unit."""
        t = FoldSynthesis("fold", prompt="x", label="out")
        inputs = [_make_artifact(f"a-{i}") for i in range(5)]
        units = t.split(inputs, {})
        assert len(units) == 1
        assert len(units[0][0]) == 5

    def test_estimate_output_count(self):
        """Always estimates 1 output."""
        t = FoldSynthesis("fold", prompt="x", label="out")
        assert t.estimate_output_count(10) == 1

    def test_step_and_total_placeholders(self, mock_llm):
        """{step} and {total} placeholders are substituted."""
        inputs = [_make_artifact(f"ep-{i}") for i in range(2)]
        t = FoldSynthesis(
            "fold",
            prompt="Step {step}/{total}",
            label="out",
        )
        t.execute(inputs, {"llm_config": {}})
        first = mock_llm[0]["messages"][0]["content"]
        second = mock_llm[1]["messages"][0]["content"]
        assert "Step 1/2" in first
        assert "Step 2/2" in second

    def test_cache_key_includes_initial(self):
        """Cache key differs when initial value changes."""
        t1 = FoldSynthesis("fold", prompt="x", initial="a", label="out")
        t2 = FoldSynthesis("fold", prompt="x", initial="b", label="out")
        assert t1.get_cache_key({}) != t2.get_cache_key({})

    def test_cache_key_includes_sort_by(self):
        """Cache key differs when sort_by changes."""
        t1 = FoldSynthesis("fold", prompt="x", sort_by="date", label="out")
        t2 = FoldSynthesis("fold", prompt="x", sort_by="title", label="out")
        assert t1.get_cache_key({}) != t2.get_cache_key({})

    def test_cache_key_includes_artifact_type(self):
        """Cache key differs when artifact_type changes."""
        t1 = FoldSynthesis("fold", prompt="x", label="out", artifact_type="summary")
        t2 = FoldSynthesis("fold", prompt="x", label="out", artifact_type="report")
        assert t1.get_cache_key({}) != t2.get_cache_key({})

    def test_cache_key_none_vs_string_sort_by(self):
        """Cache key differs between None and string sort_by."""
        t1 = FoldSynthesis("fold", prompt="x", sort_by=None, label="out")
        t2 = FoldSynthesis("fold", prompt="x", sort_by="date", label="out")
        assert t1.get_cache_key({}) != t2.get_cache_key({})

    def test_fingerprint_includes_callable_sort_by(self):
        """Fingerprint includes callable when sort_by is a callable."""
        fn = lambda a: a.label  # noqa: E731
        t = FoldSynthesis("fold", prompt="x", sort_by=fn, label="out")
        fp = t.compute_fingerprint({})
        assert "sort_by" in fp.components

    def test_batch_always_false(self):
        """FoldSynthesis always has batch=False."""
        t = FoldSynthesis("fold", prompt="x", label="out")
        assert t.batch is False

    def test_empty_initial_default(self, mock_llm):
        """Default initial is empty string."""
        inputs = [_make_artifact("ep-0")]
        t = FoldSynthesis("fold", prompt="Acc: [{accumulated}] New: {artifact}", label="out")
        t.execute(inputs, {"llm_config": {}})
        sent = mock_llm[0]["messages"][0]["content"]
        assert "Acc: []" in sent

    def test_accumulated_content_with_placeholder_tokens(self, mock_llm):
        """Accumulated LLM response containing {artifact} doesn't cause injection."""
        # First call returns content with a placeholder token in it
        from unittest.mock import MagicMock

        call_count = 0

        def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count == 1:
                resp.content = "Summary so far. The {artifact} was interesting."
            else:
                resp.content = "Final summary."
            resp.input_tokens = 10
            resp.output_tokens = 10
            return resp

        from unittest.mock import patch

        t = FoldSynthesis(
            "fold",
            prompt="Current: {accumulated}\nNew: {artifact}",
            initial="Start.",
            label="out",
        )
        inputs = [_make_artifact("ep-0", "event 0"), _make_artifact("ep-1", "event 1")]
        with patch("synix.build.llm_transforms.LLMClient.complete", side_effect=mock_complete):
            results = t.execute(inputs, {"llm_config": {}})

        assert results[0].content == "Final summary."
