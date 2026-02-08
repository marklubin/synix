"""Merge transform — groups artifacts by content similarity with constraints."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from synix.core.models import Artifact
from synix.build.transforms import BaseTransform, register_transform


def _tokenize(text: str) -> set[str]:
    """Convert text to a set of lowercased word tokens."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def _bigrams(text: str) -> set[tuple[str, str]]:
    """Convert text to a set of word bigrams for richer similarity."""
    words = re.findall(r'\b\w+\b', text.lower())
    return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts using word tokens + bigrams.

    Combines unigram and bigram token sets for better accuracy.
    Returns a float between 0.0 and 1.0.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    bigrams_a = _bigrams(text_a)
    bigrams_b = _bigrams(text_b)

    # Combine unigrams and bigrams (as string tuples for union/intersection)
    combined_a: set = tokens_a | {f"{a}_{b}" for a, b in bigrams_a}
    combined_b: set = tokens_b | {f"{a}_{b}" for a, b in bigrams_b}

    if not combined_a and not combined_b:
        return 1.0  # both empty
    if not combined_a or not combined_b:
        return 0.0

    intersection = combined_a & combined_b
    union = combined_a | combined_b
    return len(intersection) / len(union)


def _parse_constraints(constraints: list[str]) -> list[str]:
    """Extract metadata field names from constraint strings.

    Parses strings like "NEVER merge records with different customer_id"
    and returns the field name (e.g., "customer_id").
    """
    fields: list[str] = []
    for constraint in constraints:
        # Match patterns like "different <field_name>" or "distinct <field_name>"
        match = re.search(
            r'(?:different|distinct)\s+(\w+)',
            constraint,
            re.IGNORECASE,
        )
        if match:
            fields.append(match.group(1))
    return fields


def _violates_constraints(
    artifact_a: Artifact,
    artifact_b: Artifact,
    constraint_fields: list[str],
) -> bool:
    """Check if merging two artifacts would violate any constraints."""
    for field in constraint_fields:
        val_a = artifact_a.metadata.get(field)
        val_b = artifact_b.metadata.get(field)
        # If both have the field and values differ, constraint is violated
        if val_a is not None and val_b is not None and val_a != val_b:
            return True
    return False


def _generate_group_key(artifacts: list[Artifact]) -> str:
    """Generate a descriptive group key from the most common words in the group.

    Uses the most frequent significant words (>4 chars, not stopwords)
    across all artifacts in the group.
    """
    stopwords = {
        "about", "above", "after", "again", "being", "below", "between",
        "could", "during", "every", "first", "found", "their", "there",
        "these", "thing", "those", "through", "under", "using", "which",
        "while", "would", "other", "should", "still", "where", "before",
    }

    word_counts: dict[str, int] = defaultdict(int)
    for art in artifacts:
        words = re.findall(r'\b\w+\b', art.content.lower())
        for word in words:
            if len(word) > 4 and word not in stopwords:
                word_counts[word] += 1

    # Top 3 most common significant words
    top_words = sorted(word_counts, key=word_counts.get, reverse=True)[:3]  # type: ignore[arg-type]
    if top_words:
        return "_".join(top_words)

    # Fallback: use first artifact's ID
    return artifacts[0].artifact_id.replace(":", "_")


def _build_merge_groups(
    inputs: list[Artifact],
    threshold: float,
    constraint_fields: list[str],
) -> list[list[int]]:
    """Build merge groups using single-linkage clustering with constraints.

    Returns a list of groups, each group is a list of indices into inputs.
    """
    n = len(inputs)
    if n == 0:
        return []

    # Union-find for clustering
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_similarity(inputs[i].content, inputs[j].content)
            if sim >= threshold:
                # Check constraints before merging
                if not _violates_constraints(inputs[i], inputs[j], constraint_fields):
                    union(i, j)

    # Collect groups
    groups_map: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups_map[find(i)].append(i)

    return list(groups_map.values())


@register_transform("merge")
class MergeTransform(BaseTransform):
    """Merge artifacts by content similarity with optional constraints.

    Config options:
        similarity_threshold (float): Minimum similarity to merge. Default 0.85.
        constraints (list[str]): Natural language constraints, e.g.,
            ["NEVER merge records with different customer_id"]
        merge_prompt (str): Optional LLM prompt for synthesizing merged content.
            Not used in v0.1 -- content is concatenated.
    """

    def get_cache_key(self, config: dict) -> str:
        """Threshold and constraints affect output -- include in cache key."""
        threshold = config.get("similarity_threshold", 0.85)
        constraints = sorted(config.get("constraints", []))
        key_str = f"threshold={threshold}|constraints={','.join(constraints)}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:8]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        """Merge similar artifacts, respecting constraints.

        Args:
            inputs: List of artifacts to consider for merging.
            config: Transform configuration containing:
                - similarity_threshold: float (default 0.85)
                - constraints: list of constraint strings
                - merge_prompt: optional LLM prompt (unused in v0.1)

        Returns:
            List of artifacts, with similar ones merged. Count <= len(inputs).
        """
        threshold: float = config.get("similarity_threshold", 0.85)
        constraints: list[str] = config.get("constraints", [])
        constraint_fields = _parse_constraints(constraints)

        if not inputs:
            return []

        # Build merge groups
        groups = _build_merge_groups(inputs, threshold, constraint_fields)

        results: list[Artifact] = []
        for group_indices in groups:
            group_artifacts = [inputs[i] for i in group_indices]

            if len(group_artifacts) == 1:
                # Singleton -- pass through unchanged
                results.append(group_artifacts[0])
                continue

            # Multiple artifacts -- create merged artifact
            group_key = _generate_group_key(group_artifacts)
            artifact_id = f"merge-{group_key}"

            # Build merged content with source attribution headers
            content_parts: list[str] = []
            for art in group_artifacts:
                header = f"--- Source: {art.artifact_id}"
                customer_id = art.metadata.get("customer_id")
                if customer_id:
                    header += f" (customer: {customer_id})"
                header += " ---"
                content_parts.append(f"{header}\n{art.content}")
            merged_content = "\n\n".join(content_parts)

            # Collect metadata
            source_ids = [art.artifact_id for art in group_artifacts]
            source_customer_ids = list({
                art.metadata.get("customer_id", "unknown")
                for art in group_artifacts
                if art.metadata.get("customer_id")
            })

            # Compute average pairwise similarity for the group
            similarities: list[float] = []
            for i, idx_i in enumerate(group_indices):
                for j, idx_j in enumerate(group_indices):
                    if i < j:
                        sim = jaccard_similarity(
                            inputs[idx_i].content, inputs[idx_j].content
                        )
                        similarities.append(sim)
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            results.append(Artifact(
                artifact_id=artifact_id,
                artifact_type="merge",
                content=merged_content,
                input_hashes=[art.content_hash for art in group_artifacts],
                prompt_id=None,
                model_config=None,
                metadata={
                    "source_artifact_ids": source_ids,
                    "source_customer_ids": source_customer_ids,
                    "similarity_score": round(avg_similarity, 4),
                    "merge_count": len(group_artifacts),
                },
            ))

        return results
