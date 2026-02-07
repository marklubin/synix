"""LLM-powered transforms — episode summary, rollups, core synthesis."""

from __future__ import annotations

import hashlib
import sys
import time
from collections import defaultdict

import anthropic

from synix import Artifact
from synix.transforms.base import BaseTransform, register_transform


def _llm_call_with_retry(client: anthropic.Anthropic, *, model: str, max_tokens: int,
                         temperature: float, messages: list[dict],
                         artifact_desc: str = "artifact") -> str:
    """Call the Anthropic API with 1 retry on transient errors.

    Returns the text content of the first response block.
    """
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response.content[0].text
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            if attempt == 0:
                print(f"[synix] Transient error processing {artifact_desc}, retrying in 5s: {exc}",
                      file=sys.stderr)
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"Failed to process {artifact_desc} after 2 attempts: {exc}"
                ) from exc
        except anthropic.APIError as exc:
            raise RuntimeError(
                f"LLM API error processing {artifact_desc}: {exc}"
            ) from exc
    # unreachable, but satisfies type checker
    raise RuntimeError(f"Failed to process {artifact_desc}")


@register_transform("episode_summary")
class EpisodeSummaryTransform(BaseTransform):
    """One transcript → one episode summary."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        template = self.load_prompt("episode_summary")
        prompt_id = self.get_prompt_id("episode_summary")
        client = anthropic.Anthropic()
        model_config = config.get("llm_config", {})

        results: list[Artifact] = []
        for transcript in inputs:
            prompt = template.replace("{transcript}", transcript.content)
            conv_id = transcript.metadata.get(
                "source_conversation_id", transcript.artifact_id
            )

            content = _llm_call_with_retry(
                client,
                model=model_config.get("model", "claude-haiku-4-5-20251001"),
                max_tokens=model_config.get("max_tokens", 1024),
                temperature=model_config.get("temperature", 0.3),
                messages=[{"role": "user", "content": prompt}],
                artifact_desc=f"episode ep-{conv_id}",
            )

            results.append(Artifact(
                artifact_id=f"ep-{conv_id}",
                artifact_type="episode",
                content=content,
                input_hashes=[transcript.content_hash],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={
                    "source_conversation_id": conv_id,
                    "date": transcript.metadata.get("date", ""),
                    "title": transcript.metadata.get("title", ""),
                },
            ))

        return results


@register_transform("monthly_rollup")
class MonthlyRollupTransform(BaseTransform):
    """Group episodes by month, synthesize each month."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        template = self.load_prompt("monthly_rollup")
        prompt_id = self.get_prompt_id("monthly_rollup")
        client = anthropic.Anthropic()
        model_config = config.get("llm_config", {})

        # Group by month (YYYY-MM from metadata.date)
        months: dict[str, list[Artifact]] = defaultdict(list)
        for ep in inputs:
            month = ep.metadata.get("date", "")[:7]  # YYYY-MM
            if month and "-" in month:
                months[month].append(ep)
            else:
                print(f"[synix] Warning: episode '{ep.artifact_id}' has no date metadata, "
                      f"grouping as 'undated'", file=sys.stderr)
                months["undated"].append(ep)

        results: list[Artifact] = []
        for month, episodes in sorted(months.items()):
            if month == "undated":
                year, mo = "unknown", "undated"
            else:
                year, mo = month.split("-")
            episodes_text = "\n\n---\n\n".join(
                f"### {ep.metadata.get('title', ep.artifact_id)} ({ep.metadata.get('date', '')})\n{ep.content}"
                for ep in episodes
            )
            prompt = (
                template
                .replace("{month}", mo)
                .replace("{year}", year)
                .replace("{episodes}", episodes_text)
            )

            content = _llm_call_with_retry(
                client,
                model=model_config.get("model", "claude-haiku-4-5-20251001"),
                max_tokens=model_config.get("max_tokens", 1024),
                temperature=model_config.get("temperature", 0.3),
                messages=[{"role": "user", "content": prompt}],
                artifact_desc=f"monthly rollup {month}",
            )

            results.append(Artifact(
                artifact_id=f"monthly-{month}",
                artifact_type="rollup",
                content=content,
                input_hashes=[ep.content_hash for ep in episodes],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={"month": month, "episode_count": len(episodes)},
            ))

        return results


@register_transform("topical_rollup")
class TopicalRollupTransform(BaseTransform):
    """Group episodes by topic, synthesize each topic."""

    def get_cache_key(self, config: dict) -> str:
        """Topics list affects output — include in cache key."""
        topics = sorted(config.get("topics", []))
        return hashlib.sha256(",".join(topics).encode()).hexdigest()[:8]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        template = self.load_prompt("topical_rollup")
        prompt_id = self.get_prompt_id("topical_rollup")
        client = anthropic.Anthropic()
        model_config = config.get("llm_config", {})
        topics = config.get("topics", [])

        # Optionally query a search index for relevant episodes per topic
        search_db_path = config.get("search_db_path")
        index = None
        if search_db_path:
            from pathlib import Path

            from synix.search.index import SearchIndex

            index = SearchIndex(Path(search_db_path))

        results: list[Artifact] = []
        try:
            for topic in topics:
                # Find relevant episodes
                if index is not None:
                    search_results = index.query(
                        topic.replace("-", " "),
                        layers=["episodes"],
                    )
                    matching_ids = {r.artifact_id for r in search_results}
                    relevant = [ep for ep in inputs if ep.artifact_id in matching_ids]
                    if not relevant:
                        relevant = inputs  # fallback: use all
                else:
                    relevant = inputs  # use all episodes

                episodes_text = "\n\n---\n\n".join(
                    f"### {ep.metadata.get('title', ep.artifact_id)} ({ep.metadata.get('date', '')})\n{ep.content}"
                    for ep in relevant
                )
                prompt = (
                    template
                    .replace("{topic}", topic.replace("-", " "))
                    .replace("{episodes}", episodes_text)
                )

                slug = topic.lower().replace(" ", "-")
                content = _llm_call_with_retry(
                    client,
                    model=model_config.get("model", "claude-haiku-4-5-20251001"),
                    max_tokens=model_config.get("max_tokens", 1024),
                    temperature=model_config.get("temperature", 0.3),
                    messages=[{"role": "user", "content": prompt}],
                    artifact_desc=f"topical rollup topic-{slug}",
                )

                results.append(Artifact(
                    artifact_id=f"topic-{slug}",
                    artifact_type="rollup",
                    content=content,
                    input_hashes=[ep.content_hash for ep in relevant],
                    prompt_id=prompt_id,
                    model_config=model_config,
                    metadata={"topic": topic, "episode_count": len(relevant)},
                ))
        finally:
            if index is not None:
                index.close()

        return results


@register_transform("core_synthesis")
class CoreSynthesisTransform(BaseTransform):
    """All rollups → single core memory document."""

    def get_cache_key(self, config: dict) -> str:
        """context_budget affects output — include in cache key."""
        budget = config.get("context_budget", 10000)
        return hashlib.sha256(str(budget).encode()).hexdigest()[:8]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        template = self.load_prompt("core_memory")
        prompt_id = self.get_prompt_id("core_memory")
        client = anthropic.Anthropic()
        model_config = config.get("llm_config", {})
        context_budget = config.get("context_budget", 10000)

        # Derive max_tokens from context_budget; fall back to model_config
        max_tokens = context_budget if context_budget else model_config.get("max_tokens", 2048)

        rollups_text = "\n\n---\n\n".join(
            f"### {r.metadata.get('month', r.metadata.get('topic', r.artifact_id))}\n{r.content}"
            for r in inputs
        )
        prompt = (
            template
            .replace("{context_budget}", str(context_budget))
            .replace("{rollups}", rollups_text)
        )

        content = _llm_call_with_retry(
            client,
            model=model_config.get("model", "claude-haiku-4-5-20251001"),
            max_tokens=max_tokens,
            temperature=model_config.get("temperature", 0.3),
            messages=[{"role": "user", "content": prompt}],
            artifact_desc="core memory synthesis",
        )

        return [Artifact(
            artifact_id="core-memory",
            artifact_type="core_memory",
            content=content,
            input_hashes=[r.content_hash for r in inputs],
            prompt_id=prompt_id,
            model_config=model_config,
            metadata={"context_budget": context_budget, "input_count": len(inputs)},
        )]
