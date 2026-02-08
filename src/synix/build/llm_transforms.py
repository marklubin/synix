"""LLM-powered transforms — episode summary, rollups, core synthesis."""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict

from synix.core.config import LLMConfig
from synix.core.models import Artifact
from synix.build.llm_client import LLMClient, LLMResponse
from synix.build.transforms import BaseTransform, register_transform


def _make_llm_client(config: dict) -> LLMClient:
    """Create an LLMClient from the transform config dict.

    Handles backward compatibility: config dicts that don't include
    'provider' default to Anthropic.
    """
    llm_config = LLMConfig.from_dict(config.get("llm_config", {}))
    return LLMClient(llm_config)


def _logged_complete(
    client: LLMClient,
    config: dict,
    messages: list[dict],
    artifact_desc: str,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Call client.complete() with optional logging via the SynixLogger.

    If config contains '_logger' and '_layer_name' (injected by the runner),
    logs LLM call start/finish with timing and token counts.
    """
    logger = config.get("_logger")
    layer_name = config.get("_layer_name", "unknown")
    model = config.get("llm_config", {}).get("model", "unknown")

    start_time = None
    if logger is not None:
        start_time = logger.llm_call_start(layer_name, artifact_desc, model)

    kwargs: dict = {
        "messages": messages,
        "artifact_desc": artifact_desc,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.complete(**kwargs)

    if logger is not None and start_time is not None:
        logger.llm_call_finish(
            layer_name,
            artifact_desc,
            start_time,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    return response


@register_transform("episode_summary")
class EpisodeSummaryTransform(BaseTransform):
    """One transcript → one episode summary."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        template = self.load_prompt("episode_summary")
        prompt_id = self.get_prompt_id("episode_summary")
        client = _make_llm_client(config)
        model_config = config.get("llm_config", {})

        results: list[Artifact] = []
        for transcript in inputs:
            prompt = template.replace("{transcript}", transcript.content)
            conv_id = transcript.metadata.get(
                "source_conversation_id", transcript.artifact_id
            )

            response = _logged_complete(
                client, config,
                messages=[{"role": "user", "content": prompt}],
                artifact_desc=f"episode ep-{conv_id}",
            )

            ep_metadata = {
                    "source_conversation_id": conv_id,
                    "date": transcript.metadata.get("date", ""),
                    "title": transcript.metadata.get("title", ""),
            }
            # Propagate customer_id from source transcript if present
            if transcript.metadata.get("customer_id"):
                ep_metadata["customer_id"] = transcript.metadata["customer_id"]

            results.append(Artifact(
                artifact_id=f"ep-{conv_id}",
                artifact_type="episode",
                content=response.content,
                input_hashes=[transcript.content_hash],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata=ep_metadata,
            ))

        return results


@register_transform("monthly_rollup")
class MonthlyRollupTransform(BaseTransform):
    """Group episodes by month, synthesize each month."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        template = self.load_prompt("monthly_rollup")
        prompt_id = self.get_prompt_id("monthly_rollup")
        client = _make_llm_client(config)
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

            response = _logged_complete(
                client, config,
                messages=[{"role": "user", "content": prompt}],
                artifact_desc=f"monthly rollup {month}",
            )

            results.append(Artifact(
                artifact_id=f"monthly-{month}",
                artifact_type="rollup",
                content=response.content,
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
        client = _make_llm_client(config)
        model_config = config.get("llm_config", {})
        topics = config.get("topics", [])

        # Optionally query a search index for relevant episodes per topic
        search_db_path = config.get("search_db_path")
        index = None
        if search_db_path:
            from pathlib import Path

            from synix.search.indexer import SearchIndex

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
                response = _logged_complete(
                    client, config,
                    messages=[{"role": "user", "content": prompt}],
                    artifact_desc=f"topical rollup topic-{slug}",
                )

                results.append(Artifact(
                    artifact_id=f"topic-{slug}",
                    artifact_type="rollup",
                    content=response.content,
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
        client = _make_llm_client(config)
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

        response = _logged_complete(
            client, config,
            messages=[{"role": "user", "content": prompt}],
            artifact_desc="core memory synthesis",
            max_tokens=max_tokens,
        )

        return [Artifact(
            artifact_id="core-memory",
            artifact_type="core_memory",
            content=response.content,
            input_hashes=[r.content_hash for r in inputs],
            prompt_id=prompt_id,
            model_config=model_config,
            metadata={"context_budget": context_budget, "input_count": len(inputs)},
        )]
