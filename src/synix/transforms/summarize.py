"""LLM-powered transforms — episode summary, rollups, core synthesis."""

from __future__ import annotations

from collections import defaultdict

import anthropic

from synix import Artifact
from synix.transforms.base import BaseTransform, register_transform


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

            response = client.messages.create(
                model=model_config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=model_config.get("max_tokens", 1024),
                temperature=model_config.get("temperature", 0.3),
                messages=[{"role": "user", "content": prompt}],
            )

            conv_id = transcript.metadata.get(
                "source_conversation_id", transcript.artifact_id
            )
            results.append(Artifact(
                artifact_id=f"ep-{conv_id}",
                artifact_type="episode",
                content=response.content[0].text,
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
            if month:
                months[month].append(ep)

        results: list[Artifact] = []
        for month, episodes in sorted(months.items()):
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

            response = client.messages.create(
                model=model_config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=model_config.get("max_tokens", 1024),
                temperature=model_config.get("temperature", 0.3),
                messages=[{"role": "user", "content": prompt}],
            )

            results.append(Artifact(
                artifact_id=f"monthly-{month}",
                artifact_type="rollup",
                content=response.content[0].text,
                input_hashes=[ep.content_hash for ep in episodes],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={"month": month, "episode_count": len(episodes)},
            ))

        return results


@register_transform("topical_rollup")
class TopicalRollupTransform(BaseTransform):
    """Group episodes by topic, synthesize each topic."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        template = self.load_prompt("topical_rollup")
        prompt_id = self.get_prompt_id("topical_rollup")
        client = anthropic.Anthropic()
        model_config = config.get("llm_config", {})
        topics = config.get("topics", [])

        # Optionally query a search index for relevant episodes per topic
        search_db_path = config.get("search_db_path")

        results: list[Artifact] = []
        for topic in topics:
            # Find relevant episodes
            if search_db_path:
                from pathlib import Path

                from synix.search.index import SearchIndex

                index = SearchIndex(Path(search_db_path))
                search_results = index.query(
                    topic.replace("-", " "),
                    layers=["episodes"],
                )
                index.close()
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

            response = client.messages.create(
                model=model_config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=model_config.get("max_tokens", 1024),
                temperature=model_config.get("temperature", 0.3),
                messages=[{"role": "user", "content": prompt}],
            )

            slug = topic.lower().replace(" ", "-")
            results.append(Artifact(
                artifact_id=f"topic-{slug}",
                artifact_type="rollup",
                content=response.content[0].text,
                input_hashes=[ep.content_hash for ep in relevant],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={"topic": topic, "episode_count": len(relevant)},
            ))

        return results


@register_transform("core_synthesis")
class CoreSynthesisTransform(BaseTransform):
    """All rollups → single core memory document."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        template = self.load_prompt("core_memory")
        prompt_id = self.get_prompt_id("core_memory")
        client = anthropic.Anthropic()
        model_config = config.get("llm_config", {})
        context_budget = config.get("context_budget", 10000)

        rollups_text = "\n\n---\n\n".join(
            f"### {r.metadata.get('month', r.metadata.get('topic', r.artifact_id))}\n{r.content}"
            for r in inputs
        )
        prompt = (
            template
            .replace("{context_budget}", str(context_budget))
            .replace("{rollups}", rollups_text)
        )

        response = client.messages.create(
            model=model_config.get("model", "claude-sonnet-4-20250514"),
            max_tokens=model_config.get("max_tokens", 2048),
            temperature=model_config.get("temperature", 0.3),
            messages=[{"role": "user", "content": prompt}],
        )

        return [Artifact(
            artifact_id="core-memory",
            artifact_type="core_memory",
            content=response.content[0].text,
            input_hashes=[r.content_hash for r in inputs],
            prompt_id=prompt_id,
            model_config=model_config,
            metadata={"context_budget": context_budget, "input_count": len(inputs)},
        )]
