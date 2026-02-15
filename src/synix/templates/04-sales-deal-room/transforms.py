# transforms.py — Custom transforms for the Sales Deal Room demo
#
# Three LLM transforms that build citation-backed competitive intelligence:
#   1. deal_competitive_intel — per-competitor analysis citing source docs
#   2. deal_strategy — positioning document synthesizing all intel
#   3. deal_call_prep — actionable call prep brief with citations

from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.build.transforms import BaseTransform, register_transform
from synix.core.citations import make_uri
from synix.core.models import Artifact


def _source_label_block(inputs: list[Artifact]) -> str:
    """Build an 'Available sources' block listing input labels → synix:// URIs."""
    lines = []
    for a in sorted(inputs, key=lambda a: a.label):
        lines.append(f"  {a.label} → {make_uri(a.label)}")
    return "\n".join(lines)


_CITATION_INSTRUCTION = (
    "When making a factual claim, cite the source using a markdown link: "
    "`[description](synix://label)`. Every substantive claim (numbers, dates, "
    "competitive facts) must have at least one citation."
)


@register_transform("deal_competitive_intel")
class DealCompetitiveIntelTransform(BaseTransform):
    """Analyze each competitor against our product specs, with citations."""

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        competitors = sorted(
            [a for a in inputs if a.metadata.get("layer_name") == "competitor_docs"],
            key=lambda a: a.label,
        )
        product_specs = sorted(
            [a for a in inputs if a.metadata.get("layer_name") == "product_specs"],
            key=lambda a: a.label,
        )
        return [([comp] + product_specs, {}) for comp in competitors]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        client = _get_llm_client(config)
        prompt_id = "deal_competitive_intel"
        model_config = config.get("llm_config", {})

        # Sort for determinism
        inputs = sorted(inputs, key=lambda a: a.artifact_id)

        competitor = next(
            (a for a in inputs if a.metadata.get("layer_name") == "competitor_docs"),
            inputs[0],
        )
        product_specs = [a for a in inputs if a.metadata.get("layer_name") == "product_specs"]

        sources_block = _source_label_block(inputs)

        specs_text = "\n\n---\n\n".join(f"### {a.label}\n{a.content}" for a in product_specs)

        prompt = (
            "You are a competitive intelligence analyst preparing a deal-specific "
            "competitor analysis.\n\n"
            f"<competitor_profile>\n{competitor.content}\n</competitor_profile>\n\n"
            f"<our_product>\n{specs_text}\n</our_product>\n\n"
            "Available sources for citation:\n"
            f"{sources_block}\n\n"
            f"{_CITATION_INSTRUCTION}\n\n"
            "Write a competitive analysis covering:\n"
            "1. Competitor overview and market position\n"
            "2. Head-to-head feature comparison\n"
            "3. Pricing comparison\n"
            "4. Their weaknesses we can exploit\n"
            "5. Their strengths we must address\n"
            "6. Recommended talking points against this competitor\n\n"
            "Be specific — include exact numbers and cite every factual claim."
        )

        response = _logged_complete(
            client,
            config,
            messages=[{"role": "user", "content": prompt}],
            artifact_desc=f"competitive intel {competitor.label}",
        )

        safe_name = competitor.label.replace("t-text-", "")
        return [
            Artifact(
                label=f"intel-{safe_name}",
                artifact_type="competitive_intel",
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={"competitor": safe_name},
            )
        ]


@register_transform("deal_strategy")
class DealStrategyTransform(BaseTransform):
    """Synthesize all competitive intel + deal context into a positioning strategy."""

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        return [(inputs, {})]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        client = _get_llm_client(config)
        prompt_id = "deal_strategy"
        model_config = config.get("llm_config", {})

        inputs = sorted(inputs, key=lambda a: a.artifact_id)
        sources_block = _source_label_block(inputs)

        intel = [a for a in inputs if a.artifact_type == "competitive_intel"]
        deal = [a for a in inputs if a.metadata.get("layer_name") == "deal_context"]
        wins = [a for a in inputs if a.metadata.get("layer_name") == "win_reports"]

        intel_text = "\n\n---\n\n".join(f"### {a.label}\n{a.content}" for a in intel)
        deal_text = "\n\n---\n\n".join(f"### {a.label}\n{a.content}" for a in deal)
        wins_text = "\n\n---\n\n".join(f"### {a.label}\n{a.content}" for a in wins)

        prompt = (
            "You are a sales strategist preparing a deal positioning document.\n\n"
            f"<competitive_intelligence>\n{intel_text}\n</competitive_intelligence>\n\n"
            f"<deal_context>\n{deal_text}\n</deal_context>\n\n"
            f"<past_deal_reports>\n{wins_text}\n</past_deal_reports>\n\n"
            "Available sources for citation:\n"
            f"{sources_block}\n\n"
            f"{_CITATION_INSTRUCTION}\n\n"
            "Write a strategic positioning document covering:\n"
            "1. Deal summary and opportunity assessment\n"
            "2. Competitive landscape and our positioning\n"
            "3. Key differentiators to emphasize\n"
            "4. Pricing strategy and value justification\n"
            "5. Risk factors and mitigation strategies\n"
            "6. Win themes and proof points from past deals\n\n"
            "Every claim about competitors, pricing, or capabilities must cite its source."
        )

        response = _logged_complete(
            client,
            config,
            messages=[{"role": "user", "content": prompt}],
            artifact_desc="deal strategy",
        )

        return [
            Artifact(
                label="strategy",
                artifact_type="strategic_positioning",
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id=prompt_id,
                model_config=model_config,
            )
        ]


@register_transform("deal_call_prep")
class DealCallPrepTransform(BaseTransform):
    """Produce a final call prep brief from strategy + deal context."""

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        return [(inputs, {})]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        client = _get_llm_client(config)
        prompt_id = "deal_call_prep"
        model_config = config.get("llm_config", {})

        inputs = sorted(inputs, key=lambda a: a.artifact_id)
        sources_block = _source_label_block(inputs)

        strategy = next((a for a in inputs if a.artifact_type == "strategic_positioning"), None)
        deal = [a for a in inputs if a.metadata.get("layer_name") == "deal_context"]

        strategy_text = strategy.content if strategy else "(no strategy document)"
        deal_text = "\n\n---\n\n".join(f"### {a.label}\n{a.content}" for a in deal)

        prompt = (
            "You are preparing a final call prep brief for an upcoming sales meeting.\n\n"
            f"<strategy_document>\n{strategy_text}\n</strategy_document>\n\n"
            f"<deal_context>\n{deal_text}\n</deal_context>\n\n"
            "Available sources for citation:\n"
            f"{sources_block}\n\n"
            f"{_CITATION_INSTRUCTION}\n\n"
            "Write an actionable call prep brief covering:\n"
            "1. Meeting objective and desired outcome\n"
            "2. Key talking points per stakeholder\n"
            "3. Anticipated objections and responses\n"
            "4. Competitive traps to set and landmines to avoid\n"
            "5. Proof points and customer references to mention\n"
            "6. Next steps to propose\n\n"
            "Keep it concise and actionable. Cite every competitive claim and data point."
        )

        response = _logged_complete(
            client,
            config,
            messages=[{"role": "user", "content": prompt}],
            artifact_desc="call prep brief",
        )

        return [
            Artifact(
                label="call-prep",
                artifact_type="call_prep_brief",
                content=response.content,
                input_ids=[a.artifact_id for a in inputs],
                prompt_id=prompt_id,
                model_config=model_config,
            )
        ]
