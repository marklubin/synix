"""E2E tests for Synix MCP server.

Two modes:
1. Functional E2E — full lifecycle through MCP protocol (no API key needed)
2. Live agent — real Claude agent completes a task via MCP tools (needs ANTHROPIC_API_KEY)

Run functional tests:
    uv run pytest tests/e2e/test_mcp_agent.py -k "not live_agent"

Run live agent tests:
    ANTHROPIC_API_KEY=sk-ant-... uv run pytest tests/e2e/test_mcp_agent.py -k live_agent -v
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import synix

logger = logging.getLogger(__name__)

PIPELINE_PY = """\
from synix import Pipeline, Source, SearchSurface, SynixSearch, FlatFile
from synix.transforms import Chunk

pipeline = Pipeline("test-memory", source_dir="./sources")
docs = Source("docs")
chunks = Chunk("chunks", depends_on=[docs], chunk_size=200, chunk_overlap=50)
surface = SearchSurface("search", sources=[chunks], modes=["fulltext"])
search_out = SynixSearch("search", surface=surface)
context = FlatFile("context", sources=[chunks], output_path="context.md")
pipeline.add(docs, chunks, surface, search_out, context)
"""

RECIPE_DOC = """\
# Chocolate Cake Recipe

The best chocolate cake uses Dutch-process cocoa powder and buttermilk.

Preheat your oven to 350F. Mix dry ingredients: flour, cocoa, sugar,
baking soda, and salt. In a separate bowl, combine buttermilk, eggs,
oil, and vanilla extract. Combine wet and dry, then add hot coffee.

Let cool before frosting with chocolate ganache.
"""

PYTHON_DOC = """\
# Python Programming Tips

Use list comprehensions instead of map/filter for readability.
Type hints make code self-documenting and catch bugs early.
The walrus operator := can simplify while loops with assignment.

Virtual environments isolate project dependencies. Use uv for fast
package management. Always pin your dependencies in production.
"""


@pytest.fixture
def mcp_project(tmp_path):
    """Create a pre-initialized project with pipeline, empty sources."""
    (tmp_path / "pipeline.py").write_text(PIPELINE_PY)
    sources = tmp_path / "sources" / "docs"
    sources.mkdir(parents=True)
    synix.init(str(tmp_path))
    return tmp_path


@pytest.fixture
def mcp_project_scratch(tmp_path):
    """Prepare a directory for the agent to init from scratch.

    Has pipeline.py and an external file for source_add_file, but NO .synix/.
    """
    project_dir = tmp_path / "agent-project"
    project_dir.mkdir()
    (project_dir / "pipeline.py").write_text(PIPELINE_PY)
    # External file for source_add_file testing
    ext_file = tmp_path / "external-recipe.md"
    ext_file.write_text(RECIPE_DOC)
    return project_dir, ext_file


# ---------------------------------------------------------------------------
# Step validator — validates filesystem/state after each MCP tool call
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """Record of a single MCP tool call."""

    tool: str
    args: dict
    result: str
    error: bool = False


@dataclass
class ScenarioLog:
    """Full log of an agent scenario run."""

    calls: list[ToolCall] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [c.tool for c in self.calls]

    def calls_for(self, tool_name: str) -> list[ToolCall]:
        return [c for c in self.calls if c.tool == tool_name]

    def assert_tool_used(self, tool_name: str, msg: str = ""):
        assert tool_name in self.tool_names, (
            f"Expected tool {tool_name!r} to be called. "
            f"Actual calls: {self.tool_names}. {msg}"
        )

    def assert_order(self, *tool_names: str):
        """Assert tools were called in this relative order (not necessarily adjacent)."""
        indices = []
        for name in tool_names:
            try:
                idx = self.tool_names.index(name)
            except ValueError:
                raise AssertionError(f"Tool {name!r} never called. Calls: {self.tool_names}")
            indices.append(idx)
        assert indices == sorted(indices), (
            f"Expected order {list(tool_names)} but got indices {indices}. "
            f"Full call log: {self.tool_names}"
        )


class StepValidator:
    """Validates project state after each tool call."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.validations_run = 0

    def validate(self, call: ToolCall):
        """Run validation for a tool call if a validator exists.

        Skips errored calls — the agent may retry and succeed later.
        Only validates successful tool calls against filesystem state.
        """
        if call.error:
            logger.warning("Tool %s errored (agent may retry): %s", call.tool, call.result[:200])
            return
        validator = getattr(self, f"_check_{call.tool}", None)
        if validator:
            validator(call)
            self.validations_run += 1

    def _check_open_project(self, call: ToolCall):
        assert (self.project_path / ".synix").is_dir()

    def _check_load_pipeline(self, call: ToolCall):
        assert "test-memory" in call.result

    def _check_source_add_text(self, call: ToolCall):
        filename = call.args.get("filename", "")
        if filename:
            source_dir = self.project_path / "sources" / "docs"
            assert (source_dir / filename).exists(), f"Source file {filename} not created"

    def _check_build(self, call: ToolCall):
        assert (self.project_path / ".synix" / "objects").is_dir()

    def _check_release(self, call: ToolCall):
        name = call.args.get("name", "local")
        receipt = self.project_path / ".synix" / "releases" / name / "receipt.json"
        assert receipt.exists(), f"Release receipt not found at {receipt}"

    def _check_search(self, call: ToolCall):
        pass  # search may legitimately return no results

    def _check_list_layers(self, call: ToolCall):
        assert "docs" in call.result
        assert "chunks" in call.result

    def _check_list_artifacts(self, call: ToolCall):
        # Should have at least one artifact
        assert call.result and len(call.result) > 2  # non-empty JSON

    def _check_get_artifact(self, call: ToolCall):
        assert "content" in call.result

    def _check_lineage(self, call: ToolCall):
        # Lineage of a chunk should trace back to a source
        assert "docs" in call.result or call.result == "[]"

    def _check_list_releases(self, call: ToolCall):
        pass  # may be empty after clean

    def _check_show_release(self, call: ToolCall):
        assert "snapshot_oid" in call.result

    def _check_list_refs(self, call: ToolCall):
        pass  # just needs to not error

    def _check_source_remove(self, call: ToolCall):
        filename = call.args.get("filename", "")
        if filename:
            source_dir = self.project_path / "sources" / "docs"
            assert not (source_dir / filename).exists(), f"Source file {filename} should be removed"

    def _check_source_add_file(self, call: ToolCall):
        # Just check it didn't error — file existence validated by source_list
        pass

    def _check_source_clear(self, call: ToolCall):
        source_dir = self.project_path / "sources" / "docs"
        if source_dir.exists():
            files = list(source_dir.iterdir())
            assert len(files) == 0, f"source_clear should empty the dir, found: {files}"

    def _check_init_project(self, call: ToolCall):
        assert (self.project_path / ".synix").is_dir()

    def _check_get_flat_file(self, call: ToolCall):
        # Should return non-empty markdown content
        assert len(call.result) > 0

    def _check_clean(self, call: ToolCall):
        releases_dir = self.project_path / ".synix" / "releases"
        # After clean, releases dir should be gone or empty
        assert not releases_dir.exists() or not list(releases_dir.iterdir())


# ---------------------------------------------------------------------------
# Functional E2E — full lifecycle through MCP protocol
# ---------------------------------------------------------------------------


def _text(result) -> str:
    """Extract all text from an MCP CallToolResult, joining content blocks."""
    if not result.content:
        return ""
    parts = []
    for c in result.content:
        if hasattr(c, "text"):
            parts.append(c.text)
    return "\n".join(parts)


def _parse_artifacts(result) -> list[dict]:
    """Parse artifact list from MCP CallToolResult (handles FastMCP serialization)."""
    text = _text(result)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        arts = []
        for c in result.content:
            if hasattr(c, "text"):
                try:
                    parsed = json.loads(c.text)
                    if isinstance(parsed, dict):
                        arts.append(parsed)
                    elif isinstance(parsed, list):
                        arts.extend(parsed)
                except json.JSONDecodeError:
                    pass
        return arts


@pytest.mark.asyncio
async def test_mcp_protocol_lifecycle(mcp_project_scratch):
    """Connect to MCP server over stdio, exercise ALL 20 tools, validate each step."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    project_dir, ext_file = mcp_project_scratch

    # No SYNIX_PROJECT — we exercise init_project + open_project explicitly
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "synix.mcp"],
        env={k: v for k, v in os.environ.items() if k != "SYNIX_PROJECT"},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Verify all 20 tools are registered
            tools_result = await session.list_tools()
            tool_names = {t.name for t in tools_result.tools}
            assert len(tool_names) >= 20

            tools_called = set()

            async def call(name, args=None):
                result = await session.call_tool(name, args or {})
                assert not result.isError, f"{name} failed: {_text(result)}"
                tools_called.add(name)
                return result

            # --- Phase 1: Project lifecycle ---

            # 1. init_project
            await call("init_project", {"path": str(project_dir)})
            assert (project_dir / ".synix").is_dir()

            # 2. open_project
            await call("open_project", {"path": str(project_dir)})

            # 3. load_pipeline
            result = await call("load_pipeline", {"path": str(project_dir / "pipeline.py")})
            assert "test-memory" in _text(result)

            # 4. source_list (empty)
            result = await call("source_list", {"source_name": "docs"})

            # --- Phase 2: Add sources (both methods) ---

            # 5. source_add_file
            await call("source_add_file", {"source_name": "docs", "file_path": str(ext_file)})

            # 6. source_add_text
            await call("source_add_text", {
                "source_name": "docs",
                "content": PYTHON_DOC,
                "filename": "python-tips.md",
            })

            # 7. source_list (verify both)
            result = await call("source_list", {"source_name": "docs"})
            content = _text(result)
            assert "external-recipe.md" in content
            assert "python-tips.md" in content

            # --- Phase 3: Build & Release ---

            # 8. build
            result = await call("build")
            content = _text(result)
            assert "built" in content.lower() or "snapshot_oid" in content

            # 9. release
            await call("release", {"name": "local"})
            assert (project_dir / ".synix" / "releases" / "local" / "receipt.json").exists()

            # --- Phase 4: Search ---

            # 10. search
            result = await call("search", {
                "query": "chocolate cake",
                "release_name": "local",
                "mode": "keyword",
            })
            assert "chocolate" in _text(result).lower()

            # --- Phase 5: Full inspection ---

            # 11. list_layers
            result = await call("list_layers", {"release_name": "local"})
            content = _text(result)
            assert "docs" in content
            assert "chunks" in content

            # 12. list_artifacts (unfiltered)
            result = await call("list_artifacts", {"release_name": "local"})
            all_arts = _parse_artifacts(result)
            assert len(all_arts) > 0

            # 13. list_artifacts (filtered by layer)
            result = await call("list_artifacts", {"release_name": "local", "layer": "chunks"})
            chunk_arts = _parse_artifacts(result)
            assert len(chunk_arts) > 0
            assert all(a["layer"] == "chunks" for a in chunk_arts)

            # 14. get_artifact
            label = chunk_arts[0]["label"]
            result = await call("get_artifact", {"label": label, "release_name": "local"})
            assert "content" in _text(result)

            # 15. lineage
            result = await call("lineage", {"label": label, "release_name": "local"})
            assert "docs" in _text(result)

            # 16. get_flat_file
            result = await call("get_flat_file", {"name": "context", "release_name": "local"})
            assert len(_text(result)) > 0

            # 17. list_releases
            result = await call("list_releases")
            assert "local" in _text(result)

            # 18. show_release
            result = await call("show_release", {"name": "local"})
            assert "snapshot_oid" in _text(result)

            # 19. list_refs
            result = await call("list_refs")
            assert len(_text(result)) > 0

            # --- Phase 6: Modify sources ---

            # 20. source_remove
            await call("source_remove", {"source_name": "docs", "filename": "external-recipe.md"})
            assert not (project_dir / "sources" / "docs" / "external-recipe.md").exists()

            # 21. source_clear
            await call("source_clear", {"source_name": "docs"})
            source_dir = project_dir / "sources" / "docs"
            if source_dir.exists():
                assert list(source_dir.iterdir()) == []

            # --- Phase 7: Cleanup ---

            # 22. clean
            await call("clean")
            releases_dir = project_dir / ".synix" / "releases"
            assert not releases_dir.exists() or not list(releases_dir.iterdir())

            # 23. list_releases (confirm empty)
            result = await call("list_releases")

            # --- Final: verify all 20 tools were called ---
            ALL_TOOLS = {
                "init_project", "open_project", "load_pipeline",
                "source_list", "source_add_text", "source_add_file",
                "source_remove", "source_clear",
                "build", "release", "search",
                "get_artifact", "list_artifacts", "list_layers", "lineage",
                "list_releases", "show_release", "get_flat_file", "list_refs",
                "clean",
            }
            missing = ALL_TOOLS - tools_called
            assert not missing, f"Tools not exercised: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Live agent — real LLM + MCP tools (supports OpenAI and Anthropic backends)
# ---------------------------------------------------------------------------


def _mcp_tools_to_openai(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def _mcp_tools_to_anthropic(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to Anthropic API format."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in mcp_tools
    ]


def _extract_mcp_result_text(mcp_result) -> tuple[str, bool]:
    """Extract text and error status from an MCP CallToolResult."""
    is_error = bool(mcp_result.isError)
    if not mcp_result.content:
        return "", is_error
    parts = []
    for c in mcp_result.content:
        if hasattr(c, "text"):
            parts.append(c.text)
        else:
            parts.append(str(c))
    return "\n".join(parts), is_error


async def _run_openai_agent(session, task, mcp_tools, validator, model, max_turns):
    """Agent loop using OpenAI chat completions with function calling."""
    import openai

    client = openai.OpenAI()
    tools = _mcp_tools_to_openai(mcp_tools)
    messages = [{"role": "user", "content": task}]
    log = ScenarioLog()

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            tools=tools,
            messages=messages,
            max_tokens=4096,
        )
        choice = response.choices[0]

        if choice.finish_reason != "tool_calls":
            break

        # Append assistant message with tool calls
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            fn = tool_call.function
            args = json.loads(fn.arguments) if fn.arguments else {}

            mcp_result = await session.call_tool(fn.name, args)
            result_text, is_error = _extract_mcp_result_text(mcp_result)

            call = ToolCall(tool=fn.name, args=args, result=result_text, error=is_error)
            log.calls.append(call)

            if validator:
                validator.validate(call)

            logger.info(
                "Turn %d: %s(%s) -> %s",
                turn, fn.name, json.dumps(args, default=str)[:100], result_text[:200],
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })
    else:
        raise AssertionError(f"Agent did not finish within {max_turns} turns. Calls: {log.tool_names}")

    return log


async def _run_anthropic_agent(session, task, mcp_tools, validator, model, max_turns):
    """Agent loop using Anthropic messages API with tool_use."""
    import anthropic

    client = anthropic.Anthropic()
    tools = _mcp_tools_to_anthropic(mcp_tools)
    messages = [{"role": "user", "content": task}]
    log = ScenarioLog()

    for turn in range(max_turns):
        response = client.messages.create(
            model=model, tools=tools, messages=messages, max_tokens=4096,
        )

        if response.stop_reason == "end_turn":
            break

        tool_result_blocks = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            mcp_result = await session.call_tool(block.name, block.input or {})
            result_text, is_error = _extract_mcp_result_text(mcp_result)

            call = ToolCall(tool=block.name, args=block.input or {}, result=result_text, error=is_error)
            log.calls.append(call)

            if validator:
                validator.validate(call)

            logger.info(
                "Turn %d: %s(%s) -> %s",
                turn, block.name, json.dumps(block.input, default=str)[:100], result_text[:200],
            )

            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_result_blocks})
    else:
        raise AssertionError(f"Agent did not finish within {max_turns} turns. Calls: {log.tool_names}")

    return log


# Default backend/model selection: prefer OpenAI (wider connectivity), fall back to Anthropic
AGENT_BACKEND = os.environ.get("AGENT_BACKEND", "openai")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4o" if AGENT_BACKEND == "openai" else "claude-sonnet-4-20250514")


async def run_agent_scenario(
    session,
    task: str,
    validator: StepValidator | None = None,
    backend: str = AGENT_BACKEND,
    model: str = AGENT_MODEL,
    max_turns: int = 25,
) -> ScenarioLog:
    """Run an LLM agent against MCP tools and log every step.

    Args:
        session: Active MCP ClientSession.
        task: The task prompt for the agent.
        validator: Optional StepValidator to check state after each tool call.
        backend: "openai" or "anthropic".
        model: Model name for the chosen backend.
        max_turns: Max agent turns before aborting.

    Returns:
        ScenarioLog with all tool calls recorded.

    Configure via env vars:
        AGENT_BACKEND=openai|anthropic
        AGENT_MODEL=gpt-4o-mini|claude-sonnet-4-20250514|...
    """
    tools_result = await session.list_tools()

    if backend == "openai":
        return await _run_openai_agent(session, task, tools_result.tools, validator, model, max_turns)
    elif backend == "anthropic":
        return await _run_anthropic_agent(session, task, tools_result.tools, validator, model, max_turns)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'openai' or 'anthropic'.")


_has_api_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.live_agent
@pytest.mark.asyncio
@pytest.mark.skipif(
    not _has_api_key,
    reason="OPENAI_API_KEY or ANTHROPIC_API_KEY required for live agent tests",
)
async def test_live_agent_memory_lifecycle(mcp_project_scratch):
    """A real LLM agent exercises ALL 20 MCP tools end-to-end.

    Covers: init_project, open_project, load_pipeline, source_list,
    source_add_text, source_add_file, source_remove, source_clear,
    build, release, search, get_artifact, list_artifacts, list_layers,
    lineage, list_releases, show_release, get_flat_file, list_refs, clean.

    Each step is validated against filesystem state.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    project_dir, ext_file = mcp_project_scratch

    # No SYNIX_PROJECT — agent must init/open manually
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "synix.mcp"],
        env={k: v for k, v in os.environ.items() if k != "SYNIX_PROJECT"},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            validator = StepValidator(project_dir)

            task = f"""\
Complete ALL of the following steps in order. You MUST call every tool
mentioned — do not skip any step. Call each tool exactly as described.

PHASE 1 — Create project:
1. Call init_project with path "{project_dir}" to create a new synix project.
2. Call open_project with path "{project_dir}" to open it.
3. Call load_pipeline with path "{project_dir}/pipeline.py".
4. Call source_list with source_name "docs" to see current files (should be empty).

PHASE 2 — Add sources (use BOTH methods):
5. Call source_add_file with source_name "docs" and file_path "{ext_file}"
   to copy the external recipe file into the source directory.
6. Call source_add_text with source_name "docs" and write a guide to Python testing
   (filename: testing-guide.md). Write realistic content about pytest, fixtures,
   and mocking. Must mention "pytest" and "fixtures".
7. Call source_list with source_name "docs" to confirm both files are present.

PHASE 3 — Build & Release:
8.  Call build to build the pipeline.
9.  Call release with name "local" to materialize projections.

PHASE 4 — Search:
10. Call search with query "chocolate" and release_name "local", mode "keyword".
11. Call search with query "pytest" and release_name "local", mode "keyword".

PHASE 5 — Full inspection (call every inspect tool):
12. Call list_layers with release_name "local".
13. Call list_artifacts with release_name "local" (no layer filter).
14. Call list_artifacts with release_name "local" and layer "chunks".
15. Pick one chunk artifact label from step 14 and call get_artifact
    with that label and release_name "local".
16. Call lineage with that same label and release_name "local".
17. Call get_flat_file with name "context" and release_name "local"
    to read the flat file projection content.
18. Call list_releases.
19. Call show_release with name "local".
20. Call list_refs.

PHASE 6 — Modify sources:
21. Call source_remove with source_name "docs" and filename "external-recipe.md".
22. Call source_list to confirm removal.

PHASE 7 — Final cleanup:
23. Call source_clear with source_name "docs" to remove all remaining source files.
24. Call source_list to confirm the source directory is empty.
25. Call clean to remove releases and work directories.
26. Call list_releases to confirm no releases remain.

Report what you found at each phase."""

            log = await run_agent_scenario(
                session, task, validator=validator, max_turns=40
            )

            # --- Post-run assertions: all 20 tools must be called ---

            ALL_TOOLS = {
                "init_project", "open_project", "load_pipeline",
                "source_list", "source_add_text", "source_add_file",
                "source_remove", "source_clear",
                "build", "release", "search",
                "get_artifact", "list_artifacts", "list_layers", "lineage",
                "list_releases", "show_release", "get_flat_file", "list_refs",
                "clean",
            }

            actual_tools = set(log.tool_names)
            missing = ALL_TOOLS - actual_tools
            assert not missing, (
                f"Agent missed {len(missing)} tools: {sorted(missing)}. "
                f"Called: {sorted(actual_tools)}"
            )

            # Ordering constraints
            log.assert_order("init_project", "load_pipeline", "build", "release", "search")
            log.assert_order("build", "release", "list_layers", "clean")

            # Search returned relevant content
            search_calls = log.calls_for("search")
            assert len(search_calls) >= 2, f"Expected 2+ search calls, got {len(search_calls)}"
            assert any(
                "chocolate" in c.result.lower() or "pytest" in c.result.lower()
                for c in search_calls
            ), "Agent searches returned no relevant content"

            # source_add_text + source_add_file both used
            assert len(log.calls_for("source_add_text")) >= 1
            assert len(log.calls_for("source_add_file")) >= 1

            # get_flat_file returned content
            flat_calls = log.calls_for("get_flat_file")
            assert len(flat_calls) >= 1
            assert len(flat_calls[0].result) > 0, "get_flat_file returned empty content"

            # After source_clear + clean, directory should be empty
            source_dir = project_dir / "sources" / "docs"
            if source_dir.exists():
                assert list(source_dir.iterdir()) == [], "source_clear should empty the dir"

            logger.info(
                "Live agent test passed: %d tool calls, %d/%d unique tools, %d validations",
                len(log.calls),
                len(actual_tools),
                len(ALL_TOOLS),
                validator.validations_run,
            )
