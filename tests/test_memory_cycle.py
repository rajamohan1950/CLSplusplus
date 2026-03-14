"""Tests for CLS++ Memory Cycle — multi-session LLM memory lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from clsplusplus.config import Settings
from clsplusplus.memory_cycle import run_memory_cycle
from clsplusplus.models import MemoryCycleRequest, MemoryItem, ReadResponse, StoreLevel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_memory_svc(mock_memory_service):
    """Uses conftest's mock_memory_service."""
    return mock_memory_service


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
    )


# ============================================================================
# Model Validation
# ============================================================================


class TestMemoryCycleRequest:
    """Tests for MemoryCycleRequest model."""

    def test_valid_request(self):
        req = MemoryCycleRequest(
            statements=["My name is Alice"],
            queries=["What is my name?"],
        )
        assert req.models == ["claude", "openai"]
        assert req.namespace == "cycle-test"

    def test_custom_models(self):
        req = MemoryCycleRequest(
            statements=["test"],
            queries=["query"],
            models=["claude"],
        )
        assert req.models == ["claude"]

    def test_empty_statements_rejected(self):
        with pytest.raises(Exception):
            MemoryCycleRequest(statements=[], queries=["test"])

    def test_empty_queries_rejected(self):
        with pytest.raises(Exception):
            MemoryCycleRequest(statements=["test"], queries=[])

    def test_invalid_namespace_rejected(self):
        with pytest.raises(Exception):
            MemoryCycleRequest(
                statements=["test"],
                queries=["test"],
                namespace="bad namespace!",
            )


# ============================================================================
# Memory Cycle Execution
# ============================================================================


class TestMemoryCycleExecution:
    """Tests for the full memory cycle flow."""

    @pytest.mark.asyncio
    async def test_encode_phase(self, mock_memory_svc, settings):
        """Statements are written as memories."""
        statements = ["My name is Alice", "I prefer dark mode"]
        queries = ["What is my name?"]

        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "Your name is Alice."

            result = await run_memory_cycle(
                mock_memory_svc, settings,
                statements=statements,
                queries=queries,
                models=["claude"],
                namespace="test-cycle",
            )

        assert result["phases"]["encode"]["stored"] == 2
        assert result["phases"]["encode"]["total"] == 2
        assert len(result["phases"]["encode"]["items"]) == 2

    @pytest.mark.asyncio
    async def test_retrieve_phase(self, mock_memory_svc, settings):
        """Queries return stored memories."""
        statements = ["My name is Alice"]
        queries = ["What is my name?"]

        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "Your name is Alice."

            result = await run_memory_cycle(
                mock_memory_svc, settings,
                statements=statements,
                queries=queries,
                models=["claude"],
                namespace="test-cycle-2",
            )

        retrieve = result["phases"]["retrieve"]
        assert retrieve["queries"] == 1
        assert retrieve["total_found"] > 0

    @pytest.mark.asyncio
    async def test_augment_phase_calls_llm(self, mock_memory_svc, settings):
        """LLM is called with memory-augmented prompt."""
        statements = ["My favorite color is blue"]
        queries = ["What is my favorite color?"]

        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "Your favorite color is blue."

            result = await run_memory_cycle(
                mock_memory_svc, settings,
                statements=statements,
                queries=queries,
                models=["claude"],
                namespace="test-cycle-3",
            )

        augment = result["phases"]["augment"]
        assert "claude" in augment
        assert len(augment["claude"]) > 0
        assert augment["claude"][0]["response"] == "Your favorite color is blue."
        mock_claude.assert_called()

    @pytest.mark.asyncio
    async def test_multi_model_augment(self, mock_memory_svc, settings):
        """Multiple models each get the same memory context."""
        statements = ["Project name is Atlas"]
        queries = ["What project?"]

        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_claude, \
             patch("clsplusplus.demo_llm_calls.call_openai", new_callable=AsyncMock) as mock_openai:
            mock_claude.return_value = "The project is Atlas."
            mock_openai.return_value = "It's called Atlas."

            result = await run_memory_cycle(
                mock_memory_svc, settings,
                statements=statements,
                queries=queries,
                models=["claude", "openai"],
                namespace="test-cycle-4",
            )

        augment = result["phases"]["augment"]
        assert "claude" in augment
        assert "openai" in augment
        assert augment["claude"][0]["memory_used"] is True
        assert augment["openai"][0]["memory_used"] is True

    @pytest.mark.asyncio
    async def test_cross_session_persistence(self, mock_memory_svc, settings):
        """Memories persist and can be read back."""
        statements = ["User prefers dark mode"]
        queries = ["preferences"]

        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "Dark mode."

            result = await run_memory_cycle(
                mock_memory_svc, settings,
                statements=statements,
                queries=queries,
                models=["claude"],
                namespace="test-cycle-5",
            )

        cross = result["phases"]["cross_session"]
        assert cross["memories_persisted"] is True
        assert cross["items_found"] > 0

    @pytest.mark.asyncio
    async def test_verdict_pass(self, mock_memory_svc, settings):
        """Full cycle produces PASS verdict."""
        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "Yes, I know."

            result = await run_memory_cycle(
                mock_memory_svc, settings,
                statements=["fact one"],
                queries=["query one"],
                models=["claude"],
                namespace="test-cycle-6",
            )

        assert result["verdict"] == "PASS"
        assert "cycle_id" in result
        assert result["namespace"] == "test-cycle-6"

    @pytest.mark.asyncio
    async def test_encode_error_handling(self, settings):
        """Encode phase handles write errors gracefully."""
        mock_svc = MagicMock()
        mock_svc.write = AsyncMock(side_effect=Exception("DB error"))
        mock_svc.read = AsyncMock(return_value=ReadResponse(items=[], query="q", namespace="ns"))

        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "response"

            result = await run_memory_cycle(
                mock_svc, settings,
                statements=["test"],
                queries=["query"],
                models=["claude"],
                namespace="test-err",
            )

        assert result["phases"]["encode"]["stored"] == 0
        assert "error" in result["phases"]["encode"]["items"][0]

    @pytest.mark.asyncio
    async def test_cycle_returns_models(self, mock_memory_svc, settings):
        """Result includes the models used."""
        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock:
            mock.return_value = "ok"

            result = await run_memory_cycle(
                mock_memory_svc, settings,
                statements=["test"],
                queries=["test"],
                models=["claude"],
                namespace="test-cycle-7",
            )

        assert result["models"] == ["claude"]

    @pytest.mark.asyncio
    async def test_gemini_model(self, mock_memory_svc, settings):
        """Gemini model can be used in cycle."""
        with patch("clsplusplus.demo_llm_calls.call_gemini", new_callable=AsyncMock) as mock_gemini:
            mock_gemini.return_value = "Gemini says hello."

            result = await run_memory_cycle(
                mock_memory_svc, settings,
                statements=["hello"],
                queries=["greet me"],
                models=["gemini"],
                namespace="test-cycle-8",
            )

        assert "gemini" in result["phases"]["augment"]
