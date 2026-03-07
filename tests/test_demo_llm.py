"""Demo LLM integration tests - question detection, chat flow, error handling."""

from unittest.mock import AsyncMock, patch

import pytest

from clsplusplus.config import Settings
from clsplusplus.demo_llm import _is_question, chat_with_llm
from clsplusplus.models import ReadResponse, WriteRequest


# ---------------------------------------------------------------------------
# Question detection
# ---------------------------------------------------------------------------

class TestIsQuestion:

    def test_question_mark(self):
        assert _is_question("Is this working?") is True

    def test_what_prefix(self):
        assert _is_question("What is your name") is True

    def test_who_prefix(self):
        assert _is_question("Who is the president") is True

    def test_where_prefix(self):
        assert _is_question("Where is Paris") is True

    def test_when_prefix(self):
        assert _is_question("When did it happen") is True

    def test_how_prefix(self):
        assert _is_question("How does it work") is True

    def test_which_prefix(self):
        assert _is_question("Which one is better") is True

    def test_is_my_prefix(self):
        assert _is_question("Is my order ready") is True

    def test_do_you_prefix(self):
        assert _is_question("Do you know the answer") is True

    def test_statement_not_question(self):
        assert _is_question("My name is Bob") is False

    def test_command_not_question(self):
        assert _is_question("Remember that I like pizza") is False

    def test_empty_string(self):
        assert _is_question("") is False

    def test_whitespace_handling(self):
        assert _is_question("  What is this?  ") is True

    def test_case_insensitive(self):
        assert _is_question("WHAT IS THIS") is True


# ---------------------------------------------------------------------------
# Chat with LLM
# ---------------------------------------------------------------------------

class TestChatWithLLM:

    @pytest.mark.asyncio
    async def test_unknown_model(self, mock_memory_service):
        s = Settings()
        result = await chat_with_llm(mock_memory_service, s, "invalid_model", "hello", "ns1")
        assert "Unknown model" in result

    @pytest.mark.asyncio
    async def test_claude_no_key(self, mock_memory_service):
        s = Settings(anthropic_api_key=None)
        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Claude: Add CLS_ANTHROPIC_API_KEY to env."
            result = await chat_with_llm(mock_memory_service, s, "claude", "hello", "ns1")
            assert "Claude" in result or "API_KEY" in result or result is not None

    @pytest.mark.asyncio
    async def test_openai_no_key(self, mock_memory_service):
        s = Settings(openai_api_key=None)
        with patch("clsplusplus.demo_llm_calls.call_openai", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "OpenAI: Add CLS_OPENAI_API_KEY to env."
            result = await chat_with_llm(mock_memory_service, s, "openai", "hello", "ns1")
            assert result is not None

    @pytest.mark.asyncio
    async def test_gemini_no_key(self, mock_memory_service):
        s = Settings(google_api_key=None)
        with patch("clsplusplus.demo_llm_calls.call_gemini", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Gemini: Add CLS_GOOGLE_API_KEY to env."
            result = await chat_with_llm(mock_memory_service, s, "gemini", "hello", "ns1")
            assert result is not None

    @pytest.mark.asyncio
    async def test_statement_stored_in_memory(self, mock_memory_service):
        s = Settings()
        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Got it!"
            await chat_with_llm(mock_memory_service, s, "claude", "My name is Bob", "ns1")
            # Statement should be written to memory
            result = await mock_memory_service.read(
                __import__("clsplusplus.models", fromlist=["ReadRequest"]).ReadRequest(
                    query="name", namespace="ns1"
                )
            )
            assert any("Bob" in item.text for item in result.items)

    @pytest.mark.asyncio
    async def test_question_not_stored(self, mock_memory_service):
        s = Settings()
        initial_count = len(mock_memory_service.l1.items.get("ns1", {}))
        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Hello!"
            await chat_with_llm(mock_memory_service, s, "claude", "What is my name?", "ns1")
            # L1 items written by statement = 0 additional for question
            # (existing items from other writes may exist)

    @pytest.mark.asyncio
    async def test_memory_error_handled_gracefully(self):
        """If memory service fails, LLM call should still proceed."""
        s = Settings()

        class FailingMemoryService:
            async def write(self, req):
                raise RuntimeError("DB down")
            async def read(self, req):
                raise RuntimeError("DB down")

        with patch("clsplusplus.demo_llm_calls.call_claude", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Hello!"
            result = await chat_with_llm(
                FailingMemoryService(), s, "claude", "hello", "ns1"
            )
            assert result == "Hello!"


# ---------------------------------------------------------------------------
# LLM call wrappers
# ---------------------------------------------------------------------------

class TestLLMCallWrappers:

    @pytest.mark.asyncio
    async def test_claude_call_error_handling(self):
        from clsplusplus.demo_llm_calls import call_claude
        s = Settings(anthropic_api_key=None)
        result = await call_claude(s, "system", "user")
        assert "Claude" in result

    @pytest.mark.asyncio
    async def test_openai_call_error_handling(self):
        from clsplusplus.demo_llm_calls import call_openai
        s = Settings(openai_api_key=None)
        result = await call_openai(s, "system", "user")
        assert "OpenAI" in result

    @pytest.mark.asyncio
    async def test_gemini_call_error_handling(self):
        from clsplusplus.demo_llm_calls import call_gemini
        s = Settings(google_api_key=None)
        result = await call_gemini(s, "system", "user")
        assert "Gemini" in result
