#!/usr/bin/env python3
"""
Test demo LLM integration locally. Requires API keys in env:
  CLS_ANTHROPIC_API_KEY, CLS_OPENAI_API_KEY, CLS_GOOGLE_API_KEY
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clsplusplus.config import Settings
from clsplusplus.demo_llm import chat_with_llm


class MockMemoryService:
    """Minimal mock - no Redis/Postgres needed."""

    async def write(self, req):
        pass

    async def read(self, req):
        from clsplusplus.models import ReadResponse
        return ReadResponse(items=[], query=req.query, namespace=req.namespace)


async def main():
    settings = Settings()
    memory = MockMemoryService()
    msg = "Say hello in 5 words."

    for model in ["claude", "openai", "gemini"]:
        try:
            reply = await chat_with_llm(memory, settings, model, msg, "test")
            if "Add " in reply and "env" in reply:
                print(f"  {model}: SKIP (no API key)")
            elif "error" in reply.lower():
                print(f"  {model}: FAIL - {reply[:100]}")
            else:
                print(f"  {model}: OK - {reply[:60]}...")
        except Exception as e:
            print(f"  {model}: EXCEPTION - {e}")


if __name__ == "__main__":
    asyncio.run(main())
