"""LLM API calls - no Redis/Postgres. Used by demo_local and demo_llm."""
import asyncio
import logging

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)


async def call_claude(settings: Settings, system: str, user: str) -> str:
    def _sync():
        import anthropic
        key = getattr(settings, "anthropic_api_key", None) or ""
        if not key:
            return "Claude: Add CLS_ANTHROPIC_API_KEY to env."
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.error("Claude call failed: %s", e)
        return "Claude: An error occurred processing your request. Check server logs."


async def call_openai(settings: Settings, system: str, user: str) -> str:
    def _sync():
        from openai import OpenAI
        key = getattr(settings, "openai_api_key", None) or ""
        if not key:
            return "OpenAI: Add CLS_OPENAI_API_KEY to env."
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        return "OpenAI: An error occurred processing your request. Check server logs."


async def call_gemini(settings: Settings, system: str, user: str) -> str:
    def _sync():
        import google.generativeai as genai
        key = getattr(settings, "google_api_key", None) or ""
        if not key:
            return "Gemini: Add CLS_GOOGLE_API_KEY to env."
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        full_prompt = f"{system}\n\nUser: {user}"
        resp = model.generate_content(full_prompt)
        try:
            return resp.text or "No response"
        except (ValueError, AttributeError):
            if resp.candidates and resp.candidates[0].content.parts:
                return resp.candidates[0].content.parts[0].text or "No response"
            return "Gemini: No response (content may have been blocked)."

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        return "Gemini: An error occurred processing your request. Check server logs."
