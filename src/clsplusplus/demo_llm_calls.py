"""LLM API calls - no Redis/Postgres. Used by demo_local and demo_llm."""
import asyncio
import logging

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)


async def call_claude(settings: Settings, system: str, user: str) -> str:
    import time as _time

    def _sync():
        import anthropic
        key = getattr(settings, "anthropic_api_key", None) or ""
        if not key:
            return "Claude: Add CLS_ANTHROPIC_API_KEY to env."
        client = anthropic.Anthropic(api_key=key)
        # Retry on transient 529/overloaded errors
        for attempt in range(6):
            try:
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=512,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            except Exception as e:
                err_str = str(e)
                if "529" in err_str or "verloaded" in err_str:
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning("Claude 529 overloaded, retry in %ds (attempt %d)", wait, attempt + 1)
                    _time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Claude API overloaded after 6 retries")

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.error("Claude call failed: %s", e)
        err = str(e).lower()
        if "credit balance" in err or "billing" in err or "quota" in err:
            return "Claude: API credits exhausted — add billing at console.anthropic.com."
        return "Claude: API error — check server logs."


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
        err = str(e).lower()
        if "insufficient_quota" in err or "billing" in err or "credit" in err:
            return "OpenAI: API quota exceeded — add billing at platform.openai.com."
        return "OpenAI: API error — check server logs."


async def call_gemini(settings: Settings, system: str, user: str) -> str:
    def _sync():
        import google.generativeai as genai
        key = getattr(settings, "google_api_key", None) or ""
        if not key:
            return "Gemini: Add CLS_GOOGLE_API_KEY to env."
        genai.configure(api_key=key)
        # Pass the CLS++ memory-augmented system prompt as a proper
        # system_instruction so Gemini treats it as grounding context on
        # every turn. Previously we concatenated system + user into a
        # single string, which Gemini would sometimes treat as a fresh
        # prompt and ignore the memory context baked into `system`.
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=system,
        )
        resp = model.generate_content(user)
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
        err = str(e).lower()
        if "quota" in err or "billing" in err or "resource exhausted" in err:
            return "Gemini: API quota exceeded — check billing at console.cloud.google.com."
        return "Gemini: API error — check server logs."
