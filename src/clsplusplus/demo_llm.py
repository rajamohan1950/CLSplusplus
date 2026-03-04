"""Demo LLM integration - Claude, OpenAI, Gemini with shared CLS++ memory."""

import asyncio
from clsplusplus.config import Settings
from clsplusplus.memory_service import MemoryService
from clsplusplus.models import ReadRequest, WriteRequest


async def chat_with_llm(
    memory_service: MemoryService,
    settings: Settings,
    model: str,
    message: str,
    namespace: str,
) -> str:
    """
    Handle user message: 1) store if statement, 2) read relevant memories from CLS++,
    3) call the appropriate LLM with memory context, 4) return response.
    """
    memory_context = "No prior context yet."

    try:
        # 1. Store facts/statements in memory (not questions)
        is_question = "?" in message or any(
            message.strip().lower().startswith(w)
            for w in ("what", "who", "where", "when", "how", "which", "is my", "do you")
        )
        if not is_question:
            await memory_service.write(
                WriteRequest(text=message, namespace=namespace, source="demo", salience=0.8)
            )

        # 2. Read relevant memories for context
        read_resp = await memory_service.read(
            ReadRequest(query=message, namespace=namespace, limit=8)
        )
        if read_resp.items:
            memory_context = "\n".join(f"- {item.text}" for item in read_resp.items)
    except Exception:
        pass  # Memory failed; proceed with empty context so LLM can still respond

    system_prompt = f"""You are a helpful assistant. You have access to shared memory (CLS++) that persists across different AI models (Claude, OpenAI, Gemini). Use this context to answer questions accurately.

Current memory context:
{memory_context}

Be concise. If the user shared a fact, acknowledge it briefly. If they asked a question, answer from the memory context above. If the memory doesn't contain the answer, say so politely."""

    # 3. Call the appropriate LLM
    if model == "claude":
        return await _call_claude(settings, system_prompt, message)
    if model == "openai":
        return await _call_openai(settings, system_prompt, message)
    if model == "gemini":
        return await _call_gemini(settings, system_prompt, message)
    return "Unknown model: " + model


async def _call_claude(settings: Settings, system: str, user: str) -> str:
    def _sync():
        import anthropic
        key = getattr(settings, "anthropic_api_key", None) or ""
        if not key:
            return "Claude: Add CLS_ANTHROPIC_API_KEY to env."
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        return f"Claude error: {str(e)[:120]}"


async def _call_openai(settings: Settings, system: str, user: str) -> str:
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
        return f"OpenAI error: {str(e)[:120]}"


async def _call_gemini(settings: Settings, system: str, user: str) -> str:
    def _sync():
        import google.generativeai as genai
        key = getattr(settings, "google_api_key", None) or ""
        if not key:
            return "Gemini: Add CLS_GOOGLE_API_KEY to env."
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
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
        return f"Gemini error: {str(e)[:120]}"
