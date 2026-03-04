"""Demo LLM integration - Claude, OpenAI, Gemini with shared CLS++ memory.

Flow: GET memory -> augment prompt -> LLM call -> PUT memory -> return LLM response.
Response is always from the real LLM; memory augments input and stores facts.
"""

from clsplusplus.config import Settings
from clsplusplus.memory_service import MemoryService
from clsplusplus.models import ReadRequest, WriteRequest


def _is_question(text: str) -> bool:
    t = text.strip().lower()
    return "?" in t or any(t.startswith(w) for w in ("what", "who", "where", "when", "how", "which", "is my", "do you"))


async def chat_with_llm(
    memory_service: MemoryService,
    settings: Settings,
    model: str,
    message: str,
    namespace: str,
) -> str:
    """
    1. PUT memory - store user statement (if not question)
    2. GET memory - read relevant context from CLS++
    3. Augment - add memory to system prompt
    4. LLM call - get real response from Claude/OpenAI/Gemini (always real LLM)
    5. Return - LLM response (never from memory; memory only augments input)
    """
    memory_context = "No prior context yet."
    try:
        # 1. PUT memory - store user statement first (so GET includes it)
        if not _is_question(message):
            await memory_service.write(
                WriteRequest(text=message, namespace=namespace, source="demo", salience=0.8)
            )
        # 2. GET memory
        read_resp = await memory_service.read(
            ReadRequest(query=message, namespace=namespace, limit=8)
        )
        if read_resp.items:
            memory_context = "\n".join(f"- {item.text}" for item in read_resp.items)
    except Exception:
        pass

    # 3. Augment prompt with memory
    system_prompt = f"""You are a friendly, helpful assistant. Chat naturally.
When the user asks a question, use this context if relevant:
{memory_context}
Respond naturally. Answer comes from you, not from repeating context verbatim."""

    # 4. LLM call - response is always from the real LLM
    from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini
    if model == "claude":
        return await call_claude(settings, system_prompt, message)
    if model == "openai":
        return await call_openai(settings, system_prompt, message)
    if model == "gemini":
        return await call_gemini(settings, system_prompt, message)
    return "Unknown model: " + model
