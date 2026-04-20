"""Demo LLM integration - Claude, OpenAI, Gemini with shared CLS++ memory.

Flow: GET memory -> augment prompt -> LLM call -> PUT memory -> return LLM response.
Response is always from the real LLM; memory augments input and stores facts.
"""

from contextlib import nullcontext
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.memory_service import MemoryService
from clsplusplus.models import ReadRequest, WriteRequest
from clsplusplus.tracer import tracer


def _is_question(text: str) -> bool:
    t = text.strip().lower()
    return "?" in t or any(t.startswith(w) for w in ("what", "who", "where", "when", "how", "which", "is my", "do you"))


def _span(trace_id, label, module, **meta):
    """Helper: returns a real span ctx if trace_id set, else nullcontext."""
    if trace_id:
        return tracer.span(trace_id, label, module, **meta)
    return nullcontext()


async def chat_with_llm(
    memory_service: MemoryService,
    settings: Settings,
    model: str,
    message: str,
    namespace: str,
    trace_id: Optional[str] = None,
) -> str:
    """Full LLM chat cycle — every step traced with input/output:
    1. user.write   — store what the user said (if statement, not question)
    2. memory.read  — retrieve relevant context from CLS++ memory
    3. llm.{model}  — call the real LLM with memory-augmented prompt
    4. reply.store  — store what the LLM replied (so future turns remember it)
    5. return reply to HTTP layer → user
    """
    memory_context = "No prior context yet."
    read_resp = None
    reply = "No response."

    try:
        # 1. Store user message into memory (skipped for questions — no new facts)
        if not _is_question(message):
            with _span(trace_id, "user.write", "memory_service",
                       input=message[:200], namespace=namespace) as hop:
                item = await memory_service.write(
                    WriteRequest(text=message, namespace=namespace,
                                 source="user", salience=0.8),
                    trace_id=trace_id,
                )
                if trace_id and hop:
                    tracer.add_metadata(trace_id, hop,
                                        output=f"stored item {str(item.id)[:8]}…  level={item.store_level.value}")

        # 2. Read relevant context from CLS++ memory
        with _span(trace_id, "memory.read", "memory_service",
                   input=message[:200], namespace=namespace) as hop:
            read_resp = await memory_service.read(
                ReadRequest(query=message, namespace=namespace, limit=8),
                trace_id=trace_id,
            )
            if trace_id and hop:
                items = read_resp.items or []
                if items:
                    preview = " | ".join(i.text[:60] for i in items[:3])
                    tracer.add_metadata(trace_id, hop,
                                        output=f"{len(items)} items: {preview}")
                else:
                    tracer.add_metadata(trace_id, hop, output="0 items — no context found")

        if read_resp and read_resp.items:
            memory_context = "\n".join(f"- {item.text}" for item in read_resp.items)
    except Exception:
        pass

    # 3. Augment prompt with memory context and call the real LLM
    system_prompt = (
        "You are a friendly, helpful assistant. Chat naturally.\n"
        "When the user asks a question, use this context if relevant:\n"
        f"{memory_context}\n"
        "Respond naturally. Answer comes from you, not from repeating context verbatim."
    )

    ctx_count = len(read_resp.items) if read_resp and read_resp.items else 0
    from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini
    with _span(trace_id, f"llm.{model}", "llm",
               input=message[:200],
               model=model,
               context_items=ctx_count,
               context_preview=(memory_context[:120] if ctx_count else "none")) as hop:
        if model == "claude":
            reply = await call_claude(settings, system_prompt, message)
        elif model == "openai":
            reply = await call_openai(settings, system_prompt, message)
        elif model == "gemini":
            reply = await call_gemini(settings, system_prompt, message)
        else:
            reply = "Unknown model: " + model

        if trace_id and hop:
            tracer.add_metadata(trace_id, hop, output=reply[:300])

    # 4. Store the LLM reply back into memory so future turns can reference it
    if reply and not reply.startswith(("Claude:", "OpenAI:", "Gemini:", "Unknown")):
        try:
            store_text = f"{model} replied: {reply[:400]}"
            with _span(trace_id, "reply.store", "memory_service",
                       input=store_text[:200],
                       model=model,
                       namespace=namespace) as hop:
                item = await memory_service.write(
                    WriteRequest(
                        text=store_text,
                        namespace=namespace,
                        source=f"assistant.{model}",
                        salience=0.6,
                    ),
                    trace_id=trace_id,
                )
                if trace_id and hop:
                    tracer.add_metadata(trace_id, hop,
                                        output=f"stored item {str(item.id)[:8]}…  level={item.store_level.value}")
        except Exception:
            pass

    # 5. Return reply → HTTP layer sends it to the user
    return reply
