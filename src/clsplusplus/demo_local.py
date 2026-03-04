"""
Standalone demo API for local testing - NO Redis, NO Postgres.
Real Claude, OpenAI, Gemini only. Requires API keys in .env.
Run: uvicorn clsplusplus.demo_local:app --reload --port 8080
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from clsplusplus.config import Settings

app = FastAPI(title="CLS++ Demo (Local)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory context per namespace (replaces Redis/Postgres)
_memory: dict[str, list[str]] = {}


class ChatRequest(BaseModel):
    model: str
    message: str
    namespace: str = "demo"


def _get_memory_context(namespace: str, query: str) -> str:
    items = _memory.get(namespace, [])
    return "\n".join(f"- {t}" for t in items) if items else "No prior context yet."


def _store_if_statement(namespace: str, message: str) -> None:
    is_q = "?" in message or any(
        message.strip().lower().startswith(w)
        for w in ("what", "who", "where", "when", "how", "which", "is my", "do you")
    )
    if not is_q:
        _memory.setdefault(namespace, []).append(message)


@app.get("/v1/demo/status")
async def status():
    s = Settings()
    return {
        "claude": bool(getattr(s, "anthropic_api_key", None)),
        "openai": bool(getattr(s, "openai_api_key", None)),
        "gemini": bool(getattr(s, "google_api_key", None)),
    }


@app.post("/v1/demo/chat")
async def chat(req: ChatRequest):
    if req.model not in ("claude", "openai", "gemini"):
        return {"error": "model must be claude, openai, or gemini"}
    if not req.message.strip():
        return {"error": "message required"}

    settings = Settings()
    # PUT memory (store statement)
    _store_if_statement(req.namespace, req.message.strip())
    # GET memory (read context)
    memory_context = _get_memory_context(req.namespace, req.message)
    # Augment prompt
    system = f"""You are a friendly, helpful assistant. Chat naturally — respond like a normal conversation.
When the user tells you something, reply naturally and engage. When they ask a question, use this context if relevant:
{memory_context}
Respond naturally. Don't mention "memory" or "context" — just talk like a normal assistant."""

    # LLM call - response is always from real LLM
    from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini
    if req.model == "claude":
        reply = await call_claude(settings, system, req.message.strip())
    elif req.model == "openai":
        reply = await call_openai(settings, system, req.message.strip())
    else:
        reply = await call_gemini(settings, system, req.message.strip())

    return {"model": req.model, "reply": reply}
