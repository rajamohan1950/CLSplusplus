"""
End-to-end: prototype memory server — store → /api/context → mock LLM echo contract.
No browser; validates the pipeline the Chrome extension relies on.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


def _proto_app():
    root = Path(__file__).resolve().parents[1]
    proto = str(root / "prototype")
    src = str(root / "src")
    for p in (proto, src):
        if p not in sys.path:
            sys.path.insert(0, p)
    import server as proto_server  # noqa: PLC0415

    return proto_server.app


@pytest.fixture
async def proto_client():
    app = _proto_app()
    # httpx<0.28: ASGITransport has no lifespan= kwarg; app has no lifespan hooks.
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _e2e_uid() -> str:
    # /api/uid is the machine-local profile; tests need isolated namespaces.
    return f"e2e_{uuid.uuid4().hex[:16]}"


@pytest.mark.asyncio
async def test_e2e_store_then_context_codename(proto_client: AsyncClient):
    uid = _e2e_uid()

    st = await proto_client.post(
        f"/api/store/{uid}",
        json={"text": "My secret codename is BLUEBADGER", "source": "user", "model": "e2e"},
    )
    assert st.status_code == 200

    ctx = await proto_client.post(
        "/api/context",
        json={"uid": uid, "query": "What is my secret codename?"},
    )
    assert ctx.status_code == 200
    data = ctx.json()
    assert data.get("count", 0) >= 1
    assert "BLUEBADGER" in (data.get("context") or "")
    await proto_client.delete(f"/api/memories/{uid}")


@pytest.mark.asyncio
async def test_e2e_store_then_context_city(proto_client: AsyncClient):
    uid = _e2e_uid()

    await proto_client.post(
        f"/api/store/{uid}",
        json={"text": "I live in Springfield", "source": "user", "model": "e2e"},
    )

    ctx = await proto_client.post(
        "/api/context",
        json={"uid": uid, "query": "What city do I live in?"},
    )
    assert ctx.status_code == 200
    assert "Springfield" in (ctx.json().get("context") or "")
    await proto_client.delete(f"/api/memories/{uid}")


@pytest.mark.asyncio
async def test_e2e_mock_chatgpt_echo_injection_contract(proto_client: AsyncClient):
    """Same shape as post-intercept body; mock must report injection_ok."""
    body = {
        "messages": [
            {
                "id": "x",
                "role": "user",
                "content": {
                    "content_type": "text",
                    "parts": [
                        "The user has shared the following about themselves in previous conversations. "
                        "Use this as background context — do not repeat it back unless asked:\n"
                        "- My secret codename is BLUEBADGER\n\n"
                        "What is my secret codename?"
                    ],
                },
            }
        ]
    }
    r = await proto_client.post("/backend-api/conversation", json=body)
    assert r.status_code == 200
    assert r.json().get("injection_ok") is True


@pytest.mark.asyncio
async def test_e2e_mock_claude_echo_injection_contract(proto_client: AsyncClient):
    prompt = (
        "The user has shared the following about themselves in previous conversations.\n"
        "- I live in Springfield\n\nWhat city do I live in?"
    )
    r = await proto_client.post(
        "/api/e2e/chat_conversations/local/completion",
        json={"prompt": prompt},
    )
    assert r.status_code == 200
    assert r.json().get("injection_ok") is True
