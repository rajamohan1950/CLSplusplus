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


@pytest.mark.asyncio
async def test_install_macos_route_exists(proto_client: AsyncClient):
    """Browser install uses GET /install/macos; 404 is OK when downloads/ has no zip yet."""
    r = await proto_client.get("/install/macos")
    assert r.status_code in (200, 404)
    if r.status_code == 404:
        data = r.json()
        assert data.get("error") == "no_installer"
    else:
        ct = (r.headers.get("content-type") or "").lower()
        assert "zip" in ct or "octet-stream" in ct


@pytest.mark.asyncio
async def test_install_macos_apply_and_status(proto_client: AsyncClient):
    """One-click POST exists; outcome depends on OS, loopback, zip, CLS_E2E."""
    st = await proto_client.get("/install/macos/status")
    assert st.status_code == 200
    assert "phase" in st.json()

    r = await proto_client.post("/install/macos/apply")
    assert r.status_code in (200, 403, 404, 409, 503)


def test_workspace_repo_detection():
    """Full repo layout (src/clsplusplus) enables one-click install without a zip."""
    _proto_app()
    import server as proto_server  # noqa: PLC0415

    root = proto_server._workspace_repo_with_engine()
    assert root is not None
    assert (Path(root) / "src" / "clsplusplus" / "memory_phase.py").is_file()
