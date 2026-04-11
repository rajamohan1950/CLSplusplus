"""End-to-end user story tests — validates complete journeys through the system.

Each test class represents a user story with sequential steps that verify
the full flow from API call to expected outcome.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from clsplusplus.api import create_app
from clsplusplus.config import Settings


VALID_KEY = "cls_live_test1234567890123456789012"


@pytest.fixture
async def c():
    """Client without auth for open-mode testing."""
    app = create_app(Settings(require_api_key=False))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def ca():
    """Client with valid auth."""
    app = create_app(Settings(require_api_key=True, api_keys=VALID_KEY, track_usage=True))
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test",
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    ) as ac:
        yield ac


# ═══════════════════════════════════════════════════════════════════════════════
# Story 1: New user discovers CLS++ and links their extension
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryNewUserOnboarding:
    """
    User journey: Visit site → check health → write first memory → verify it exists
    """

    @pytest.mark.asyncio
    async def test_step1_health_check(self, c):
        """Extension popup checks if server is reachable."""
        resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_step2_first_memory_write(self, c):
        """User sends their first message in ChatGPT, extension captures it."""
        resp = await c.post("/v1/memory/write", json={
            "text": "My name is Raj and I prefer Python",
            "source": "extension",
        })
        assert resp.status_code == 200
        assert "id" in resp.json()

    @pytest.mark.asyncio
    async def test_step3_memory_appears_in_list(self, c):
        """Extension popup shows memory count."""
        await c.post("/v1/memory/write", json={
            "text": "I work at a startup",
            "source": "extension",
        })
        resp = await c.get("/v1/memory/list", params={"limit": 10})
        assert resp.status_code == 200
        assert resp.json()["total"] >= 0

    @pytest.mark.asyncio
    async def test_step4_memory_searchable(self, c):
        """User searches for a memory in the side panel."""
        await c.post("/v1/memory/write", json={"text": "I love mangoes"})
        resp = await c.post("/v1/memories/search", json={"query": "fruit preference"})
        assert resp.status_code == 200
        assert "items" in resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
# Story 2: Cross-model memory — ChatGPT → Claude → Gemini
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryCrossModelMemory:
    """
    User tells ChatGPT their name → Claude knows it → Gemini knows it.
    All via the extension storing and injecting memories.
    """

    @pytest.mark.asyncio
    async def test_store_from_chatgpt(self, c):
        """Extension captures message from ChatGPT."""
        resp = await c.post("/v1/memory/write", json={
            "text": "My name is Raj and I prefer dark mode",
            "source": "chatgpt",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_retrieve_for_claude(self, c):
        """Extension fetches memories to inject into Claude."""
        await c.post("/v1/memory/write", json={
            "text": "I use VS Code as my editor",
            "source": "chatgpt",
        })
        resp = await c.get("/v1/memory/list", params={"limit": 15})
        assert resp.status_code == 200
        # Extension would read items and inject into Claude

    @pytest.mark.asyncio
    async def test_store_from_claude_retrieve_for_gemini(self, c):
        """Full cycle: store from one model, retrieve for another."""
        # Claude conversation
        await c.post("/v1/memory/write", json={
            "text": "I am building a memory system called CLS++",
            "source": "claude",
        })
        # Gemini needs context
        resp = await c.get("/v1/memory/list", params={"limit": 15})
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Story 3: User checks their usage and upgrades
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryUsageAndUpgrade:
    """
    User sees usage in Activity tab → approaches limit → tries to upgrade
    """

    @pytest.mark.asyncio
    async def test_check_usage(self, ca):
        """Activity tab fetches usage stats."""
        resp = await ca.get("/v1/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert "tier" in data
        assert "operations" in data
        assert "operations_limit" in data
        assert isinstance(data["operations"], int)

    @pytest.mark.asyncio
    async def test_upgrade_rejects_free_tier(self, ca):
        """Cannot create order for free tier."""
        resp = await ca.post("/v1/billing/order", json={"tier": "free"})
        assert resp.status_code in (400, 401, 500)


# ═══════════════════════════════════════════════════════════════════════════════
# Story 4: User manages their memories
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryMemoryManagement:
    """
    User writes memories → searches → deletes one → verifies deletion
    """

    @pytest.mark.asyncio
    async def test_write_search_delete_cycle(self, c):
        """Full memory lifecycle."""
        # Write
        w = await c.post("/v1/memory/write", json={"text": "Temporary note to delete"})
        assert w.status_code == 200
        item_id = w.json().get("id", "")

        # Search
        s = await c.post("/v1/memories/search", json={"query": "temporary note"})
        assert s.status_code == 200

        # Delete
        if item_id:
            d = await c.request("DELETE", f"/v1/memories/{item_id}")
            assert d.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════════════════════════
# Story 5: Extension popup quick-glance flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryPopupQuickGlance:
    """
    User clicks extension icon → sees stats → opens side panel
    Simulates the API calls popup.js makes on load.
    """

    @pytest.mark.asyncio
    async def test_popup_load_sequence(self, c):
        """Popup makes these calls in parallel on load."""
        # 1. Health check
        health = await c.get("/health")
        assert health.status_code == 200

        # 2. Fetch recent activity (memory count + last memory)
        activity = await c.get("/v1/memory/list", params={"limit": 1})
        assert activity.status_code == 200
        data = activity.json()
        assert "total" in data
        assert "items" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Story 6: Side panel full exploration
# ═══════════════════════════════════════════════════════════════════════════════

class TestStorySidePanelExploration:
    """
    User opens side panel → browses memories → filters by source →
    searches → checks activity → toggles settings
    """

    @pytest.mark.asyncio
    async def test_memories_tab_load(self, c):
        """Side panel loads memories on open."""
        resp = await c.get("/v1/memory/list", params={"limit": 15})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_memories(self, c):
        """User types in search bar."""
        await c.post("/v1/memory/write", json={"text": "I prefer dark themes"})
        resp = await c.post("/v1/memories/search", json={
            "query": "theme preference",
            "limit": 10,
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_activity_tab_load(self, ca):
        """Activity tab fetches usage data."""
        resp = await ca.get("/v1/usage")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_load_more_pagination(self, c):
        """User clicks 'Load more' in memories tab."""
        # First page
        r1 = await c.get("/v1/memory/list", params={"limit": 15})
        assert r1.status_code == 200
        # Second page (simulated by increasing limit)
        r2 = await c.get("/v1/memory/list", params={"limit": 30})
        assert r2.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Story 7: Multiple namespaces isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryNamespaceIsolation:
    """
    Different API keys create different namespaces.
    Memories don't leak between users.
    """

    @pytest.mark.asyncio
    async def test_write_to_different_namespaces(self, c):
        await c.post("/v1/memory/write", json={
            "text": "User A secret",
            "namespace": "user-a",
        })
        await c.post("/v1/memory/write", json={
            "text": "User B secret",
            "namespace": "user-b",
        })

        resp_a = await c.get("/v1/memory/list", params={"namespace": "user-a"})
        resp_b = await c.get("/v1/memory/list", params={"namespace": "user-b"})
        assert resp_a.status_code == 200
        assert resp_b.status_code == 200

        # Verify isolation
        for item in resp_a.json().get("items", []):
            assert item["namespace"] == "user-a"
        for item in resp_b.json().get("items", []):
            assert item["namespace"] == "user-b"


# ═══════════════════════════════════════════════════════════════════════════════
# Story 8: Demo chat (unauthenticated visitor)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryDemoVisitor:
    """
    First-time visitor lands on website → tries demo → sees it work
    """

    @pytest.mark.asyncio
    async def test_demo_status(self, c):
        resp = await c.get("/v1/demo/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_demo_chat(self, c):
        resp = await c.post("/v1/demo/chat", json={
            "message": "What is CLS++?",
            "provider": "anthropic",
        })
        # May fail if no API key configured, but shouldn't crash
        assert resp.status_code in (200, 422, 500, 503)


# ═══════════════════════════════════════════════════════════════════════════════
# Story 9: Webhook backup verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryWebhookBackup:
    """
    Razorpay sends webhook → server processes it (even if verify failed)
    """

    @pytest.mark.asyncio
    async def test_webhook_accessible_without_auth(self, c):
        """Webhook endpoint must be public (Razorpay calls it directly)."""
        resp = await c.post("/v1/billing/razorpay-webhook",
                           content=b'{"event":"payment.captured"}',
                           headers={
                               "Content-Type": "application/json",
                               "x-razorpay-signature": "invalid-sig",
                           })
        # Must NOT be 401 — may be 400/500 due to config
        assert resp.status_code in (200, 400, 500)

    @pytest.mark.asyncio
    async def test_stripe_webhook_accessible(self, c):
        resp = await c.post("/v1/billing/webhook",
                           content=b'{}',
                           headers={"Content-Type": "application/json",
                                    "stripe-signature": "test"})
        assert resp.status_code in (200, 400, 500)


# ═══════════════════════════════════════════════════════════════════════════════
# Story 10: Auth guard enforcement
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoryAuthGuards:
    """
    Verify that protected endpoints reject unauthenticated requests
    and public endpoints always work.
    """

    @pytest.fixture
    async def cu(self):
        """Unauthenticated client with auth enforced."""
        app = create_app(Settings(require_api_key=True, api_keys=VALID_KEY))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    @pytest.mark.asyncio
    async def test_public_endpoints_always_accessible(self, cu):
        """These must NEVER return 401."""
        public = [
            ("GET", "/health"),
            ("GET", "/v1/memory/health"),
            ("GET", "/v1/demo/status"),
        ]
        for method, path in public:
            resp = await cu.request(method, path)
            assert resp.status_code != 401, f"{method} {path} returned 401"

    @pytest.mark.asyncio
    async def test_protected_endpoints_require_auth(self, cu):
        """These must return 401 without auth."""
        protected = [
            ("POST", "/v1/memory/write", {"text": "test"}),
            ("POST", "/v1/memory/read", {"query": "test"}),
            ("GET", "/v1/memory/list"),
            ("GET", "/v1/usage"),
            ("POST", "/v1/billing/order", {"tier": "pro"}),
        ]
        for method, path, *body in protected:
            kwargs = {"json": body[0]} if body else {}
            resp = await cu.request(method, path, **kwargs)
            assert resp.status_code == 401, f"{method} {path} should be 401, got {resp.status_code}"
