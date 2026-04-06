"""CLS++ Python SDK — Memory that thinks like a brain.

Usage:

    from clsplusplus import Brain

    # Create a brain for a user — auto-connects, zero config
    brain = Brain("alice")

    # Teach it anything in natural language
    brain.learn("I work at Google as a senior engineer")
    brain.learn("I prefer Python over JavaScript")
    brain.learn("My favorite editor is VS Code")

    # Ask it anything — semantic recall, not keyword matching
    brain.ask("What's my job?")           # → "senior engineer at Google"
    brain.ask("coding preferences?")      # → ["I prefer Python...", "My favorite editor..."]

    # Get formatted context for any LLM prompt
    brain.context("Help me with a coding task")
    # → "User works at Google... Prefers Python... Uses VS Code..."

    # Forget (GDPR right to be forgotten)
    brain.forget("I work at Google as a senior engineer")

Environment variables (all optional):
    CLS_API_KEY   — API key for cloud/production
    CLS_BASE_URL  — Server URL (default: https://www.clsplusplus.com)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx


class Brain:
    """A persistent memory for one user. Learns, recalls, forgets — like a brain.

    Args:
        user: A unique user identifier (string). This is the "whose brain" key.
        api_key: Optional API key. Reads CLS_API_KEY env var if not provided.
        url: Optional server URL. Reads CLS_BASE_URL env var or defaults to localhost:8080.
    """

    def __init__(
        self,
        user: str,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.user = user
        self._url = (url or os.environ.get("CLS_BASE_URL", "https://www.clsplusplus.com")).rstrip("/")
        self._key = api_key or os.environ.get("CLS_API_KEY", "")
        self._http = httpx.Client(
            base_url=self._url,
            headers={"Authorization": f"Bearer {self._key}"} if self._key else {},
            timeout=30.0,
        )

    # ── Core API: learn / ask / context / forget ─────────────────────────

    def learn(self, fact: str, **meta) -> str:
        """Teach the brain a fact. Returns the memory ID.

            brain.learn("User prefers dark mode")
            brain.learn("Allergic to peanuts", source="medical")
        """
        body = {"text": fact, "namespace": self.user, "source": meta.pop("source", "user")}
        if meta:
            body["metadata"] = meta
        resp = self._http.post("/v1/memory/write", json=body)
        resp.raise_for_status()
        return resp.json().get("id", "")

    def ask(self, question: str, limit: int = 5) -> list[str]:
        """Ask the brain a question. Returns relevant facts as strings.

            brain.ask("What's my job?")
            # → ["I work at Google as a senior engineer"]
        """
        resp = self._http.post("/v1/memory/read", json={
            "query": question, "namespace": self.user, "limit": limit,
        })
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [item["text"] for item in items]

    def context(self, topic: str = "", limit: int = 8) -> str:
        """Get LLM-ready context string. Inject directly into your system prompt.

            prompt = brain.context("coding help")
            response = openai.chat(system=prompt, user=message)
        """
        query = topic or "everything about this user"
        facts = self.ask(query, limit=limit)
        if not facts:
            return ""
        lines = ["Known facts about this user:"]
        for f in facts:
            lines.append(f"- {f}")
        return "\n".join(lines)

    def forget(self, fact_or_id: str) -> bool:
        """Forget a specific memory. Pass the fact text or memory ID.

            brain.forget("Allergic to peanuts")
        """
        # Try as ID first
        resp = self._http.request("DELETE", "/v1/memory/forget", json={
            "item_id": fact_or_id, "namespace": self.user,
        })
        if resp.status_code == 200:
            return True
        # If not found by ID, search by text and delete the best match
        matches = self.ask(fact_or_id, limit=1)
        if matches:
            # Search again to get IDs
            resp2 = self._http.post("/v1/memory/read", json={
                "query": fact_or_id, "namespace": self.user, "limit": 1,
            })
            if resp2.status_code == 200:
                items = resp2.json().get("items", [])
                if items:
                    del_resp = self._http.request("DELETE", "/v1/memory/forget", json={
                        "item_id": items[0]["id"], "namespace": self.user,
                    })
                    return del_resp.status_code == 200
        return False

    # ═══════════════════════════════════════════════════════════════════════
    # INNOVATIVE INTEGRATIONS
    # ═══════════════════════════════════════════════════════════════════════

    # ── 1. LLM Wrapper — any function gets memory for free ───────────────

    def wrap(self, llm_fn):
        """Decorator: auto-inject memory context into any LLM call.

            @brain.wrap
            def chat(system_prompt, user_message):
                return openai.chat(system=system_prompt, user=user_message)

            # Now chat() auto-prepends user's memory to system_prompt
            # AND auto-learns from the conversation
            response = chat("You are a helpful assistant", "Help me with Python")
        """
        from functools import wraps

        @wraps(llm_fn)
        def wrapper(system_prompt: str, user_message: str, *args, **kwargs):
            # Inject memory into system prompt
            mem_context = self.context(user_message)
            augmented = (mem_context + "\n\n" + system_prompt) if mem_context else system_prompt
            # Call the LLM
            result = llm_fn(augmented, user_message, *args, **kwargs)
            # Auto-learn from the conversation
            self.learn(user_message, source="user")
            return result
        return wrapper

    # ── 2. Absorb — bulk-learn from conversation or document ─────────────

    def absorb(self, content, source: str = "document") -> int:
        """Bulk-learn from a conversation, document, or any text block.
        Splits on sentences/paragraphs and learns each one.

            # Learn from a chat transcript
            brain.absorb('''
            User: I'm a vegetarian
            User: I live in San Francisco
            User: I work remotely for Stripe
            ''')

            # Learn from a document
            with open("user_profile.txt") as f:
                brain.absorb(f.read())

            # Learn from a list
            brain.absorb(["Loves sushi", "Allergic to nuts", "Birthday is March 5"])
        """
        if isinstance(content, list):
            facts = content
        else:
            # Split on newlines, periods, or "User:" prefixes
            import re
            lines = re.split(r'\n|(?<=[.!?])\s+', str(content))
            facts = []
            for line in lines:
                line = re.sub(r'^(User|Assistant|Human|AI)\s*:\s*', '', line).strip()
                if len(line) > 8:  # Skip tiny fragments
                    facts.append(line)

        count = 0
        for fact in facts:
            try:
                self.learn(fact, source=source)
                count += 1
            except Exception:
                pass
        return count

    # ── 3. Who — auto-generated user profile ─────────────────────────────

    def who(self) -> dict:
        """Auto-generate a structured user profile from everything stored.

            profile = brain.who()
            # → {
            #     "facts": ["works at Google", "prefers Python", ...],
            #     "summary": "Senior engineer at Google who prefers Python and VS Code",
            #     "count": 5
            # }
        """
        facts = self.ask("everything about this user", limit=20)
        return {
            "user": self.user,
            "facts": facts,
            "summary": "; ".join(facts[:5]) if facts else "No information stored",
            "count": len(facts),
        }

    # ── 4. Correct — smart belief update ─────────────────────────────────

    def correct(self, wrong: str, right: str) -> str:
        """Update a belief. Forgets the old fact, learns the new one.

            brain.learn("I work at Google")
            brain.correct("I work at Google", "I just moved to Microsoft")
        """
        self.forget(wrong)
        return self.learn(right, source="correction")

    # ── 5. Chat — full conversation handler ──────────────────────────────

    def chat(self, user_message: str, llm_fn=None, system: str = "You are a helpful assistant.") -> str:
        """Complete conversation handler: recall → inject → call LLM → learn.

            # With any LLM function
            def my_llm(prompt, message):
                return openai.chat(system=prompt, user=message).content

            response = brain.chat("What editor should I use?", llm_fn=my_llm)
            # Brain auto-recalls relevant memory, injects into prompt,
            # calls your LLM, learns from the exchange, returns response.

            # Without LLM (just returns what the brain knows)
            response = brain.chat("What do you know about me?")
        """
        # Build augmented prompt
        mem = self.context(user_message)
        augmented = (mem + "\n\n" + system) if mem else system

        if llm_fn:
            response = llm_fn(augmented, user_message)
            # Learn from the exchange
            self.learn(user_message, source="user")
            if isinstance(response, str) and len(response) > 10:
                self.learn(f"Assistant replied: {response[:300]}", source="assistant")
            return response
        else:
            # No LLM — return what the brain knows
            facts = self.ask(user_message)
            return "\n".join(facts) if facts else "I don't have any information about that yet."

    # ── 6. Teach from dict — structured data learning ────────────────────

    def teach(self, data: dict) -> int:
        """Learn from a dictionary of key-value pairs.

            brain.teach({
                "name": "Priya",
                "company": "Netflix",
                "role": "Data Scientist",
                "languages": ["Python", "R", "SQL"],
                "preferences": {"theme": "dark", "editor": "VS Code"}
            })
        """
        count = 0

        def _flatten(obj, prefix=""):
            nonlocal count
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(v, f"{prefix}{k}: " if prefix else f"{k}: ")
            elif isinstance(obj, list):
                val = ", ".join(str(x) for x in obj)
                self.learn(f"{prefix}{val}")
                count += 1
            else:
                self.learn(f"{prefix}{obj}")
                count += 1

        _flatten(data)
        return count

    # ── 7. Watch — auto-learn from a stream of messages ──────────────────

    def watch(self, messages: list[dict]) -> int:
        """Learn from OpenAI/Anthropic-style message list.
        Extracts user statements (skips questions).

            messages = [
                {"role": "user", "content": "I'm a Python developer at Stripe"},
                {"role": "assistant", "content": "Nice! How can I help?"},
                {"role": "user", "content": "I prefer functional programming"},
            ]
            brain.watch(messages)  # Learns 2 facts (skips assistant messages)
        """
        count = 0
        for msg in messages:
            if msg.get("role") == "user":
                text = msg.get("content", "").strip()
                if text and len(text) > 8 and "?" not in text:
                    self.learn(text, source="conversation")
                    count += 1
        return count

    # ── Convenience ──────────────────────────────────────────────────────

    def all(self, limit: int = 50) -> list[str]:
        """Get all memories for this user."""
        return self.ask("everything", limit=limit)

    def count(self) -> int:
        """How many things the brain remembers."""
        return len(self.all(limit=10000))

    def __repr__(self) -> str:
        return f"Brain('{self.user}')"

    def __del__(self):
        try:
            self._http.close()
        except Exception:
            pass


# ── Module-level convenience (for scripts and one-liners) ────────────────

_default_brains: dict[str, Brain] = {}


def _get_brain(user: str) -> Brain:
    if user not in _default_brains:
        _default_brains[user] = Brain(user)
    return _default_brains[user]


def learn(user: str, fact: str, **meta) -> str:
    """Module-level: learn a fact for a user.

        import clsplusplus as mem
        mem.learn("alice", "Prefers dark mode")
    """
    return _get_brain(user).learn(fact, **meta)


def ask(user: str, question: str, limit: int = 5) -> list[str]:
    """Module-level: ask a question about a user.

        import clsplusplus as mem
        mem.ask("alice", "What theme?")
    """
    return _get_brain(user).ask(question, limit)


def context(user: str, topic: str = "", limit: int = 8) -> str:
    """Module-level: get LLM context for a user.

        import clsplusplus as mem
        prompt = mem.context("alice", "coding help")
    """
    return _get_brain(user).context(topic, limit)


def forget(user: str, fact_or_id: str) -> bool:
    """Module-level: forget a fact.

        import clsplusplus as mem
        mem.forget("alice", "old fact")
    """
    return _get_brain(user).forget(fact_or_id)


# ── Backward compatibility ──────────────────────────────────────────────

class MemoriesClient:
    """Legacy wrapper. Use Brain instead."""
    def __init__(self, client):
        self._client = client
    def encode(self, content, agent_id=None, namespace="default", metadata=None):
        ns = agent_id or namespace
        return self._client.write(text=content, namespace=ns, metadata=metadata or {})
    def retrieve(self, query, agent_id=None, namespace="default", limit=10):
        ns = agent_id or namespace
        return self._client.read(query=query, namespace=ns, limit=limit)


class CLSClient:
    """Legacy client. Use Brain instead."""
    def __init__(self, base_url=None, api_key=None):
        self._url = (base_url or os.environ.get("CLS_BASE_URL", "https://www.clsplusplus.com")).rstrip("/")
        self._key = api_key or os.environ.get("CLS_API_KEY", "")
        self._client = httpx.Client(
            base_url=self._url,
            headers={"Authorization": f"Bearer {self._key}"} if self._key else {},
            timeout=30.0,
        )
        self.memories = MemoriesClient(self)

    def write(self, text, namespace="default", source="user", salience=0.5, authority=0.5, metadata=None):
        from clsplusplus.models import WriteRequest
        req = WriteRequest(text=text, namespace=namespace, source=source, salience=salience, authority=authority, metadata=metadata or {})
        resp = self._client.post("/v1/memory/write", json=req.model_dump())
        resp.raise_for_status()
        return resp.json()

    def read(self, query, namespace="default", limit=10, min_confidence=0.0):
        from clsplusplus.models import ReadRequest, ReadResponse
        req = ReadRequest(query=query, namespace=namespace, limit=limit, min_confidence=min_confidence)
        resp = self._client.post("/v1/memory/read", json=req.model_dump())
        resp.raise_for_status()
        return ReadResponse(**resp.json())

    def close(self):
        self._client.close()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.close()


CLS = CLSClient  # Legacy alias
