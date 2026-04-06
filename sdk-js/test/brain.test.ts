import { describe, it, expect, vi, beforeEach } from "vitest";
import { Brain, learn, ask, context, forget } from "../src/index.js";

// ── Mock fetch globally ───────────────────────────────────────────────

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  };
}

beforeEach(() => {
  mockFetch.mockReset();
});

// ── Brain class tests ─────────────────────────────────────────────────

describe("Brain", () => {
  const brain = new Brain("alice", {
    url: "http://localhost:8181",
    apiKey: "test-key",
  });

  describe("learn", () => {
    it("sends correct request and returns id", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ id: "mem-123" }));

      const id = await brain.learn("I like Python");

      expect(id).toBe("mem-123");
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8181/v1/memory/write",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            text: "I like Python",
            namespace: "alice",
            source: "user",
          }),
        }),
      );
    });

    it("passes custom source and metadata", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ id: "mem-456" }));

      await brain.learn("Allergic to peanuts", {
        source: "medical",
        severity: "high",
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.source).toBe("medical");
      expect(body.metadata).toEqual({ severity: "high" });
    });
  });

  describe("ask", () => {
    it("returns fact texts", async () => {
      mockFetch.mockResolvedValueOnce(
        mockResponse({
          items: [
            { id: "1", text: "Works at Google" },
            { id: "2", text: "Prefers Python" },
          ],
        }),
      );

      const facts = await brain.ask("What's my job?");

      expect(facts).toEqual(["Works at Google", "Prefers Python"]);
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.query).toBe("What's my job?");
      expect(body.namespace).toBe("alice");
      expect(body.limit).toBe(5);
    });

    it("respects custom limit", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ items: [] }));

      await brain.ask("anything", 20);

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.limit).toBe(20);
    });
  });

  describe("context", () => {
    it("formats facts as LLM-ready string", async () => {
      mockFetch.mockResolvedValueOnce(
        mockResponse({
          items: [
            { id: "1", text: "Works at Google" },
            { id: "2", text: "Prefers Python" },
          ],
        }),
      );

      const ctx = await brain.context("coding");

      expect(ctx).toBe(
        "Known facts about this user:\n- Works at Google\n- Prefers Python",
      );
    });

    it("returns empty string when no facts", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ items: [] }));
      const ctx = await brain.context();
      expect(ctx).toBe("");
    });
  });

  describe("forget", () => {
    it("returns true when ID delete succeeds", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ deleted: true }));
      const result = await brain.forget("mem-123");
      expect(result).toBe(true);
    });

    it("falls back to search-and-delete when ID fails", async () => {
      // First call: ID delete fails
      mockFetch.mockResolvedValueOnce(mockResponse({ error: "not found" }, 404));
      // Second call: search by text
      mockFetch.mockResolvedValueOnce(
        mockResponse({ items: [{ id: "found-id", text: "old fact" }] }),
      );
      // Third call: delete by found ID
      mockFetch.mockResolvedValueOnce(mockResponse({ deleted: true }));

      const result = await brain.forget("old fact");

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });

    it("returns false when nothing found", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({}, 404));
      mockFetch.mockResolvedValueOnce(mockResponse({ items: [] }));

      const result = await brain.forget("nonexistent");
      expect(result).toBe(false);
    });
  });

  describe("who", () => {
    it("returns structured profile", async () => {
      mockFetch.mockResolvedValueOnce(
        mockResponse({
          items: [
            { id: "1", text: "Works at Google" },
            { id: "2", text: "Prefers Python" },
          ],
        }),
      );

      const profile = await brain.who();

      expect(profile.user).toBe("alice");
      expect(profile.facts).toEqual(["Works at Google", "Prefers Python"]);
      expect(profile.summary).toBe("Works at Google; Prefers Python");
      expect(profile.count).toBe(2);
    });
  });

  describe("correct", () => {
    it("forgets old and learns new", async () => {
      // forget: ID delete succeeds
      mockFetch.mockResolvedValueOnce(mockResponse({ deleted: true }));
      // learn: write succeeds
      mockFetch.mockResolvedValueOnce(mockResponse({ id: "new-id" }));

      const id = await brain.correct("old fact", "new fact");
      expect(id).toBe("new-id");
    });
  });

  describe("absorb", () => {
    it("learns from array", async () => {
      mockFetch.mockResolvedValue(mockResponse({ id: "x" }));

      const count = await brain.absorb(["fact one", "fact two", "fact three"]);
      expect(count).toBe(3);
    });

    it("splits text and filters short lines", async () => {
      mockFetch.mockResolvedValue(mockResponse({ id: "x" }));

      const count = await brain.absorb(
        "User: I am a developer\nAssistant: Cool\nI live in SF and love it",
      );
      // "I am a developer" (>8 chars) ✓
      // "Cool" (<8 chars) ✗
      // "I live in SF and love it" (>8 chars) ✓
      expect(count).toBe(2);
    });
  });

  describe("teach", () => {
    it("flattens nested objects", async () => {
      mockFetch.mockResolvedValue(mockResponse({ id: "x" }));

      const count = await brain.teach({
        name: "Priya",
        languages: ["Python", "R"],
      });

      expect(count).toBe(2);
      const calls = mockFetch.mock.calls.map((c: unknown[]) =>
        JSON.parse((c[1] as RequestInit).body as string),
      );
      expect(calls[0].text).toBe("name: Priya");
      expect(calls[1].text).toBe("languages: Python, R");
    });
  });

  describe("watch", () => {
    it("learns user messages, skips questions and short text", async () => {
      mockFetch.mockResolvedValue(mockResponse({ id: "x" }));

      const count = await brain.watch([
        { role: "user", content: "I'm a Python developer at Stripe" },
        { role: "assistant", content: "Nice! How can I help?" },
        { role: "user", content: "What time is it?" },
        { role: "user", content: "short" },
        { role: "user", content: "I prefer functional programming" },
      ]);

      // Only the 2 long non-question user messages
      expect(count).toBe(2);
    });
  });

  describe("wrap", () => {
    it("injects context and auto-learns", async () => {
      // context → ask
      mockFetch.mockResolvedValueOnce(
        mockResponse({ items: [{ id: "1", text: "Likes Python" }] }),
      );
      // learn user message
      mockFetch.mockResolvedValueOnce(mockResponse({ id: "x" }));

      const llm = vi.fn().mockResolvedValue("Use pytest!");
      const wrapped = brain.wrap(llm);

      const result = await wrapped("You are a helper", "How to test?");

      expect(result).toBe("Use pytest!");
      // LLM called with augmented system prompt
      expect(llm.mock.calls[0][0]).toContain("Likes Python");
      expect(llm.mock.calls[0][1]).toBe("How to test?");
    });
  });

  describe("chat", () => {
    it("without LLM returns known facts", async () => {
      mockFetch
        .mockResolvedValueOnce(mockResponse({ items: [] })) // context
        .mockResolvedValueOnce(
          mockResponse({ items: [{ id: "1", text: "Fact A" }] }),
        ); // ask

      const result = await brain.chat("What do you know?");
      expect(result).toBe("Fact A");
    });

    it("returns default when no facts and no LLM", async () => {
      mockFetch
        .mockResolvedValueOnce(mockResponse({ items: [] }))
        .mockResolvedValueOnce(mockResponse({ items: [] }));

      const result = await brain.chat("anything");
      expect(result).toBe("I don't have any information about that yet.");
    });
  });

  describe("all and count", () => {
    it("all returns all facts", async () => {
      mockFetch.mockResolvedValueOnce(
        mockResponse({ items: [{ id: "1", text: "A" }, { id: "2", text: "B" }] }),
      );
      const facts = await brain.all();
      expect(facts).toEqual(["A", "B"]);
    });

    it("count returns number of facts", async () => {
      mockFetch.mockResolvedValueOnce(
        mockResponse({ items: [{ id: "1", text: "A" }, { id: "2", text: "B" }] }),
      );
      const n = await brain.count();
      expect(n).toBe(2);
    });
  });

  describe("auth header", () => {
    it("includes Bearer token when apiKey provided", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ items: [] }));
      await brain.ask("test");
      const headers = mockFetch.mock.calls[0][1].headers;
      expect(headers["Authorization"]).toBe("Bearer test-key");
    });

    it("omits auth header when no apiKey", async () => {
      const noKeyBrain = new Brain("bob", {
        url: "http://localhost:8181",
      });
      mockFetch.mockResolvedValueOnce(mockResponse({ items: [] }));
      await noKeyBrain.ask("test");
      const headers = mockFetch.mock.calls[0][1].headers;
      expect(headers["Authorization"]).toBeUndefined();
    });
  });

  describe("error handling", () => {
    it("throws on non-2xx response", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse("server error", 500));
      await expect(brain.ask("test")).rejects.toThrow("CLS++ API error 500");
    });
  });
});

// ── Module-level convenience function tests ───────────────────────────

describe("module-level functions", () => {
  it("learn delegates to Brain.learn", async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ id: "conv-1" }));
    const id = await learn("bob", "Likes pizza");
    expect(id).toBe("conv-1");
  });

  it("ask delegates to Brain.ask", async () => {
    mockFetch.mockResolvedValueOnce(
      mockResponse({ items: [{ id: "1", text: "Likes pizza" }] }),
    );
    const facts = await ask("bob", "food?");
    expect(facts).toEqual(["Likes pizza"]);
  });

  it("context delegates to Brain.context", async () => {
    mockFetch.mockResolvedValueOnce(
      mockResponse({ items: [{ id: "1", text: "Likes pizza" }] }),
    );
    const ctx = await context("bob", "food");
    expect(ctx).toContain("Likes pizza");
  });

  it("forget delegates to Brain.forget", async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ deleted: true }));
    const result = await forget("bob", "some-id");
    expect(result).toBe(true);
  });
});
