import type {
  BrainOptions,
  ReadResponse,
  WriteResponse,
  WhoResult,
  Message,
  LlmFunction,
} from "./types.js";

/** Read an environment variable safely (no-op in browsers). */
function getEnv(key: string, fallback: string = ""): string {
  try {
    return (typeof process !== "undefined" && process.env?.[key]) || fallback;
  } catch {
    return fallback;
  }
}

/**
 * A persistent memory for one user. Learns, recalls, forgets — like a brain.
 *
 * @example
 * ```ts
 * import { Brain } from "clsplusplus";
 *
 * const brain = new Brain("alice");
 * await brain.learn("I work at Google as a senior engineer");
 * const facts = await brain.ask("What's my job?");
 * ```
 */
export class Brain {
  readonly user: string;
  private readonly _url: string;
  private readonly _key: string;

  constructor(user: string, opts?: BrainOptions) {
    this.user = user;
    this._url = (
      opts?.url || getEnv("CLS_BASE_URL", "https://www.clsplusplus.com")
    ).replace(/\/+$/, "");
    this._key = opts?.apiKey || getEnv("CLS_API_KEY");
  }

  // ── Internal HTTP helper ────────────────────────────────────────────

  private async _fetch<T>(
    method: string,
    path: string,
    body?: Record<string, unknown>,
  ): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this._key) {
      headers["Authorization"] = `Bearer ${this._key}`;
    }

    try {
      const resp = await fetch(`${this._url}${path}`, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(
          `CLS++ API error ${resp.status}: ${text || resp.statusText}`,
        );
      }
      return (await resp.json()) as T;
    } finally {
      clearTimeout(timeout);
    }
  }

  // ── Core API: learn / ask / context / forget ────────────────────────

  /**
   * Teach the brain a fact. Returns the memory ID.
   *
   * @example
   * ```ts
   * await brain.learn("User prefers dark mode");
   * await brain.learn("Allergic to peanuts", { source: "medical" });
   * ```
   */
  async learn(
    fact: string,
    meta?: Record<string, unknown>,
  ): Promise<string> {
    const body: Record<string, unknown> = {
      text: fact,
      namespace: this.user,
      source: (meta?.source as string) || "user",
    };
    if (meta) {
      const { source: _, ...rest } = meta;
      if (Object.keys(rest).length > 0) {
        body.metadata = rest;
      }
    }
    const resp = await this._fetch<WriteResponse>("POST", "/v1/memory/write", body);
    return resp.id || "";
  }

  /**
   * Ask the brain a question. Returns relevant facts as strings.
   *
   * @example
   * ```ts
   * const facts = await brain.ask("What's my job?");
   * // → ["I work at Google as a senior engineer"]
   * ```
   */
  async ask(question: string, limit: number = 5): Promise<string[]> {
    const resp = await this._fetch<ReadResponse>("POST", "/v1/memory/read", {
      query: question,
      namespace: this.user,
      limit,
    });
    return (resp.items || []).map((item) => item.text);
  }

  /**
   * Get LLM-ready context string. Inject directly into your system prompt.
   *
   * @example
   * ```ts
   * const prompt = await brain.context("coding help");
   * ```
   */
  async context(topic: string = "", limit: number = 8): Promise<string> {
    const query = topic || "everything about this user";
    const facts = await this.ask(query, limit);
    if (facts.length === 0) return "";
    return ["Known facts about this user:", ...facts.map((f) => `- ${f}`)].join(
      "\n",
    );
  }

  /**
   * Forget a specific memory. Pass the fact text or memory ID.
   *
   * @example
   * ```ts
   * await brain.forget("Allergic to peanuts");
   * ```
   */
  async forget(factOrId: string): Promise<boolean> {
    // Try as ID first
    try {
      await this._fetch("DELETE", "/v1/memory/forget", {
        item_id: factOrId,
        namespace: this.user,
      });
      return true;
    } catch {
      // Not found by ID — fall through to search
    }

    // Search by text and delete the best match
    try {
      const resp = await this._fetch<ReadResponse>(
        "POST",
        "/v1/memory/read",
        { query: factOrId, namespace: this.user, limit: 1 },
      );
      if (resp.items?.length > 0) {
        await this._fetch("DELETE", "/v1/memory/forget", {
          item_id: resp.items[0].id,
          namespace: this.user,
        });
        return true;
      }
    } catch {
      // Ignore
    }
    return false;
  }

  // ── Innovative Integrations ─────────────────────────────────────────

  /**
   * Wrap an LLM function with auto-memory injection and learning.
   *
   * @example
   * ```ts
   * const chat = brain.wrap(async (system, message) => {
   *   return await openai.chat({ system, user: message });
   * });
   * const response = await chat("You are a helper", "Help with Python");
   * ```
   */
  wrap(llmFn: LlmFunction) {
    return async (
      systemPrompt: string,
      userMessage: string,
      ...args: unknown[]
    ): Promise<string> => {
      const memContext = await this.context(userMessage);
      const augmented = memContext
        ? memContext + "\n\n" + systemPrompt
        : systemPrompt;
      const result = await llmFn(augmented, userMessage, ...args);
      await this.learn(userMessage, { source: "user" });
      return result;
    };
  }

  /**
   * Bulk-learn from a conversation, document, or text block.
   *
   * @example
   * ```ts
   * await brain.absorb("User: I'm a vegetarian\nUser: I live in SF");
   * await brain.absorb(["Loves sushi", "Birthday is March 5"]);
   * ```
   */
  async absorb(
    content: string | string[],
    source: string = "document",
  ): Promise<number> {
    let facts: string[];

    if (Array.isArray(content)) {
      facts = content;
    } else {
      const lines = String(content).split(/\n|(?<=[.!?])\s+/);
      facts = [];
      for (const line of lines) {
        const cleaned = line
          .replace(/^(User|Assistant|Human|AI)\s*:\s*/i, "")
          .trim();
        if (cleaned.length > 8) {
          facts.push(cleaned);
        }
      }
    }

    let count = 0;
    for (const fact of facts) {
      try {
        await this.learn(fact, { source });
        count++;
      } catch {
        // Skip failures
      }
    }
    return count;
  }

  /**
   * Auto-generate a structured user profile from stored memories.
   *
   * @example
   * ```ts
   * const profile = await brain.who();
   * // → { user: "alice", facts: [...], summary: "...", count: 5 }
   * ```
   */
  async who(): Promise<WhoResult> {
    const facts = await this.ask("everything about this user", 20);
    return {
      user: this.user,
      facts,
      summary:
        facts.length > 0
          ? facts.slice(0, 5).join("; ")
          : "No information stored",
      count: facts.length,
    };
  }

  /**
   * Update a belief. Forgets the old fact, learns the new one.
   *
   * @example
   * ```ts
   * await brain.correct("I work at Google", "I just moved to Microsoft");
   * ```
   */
  async correct(wrong: string, right: string): Promise<string> {
    await this.forget(wrong);
    return this.learn(right, { source: "correction" });
  }

  /**
   * Complete conversation handler: recall, inject, call LLM, learn.
   *
   * @example
   * ```ts
   * const response = await brain.chat("What editor should I use?", myLlm);
   * ```
   */
  async chat(
    userMessage: string,
    llmFn?: LlmFunction,
    system: string = "You are a helpful assistant.",
  ): Promise<string> {
    const mem = await this.context(userMessage);
    const augmented = mem ? mem + "\n\n" + system : system;

    if (llmFn) {
      const response = await llmFn(augmented, userMessage);
      await this.learn(userMessage, { source: "user" });
      if (typeof response === "string" && response.length > 10) {
        await this.learn(`Assistant replied: ${response.slice(0, 300)}`, {
          source: "assistant",
        });
      }
      return response;
    }

    // No LLM — return what the brain knows
    const facts = await this.ask(userMessage);
    return facts.length > 0
      ? facts.join("\n")
      : "I don't have any information about that yet.";
  }

  /**
   * Learn from a dictionary of key-value pairs.
   *
   * @example
   * ```ts
   * await brain.teach({ name: "Priya", company: "Netflix", languages: ["Python", "R"] });
   * ```
   */
  async teach(data: Record<string, unknown>): Promise<number> {
    let count = 0;

    const flatten = async (
      obj: unknown,
      prefix: string = "",
    ): Promise<void> => {
      if (obj && typeof obj === "object" && !Array.isArray(obj)) {
        for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
          await flatten(v, prefix ? `${prefix}${k}: ` : `${k}: `);
        }
      } else if (Array.isArray(obj)) {
        const val = obj.map(String).join(", ");
        await this.learn(`${prefix}${val}`);
        count++;
      } else {
        await this.learn(`${prefix}${obj}`);
        count++;
      }
    };

    await flatten(data);
    return count;
  }

  /**
   * Learn from OpenAI/Anthropic-style message list.
   * Extracts user statements (skips questions and assistant messages).
   *
   * @example
   * ```ts
   * await brain.watch([
   *   { role: "user", content: "I'm a Python developer at Stripe" },
   *   { role: "assistant", content: "Nice!" },
   * ]);
   * ```
   */
  async watch(messages: Message[]): Promise<number> {
    let count = 0;
    for (const msg of messages) {
      if (msg.role === "user") {
        const text = (msg.content || "").trim();
        if (text && text.length > 8 && !text.includes("?")) {
          await this.learn(text, { source: "conversation" });
          count++;
        }
      }
    }
    return count;
  }

  /** Get all memories for this user. */
  async all(limit: number = 50): Promise<string[]> {
    return this.ask("everything", limit);
  }

  /** How many things the brain remembers. */
  async count(): Promise<number> {
    const items = await this.all(100);
    return items.length;
  }

  toString(): string {
    return `Brain('${this.user}')`;
  }
}
