# CLS++ JavaScript/TypeScript SDK

Brain-inspired, model-agnostic persistent memory for Large Language Models.

## Installation

```bash
npm install clsplusplus
```

## Quick Start

```ts
import { Brain } from "clsplusplus";

const brain = new Brain("alice");

// Teach it anything
await brain.learn("I work at Google as a senior engineer");
await brain.learn("I prefer Python over JavaScript");

// Ask it anything — semantic recall, not keyword matching
const facts = await brain.ask("What's my job?");
// → ["I work at Google as a senior engineer"]

// Get LLM-ready context for any prompt
const context = await brain.context("coding help");
// → "Known facts about this user:\n- I work at Google..."

// Forget (GDPR right to be forgotten)
await brain.forget("I work at Google as a senior engineer");
```

## Configuration

```ts
const brain = new Brain("alice", {
  apiKey: "your-api-key",             // or set CLS_API_KEY env var
  url: "https://www.clsplusplus.com", // or set CLS_BASE_URL env var
});
```

## Full API

| Method | Description |
|--------|-------------|
| `brain.learn(fact, meta?)` | Teach a fact. Returns memory ID. |
| `brain.ask(question, limit?)` | Query for relevant facts. Returns string array. |
| `brain.context(topic?, limit?)` | Get LLM-ready context string. |
| `brain.forget(factOrId)` | Forget by text or ID. Returns boolean. |
| `brain.absorb(content, source?)` | Bulk-learn from text or array. Returns count. |
| `brain.who()` | Auto-generated user profile. |
| `brain.correct(wrong, right)` | Update a belief (forget old, learn new). |
| `brain.chat(message, llmFn?, system?)` | Full conversation handler with memory. |
| `brain.teach(data)` | Learn from key-value object. |
| `brain.watch(messages)` | Learn from chat message array. |
| `brain.all(limit?)` | Get all memories. |
| `brain.count()` | Count stored memories. |
| `brain.wrap(llmFn)` | Wrap any LLM function with auto-memory. |

## Module-Level Functions

For scripts and one-liners:

```ts
import { learn, ask, context, forget } from "clsplusplus";

await learn("alice", "Prefers dark mode");
const facts = await ask("alice", "What theme?");
```

## Use with OpenAI

```ts
import OpenAI from "openai";
import { Brain } from "clsplusplus";

const openai = new OpenAI();
const brain = new Brain("alice");

const response = await brain.chat(
  "Help me pick a framework",
  async (system, message) => {
    const res = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        { role: "system", content: system },
        { role: "user", content: message },
      ],
    });
    return res.choices[0].message.content || "";
  },
);
```

## Use with Anthropic

```ts
import Anthropic from "@anthropic-ai/sdk";
import { Brain } from "clsplusplus";

const anthropic = new Anthropic();
const brain = new Brain("alice");

const chat = brain.wrap(async (system, message) => {
  const res = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    system,
    messages: [{ role: "user", content: message }],
  });
  return res.content[0].type === "text" ? res.content[0].text : "";
});

const response = await chat("You are a helpful assistant", "What editor should I use?");
```

## Browser Usage

The SDK uses native `fetch` and works in all modern browsers. Environment variables are not available in browsers, so pass `apiKey` and `url` directly:

```ts
const brain = new Brain("alice", {
  apiKey: "your-key",
  url: "https://www.clsplusplus.com",
});
```

## Requirements

- Node.js 18 or later (uses native `fetch`)
- Zero runtime dependencies

## License

Apache-2.0
