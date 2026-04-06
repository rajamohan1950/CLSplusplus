/** Options for creating a Brain instance. */
export interface BrainOptions {
  /** API key for authentication. Falls back to CLS_API_KEY env var. */
  apiKey?: string;
  /** Server URL. Falls back to CLS_BASE_URL env var, then https://www.clsplusplus.com. */
  url?: string;
}

/** A single memory item returned by the API. */
export interface MemoryItem {
  id: string;
  text: string;
  confidence?: number;
  store_level?: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

/** Response from POST /v1/memory/read. */
export interface ReadResponse {
  items: MemoryItem[];
  query: string;
  namespace: string;
  trace_id?: string;
}

/** Response from POST /v1/memory/write. */
export interface WriteResponse {
  id: string;
  store_level?: string;
  text?: string;
  trace_id?: string;
}

/** Response from DELETE /v1/memory/forget. */
export interface ForgetResponse {
  deleted: boolean;
  item_id?: string;
}

/** Auto-generated user profile from Brain.who(). */
export interface WhoResult {
  user: string;
  facts: string[];
  summary: string;
  count: number;
}

/** An OpenAI/Anthropic-style chat message. */
export interface Message {
  role: string;
  content: string;
}

/** A function that calls an LLM. */
export type LlmFunction = (
  systemPrompt: string,
  userMessage: string,
  ...args: unknown[]
) => string | Promise<string>;
