export { Brain } from "./brain.js";
export type {
  BrainOptions,
  MemoryItem,
  ReadResponse,
  WriteResponse,
  ForgetResponse,
  WhoResult,
  Message,
  LlmFunction,
} from "./types.js";

// ── Module-level convenience functions ────────────────────────────────

import { Brain } from "./brain.js";

const _brains = new Map<string, Brain>();

function _getBrain(user: string): Brain {
  let brain = _brains.get(user);
  if (!brain) {
    brain = new Brain(user);
    _brains.set(user, brain);
  }
  return brain;
}

/**
 * Module-level: teach a fact for a user.
 *
 * @example
 * ```ts
 * import { learn } from "clsplusplus";
 * await learn("alice", "Prefers dark mode");
 * ```
 */
export function learn(
  user: string,
  fact: string,
  meta?: Record<string, unknown>,
): Promise<string> {
  return _getBrain(user).learn(fact, meta);
}

/**
 * Module-level: ask a question about a user.
 *
 * @example
 * ```ts
 * import { ask } from "clsplusplus";
 * const facts = await ask("alice", "What theme?");
 * ```
 */
export function ask(
  user: string,
  question: string,
  limit: number = 5,
): Promise<string[]> {
  return _getBrain(user).ask(question, limit);
}

/**
 * Module-level: get LLM context for a user.
 *
 * @example
 * ```ts
 * import { context } from "clsplusplus";
 * const prompt = await context("alice", "coding help");
 * ```
 */
export function context(
  user: string,
  topic: string = "",
  limit: number = 8,
): Promise<string> {
  return _getBrain(user).context(topic, limit);
}

/**
 * Module-level: forget a fact for a user.
 *
 * @example
 * ```ts
 * import { forget } from "clsplusplus";
 * await forget("alice", "old fact");
 * ```
 */
export function forget(user: string, factOrId: string): Promise<boolean> {
  return _getBrain(user).forget(factOrId);
}
