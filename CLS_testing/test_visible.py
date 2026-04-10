"""
CLS++ Memory Injection — Visible at every step.

OpenAI call:
  1. What prompt OpenAI sends
  2. What CLS++ injects
  3. What OpenAI responds

Then Claude call:
  1. What prompt Claude sends
  2. What CLS++ injects
  3. What Claude responds
"""

import os, httpx
from clsplusplus import Brain

CLS = "http://localhost:8181"
OAI = os.environ.get("OPENAI_API_KEY") or "sk-REDACTED"
ANT = os.environ.get("ANTHROPIC_API_KEY") or "sk-REDACTED"

brain = Brain("visible", url=CLS)

# Teach
brain.learn("My name is Raj Jabbala")
brain.learn("I am the founder of AlphaForge AI Labs")
brain.learn("I prefer Python and FastAPI")
brain.learn("I live in Hyderabad, India")
brain.learn("I am vegetarian")


# ══════════════════════════════════════════════════════════════════════════
# OPENAI
# ══════════════════════════════════════════════════════════════════════════

user_prompt_openai = "What tech stack should I use for my next project?"
injection_openai = brain.context(user_prompt_openai)

print("┌─────────────────────────────────────────────────────────────────┐")
print("│                        OPENAI CALL                            │")
print("├─────────────────────────────────────────────────────────────────┤")
print("│ 1. USER PROMPT (what the user typed):                         │")
print("└─────────────────────────────────────────────────────────────────┘")
print()
print(f"   {user_prompt_openai}")
print()
print("┌─────────────────────────────────────────────────────────────────┐")
print("│ 2. CLS++ INJECTION (what CLS++ added to the system prompt):   │")
print("└─────────────────────────────────────────────────────────────────┘")
print()
for line in injection_openai.split("\n"):
    print(f"   {line}")
print()
print("┌─────────────────────────────────────────────────────────────────┐")
print("│ 3. OPENAI RESPONSE (what GPT-4 replied):                      │")
print("└─────────────────────────────────────────────────────────────────┘")
print()

r = httpx.post(f"{CLS}/v1/chat/completions", json={
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": injection_openai + "\nYou are a helpful assistant. Use the known facts to personalize."},
        {"role": "user", "content": user_prompt_openai},
    ],
}, headers={"Authorization": f"Bearer {OAI}"}, timeout=60)
gpt_reply = r.json()["choices"][0]["message"]["content"]

for line in gpt_reply.split("\n"):
    print(f"   {line}")


# ══════════════════════════════════════════════════════════════════════════
# CLAUDE
# ══════════════════════════════════════════════════════════════════════════

user_prompt_claude = "What do you know about me? Suggest dinner tonight."
injection_claude = brain.context(user_prompt_claude)

print()
print()
print("┌─────────────────────────────────────────────────────────────────┐")
print("│                        CLAUDE CALL                            │")
print("├─────────────────────────────────────────────────────────────────┤")
print("│ 1. USER PROMPT (what the user typed):                         │")
print("└─────────────────────────────────────────────────────────────────┘")
print()
print(f"   {user_prompt_claude}")
print()
print("┌─────────────────────────────────────────────────────────────────┐")
print("│ 2. CLS++ INJECTION (what CLS++ added to the system prompt):   │")
print("└─────────────────────────────────────────────────────────────────┘")
print()
for line in injection_claude.split("\n"):
    print(f"   {line}")
print()
print("┌─────────────────────────────────────────────────────────────────┐")
print("│ 3. CLAUDE RESPONSE (what Claude replied):                     │")
print("└─────────────────────────────────────────────────────────────────┘")
print()

r2 = httpx.post(f"{CLS}/v1/messages", json={
    "model": "claude-haiku-4-5-20251001", "max_tokens": 512,
    "system": injection_claude + "\nYou are a helpful assistant. Use the known facts to personalize.",
    "messages": [{"role": "user", "content": user_prompt_claude}],
}, headers={"x-api-key": ANT, "anthropic-version": "2023-06-01"}, timeout=60)
claude_reply = r2.json()["content"][0]["text"]

for line in claude_reply.split("\n"):
    print(f"   {line}")

print()
