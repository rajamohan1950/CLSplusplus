"""
CLS++ Brain SDK — 7 Innovative Integrations
Each one solves a real developer problem in 1-3 lines.
"""

from clsplusplus import Brain

URL = "http://localhost:8181"


# ═══════════════════════════════════════════════════════════════════════════
# 1. @brain.wrap — Give any LLM function memory for free
# ═══════════════════════════════════════════════════════════════════════════
print("1. @brain.wrap — Auto-memory for any LLM function")
print("─" * 55)

brain = Brain("wrap-demo", url=URL)
brain.learn("I'm a backend engineer who uses FastAPI")

@brain.wrap
def fake_llm(system_prompt, user_message):
    # This simulates an LLM — in real life, call OpenAI/Claude here
    return f"[LLM received system with memory: {'Yes' if 'Known facts' in system_prompt else 'No'}]"

response = fake_llm("You are a coding assistant", "Help me build an API")
print(f"  LLM got memory context injected: {response}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. brain.absorb() — Bulk learn from text, transcripts, documents
# ═══════════════════════════════════════════════════════════════════════════
print("2. brain.absorb() — Learn from a conversation transcript")
print("─" * 55)

brain2 = Brain("absorb-demo", url=URL)

# Paste an entire conversation and the brain extracts facts
count = brain2.absorb("""
User: I'm moving from New York to Austin next month
User: I just adopted a golden retriever named Max
User: I'm switching from React to Svelte for my side projects
User: My budget for the new apartment is $2000/month
""")
print(f"  Absorbed {count} facts from conversation")
print(f"  Ask about pets: {brain2.ask('pets')}")
print(f"  Ask about budget: {brain2.ask('budget')}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. brain.who() — Auto-generated user profile
# ═══════════════════════════════════════════════════════════════════════════
print("3. brain.who() — Auto-generate a user profile")
print("─" * 55)

profile = brain2.who()
print(f"  User: {profile['user']}")
print(f"  Facts known: {profile['count']}")
print(f"  Summary: {profile['summary']}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 4. brain.correct() — Smart belief update (not just overwrite)
# ═══════════════════════════════════════════════════════════════════════════
print("4. brain.correct() — Update beliefs, not just overwrite")
print("─" * 55)

brain4 = Brain("correct-demo", url=URL)
brain4.learn("I work at Google")
print(f"  Before: {brain4.ask('Where do I work?')}")

brain4.correct("I work at Google", "I just switched to Microsoft")
print(f"  After:  {brain4.ask('Where do I work?')}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 5. brain.chat() — Full conversation handler in 1 line
# ═══════════════════════════════════════════════════════════════════════════
print("5. brain.chat() — Complete conversation handler")
print("─" * 55)

brain5 = Brain("chat-demo", url=URL)
brain5.learn("My favorite cuisine is Japanese")
brain5.learn("I'm lactose intolerant")

# Without LLM — brain answers from memory
answer = brain5.chat("What food should I avoid?")
print(f"  Brain answers from memory: {answer}")

# With LLM — brain auto-injects context, calls LLM, learns from response
def mock_llm(system, message):
    if "lactose" in system.lower():
        return "Based on your dietary needs, I'd suggest trying sushi or ramen!"
    return "I'd suggest trying various restaurants."

answer = brain5.chat("Suggest dinner options", llm_fn=mock_llm)
print(f"  LLM with memory context: {answer}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 6. brain.teach() — Learn from structured data (dicts, JSON, APIs)
# ═══════════════════════════════════════════════════════════════════════════
print("6. brain.teach() — Learn from structured data")
print("─" * 55)

brain6 = Brain("teach-demo", url=URL)

# Learn from a user profile dict (from your database, CRM, or API)
count = brain6.teach({
    "name": "Sarah Chen",
    "role": "VP of Engineering",
    "company": "Datadog",
    "skills": ["Go", "Kubernetes", "Observability"],
    "preferences": {
        "communication": "async over meetings",
        "work_hours": "9am-5pm Pacific"
    }
})
print(f"  Learned {count} facts from structured data")
print(f"  Ask about skills: {brain6.ask('technical skills')}")
print(f"  Ask about work style: {brain6.ask('communication preference')}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 7. brain.watch() — Auto-learn from OpenAI/Anthropic message format
# ═══════════════════════════════════════════════════════════════════════════
print("7. brain.watch() — Learn from OpenAI/Anthropic message history")
print("─" * 55)

brain7 = Brain("watch-demo", url=URL)

# Paste your existing OpenAI messages array — brain auto-extracts facts
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "I'm building a SaaS for healthcare"},
    {"role": "assistant", "content": "That's interesting! What's your tech stack?"},
    {"role": "user", "content": "We use Django and PostgreSQL with HIPAA compliance"},
    {"role": "assistant", "content": "Great choices for healthcare!"},
    {"role": "user", "content": "Can you help me optimize our database queries?"},
]

learned = brain7.watch(messages)
print(f"  Watched {len(messages)} messages, learned {learned} facts")
print(f"  Ask about tech: {brain7.ask('tech stack')}")
print(f"  Ask about industry: {brain7.ask('what industry')}")
print()


# ═══════════════════════════════════════════════════════════════════════════
print("═" * 55)
print("THE FULL API — 11 methods, zero complexity")
print("═" * 55)
print("""
  CORE (4):
    brain.learn(fact)              — Teach it anything
    brain.ask(question)            — Semantic recall
    brain.context(topic)           — LLM-ready prompt injection
    brain.forget(fact)             — GDPR delete

  INNOVATIVE (7):
    brain.wrap(fn)                 — Auto-memory for any LLM function
    brain.absorb(text)             — Bulk learn from text/transcript
    brain.who()                    — Auto user profile
    brain.correct(old, new)        — Smart belief update
    brain.chat(msg, llm_fn)        — Full conversation handler
    brain.teach(dict)              — Learn from structured data
    brain.watch(messages)          — Learn from OpenAI/Anthropic format

  Every method: 1 line. Every integration: 0 config.
""")
