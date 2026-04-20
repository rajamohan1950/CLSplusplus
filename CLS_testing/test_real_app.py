"""
How CLS++ actually works in a real app.

The developer NEVER manually calls brain.learn() for each message.
The system auto-captures conversations. Two ways:
"""

from clsplusplus import Brain

brain = Brain("real-user", url="http://localhost:8181")

print("=" * 60)
print("APPROACH 1: @brain.wrap — auto-capture everything")
print("=" * 60)
print()

# Developer wraps their LLM function ONCE.
# After that, every conversation is auto-captured.

@brain.wrap
def call_llm(system_prompt, user_message):
    """This is the developer's existing LLM call.
    Could be OpenAI, Claude, Gemini — anything."""
    # Simulating an LLM response
    if "dark mode" in system_prompt.lower() or "dark mode" in user_message.lower():
        return "I'll use dark theme for all code examples since you prefer that."
    return "Sure, I can help with that!"

# User has a conversation — developer's code is UNCHANGED
# brain.wrap auto-injects memory AND auto-captures the conversation
print("User: I prefer dark mode in all my tools")
response = call_llm("You are a coding assistant", "I prefer dark mode in all my tools")
print(f"Bot: {response}")
print()

print("User: My tech stack is FastAPI + React + PostgreSQL")
response = call_llm("You are a coding assistant", "My tech stack is FastAPI + React + PostgreSQL")
print(f"Bot: {response}")
print()

# Later session — the brain ALREADY knows everything
# Developer didn't write ANY capture code
print("--- NEW SESSION (hours later) ---")
print()
print("User: Help me with my project")
response = call_llm("You are a coding assistant", "Help me with my project")
print(f"Bot: {response}")
print()

# What did the brain auto-capture?
print("What the brain auto-captured (developer wrote zero capture code):")
for fact in brain.ask("everything about this user", limit=10):
    print(f"  - {fact}")
print()


print("=" * 60)
print("APPROACH 2: brain.watch() — ingest existing conversations")
print("=" * 60)
print()

brain2 = Brain("watch-user", url="http://localhost:8181")

# Developer already has OpenAI message history in their DB.
# Just feed it to brain.watch() — it auto-extracts user facts.

existing_chat_history = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "I'm a data scientist at Netflix"},
    {"role": "assistant", "content": "That's cool! How can I help?"},
    {"role": "user", "content": "I mainly use Python and PySpark for big data"},
    {"role": "assistant", "content": "Great tools for large-scale data processing!"},
    {"role": "user", "content": "Can you help me optimize a recommendation model?"},
    {"role": "assistant", "content": "Of course! What's your current approach?"},
    {"role": "user", "content": "I'm using collaborative filtering with matrix factorization"},
]

learned = brain2.watch(existing_chat_history)
print(f"Fed {len(existing_chat_history)} messages → auto-learned {learned} facts")
print(f"(Skipped questions and assistant messages automatically)")
print()

print("Now ask the brain about this user:")
print(f"  Company: {brain2.ask('What company?')}")
print(f"  Skills:  {brain2.ask('technical skills')}")
print(f"  Project: {brain2.ask('what are they building')}")
print()


print("=" * 60)
print("APPROACH 3: brain.absorb() — ingest support tickets, docs, notes")
print("=" * 60)
print()

brain3 = Brain("absorb-user", url="http://localhost:8181")

# Customer support agent pastes a ticket — brain extracts everything
support_ticket = """
Customer: John Smith
Account: Enterprise plan
Issue: API rate limiting errors on production
Details: They're hitting 5000 req/min limit.
Their app processes 10M events daily.
Stack: Node.js + Redis + MongoDB
Priority: High — revenue impacting
"""

count = brain3.absorb(support_ticket, source="support-ticket")
print(f"Absorbed support ticket → extracted {count} facts")
print()
print("Now any agent can ask about this customer:")
print(f"  Plan:    {brain3.ask('what plan')}")
print(f"  Issue:   {brain3.ask('what is the problem')}")
print(f"  Scale:   {brain3.ask('how many events')}")
print()


print("=" * 60)
print("THE POINT")
print("=" * 60)
print("""
Developers NEVER manually call brain.learn() in production.

They use ONE of these:
  1. Change base_url (Zero Code) — proxy auto-captures everything
  2. @brain.wrap(fn)             — decorator auto-captures conversations
  3. brain.watch(messages)       — ingest existing chat history
  4. brain.absorb(text)          — ingest documents, tickets, notes
  5. brain.teach(dict)           — ingest structured data (CRM, DB records)

The brain fills itself. The developer just asks it questions later.
""")
