"""
CLS++ Brain SDK — What developers can actually build with this.

The power: Any AI app gets persistent, cross-model memory in 1 line.
No database setup. No embeddings. No vector store. Just brain.learn() and brain.ask().
"""

from clsplusplus import Brain

brain = Brain("demo-user", url="http://localhost:8181")


# ═══════════════════════════════════════════════════════════════════════════
# POWER 1: Any LLM remembers every user — forever
# ═══════════════════════════════════════════════════════════════════════════
# Today your ChatGPT wrapper forgets users after every session.
# Add 2 lines and it never forgets again.

print("═" * 60)
print("POWER 1: Persistent memory across sessions")
print("═" * 60)

# Session 1: User tells the bot about themselves
brain.learn("My name is Priya and I'm a data scientist at Netflix")
brain.learn("I prefer pandas over polars for data analysis")
brain.learn("I'm working on a recommendation engine project")
print("Session 1: Learned 3 facts about Priya")

# Session 2: Days later, different conversation, the bot remembers
facts = brain.ask("What's her current project?")
print(f"Session 2: Asked about project → {facts[0]}")

facts = brain.ask("What tools does she use?")
print(f"Session 2: Asked about tools → {facts[0]}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# POWER 2: Switch models — memory stays
# ═══════════════════════════════════════════════════════════════════════════
# User talks to Claude today, GPT-4 tomorrow, Gemini next week.
# The brain doesn't care which model is asking — it remembers.

print("═" * 60)
print("POWER 2: Cross-model memory (Claude → GPT → Gemini)")
print("═" * 60)

# Claude learns
brain.learn("User corrected: the API returns JSON not XML", source="claude")
# GPT-4 asks later
facts = brain.ask("What format does the API return?")
print(f"GPT-4 asks about API format → {facts[0]}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# POWER 3: One-line LLM context injection
# ═══════════════════════════════════════════════════════════════════════════
# The killer feature: brain.context() returns a formatted string you
# paste directly into any LLM's system prompt. Zero formatting work.

print("═" * 60)
print("POWER 3: LLM-ready context injection")
print("═" * 60)

context = brain.context("data science help")
print("System prompt for any LLM:")
print(context)
print()

# This is how you'd use it with OpenAI:
#
#   context = brain.context("coding help")
#   response = openai.chat.completions.create(
#       model="gpt-4",
#       messages=[
#           {"role": "system", "content": context + "\nYou are a helpful assistant."},
#           {"role": "user", "content": user_message},
#       ]
#   )
#
# That's it. GPT-4 now knows everything about this user.


# ═══════════════════════════════════════════════════════════════════════════
# POWER 4: Semantic search — not keyword matching
# ═══════════════════════════════════════════════════════════════════════════
# brain.ask() understands meaning, not just words.

print("═" * 60)
print("POWER 4: Semantic understanding")
print("═" * 60)

# None of these queries contain the exact stored words
results = brain.ask("What company does she work for?")  # Never said "company"
print(f"'company' → {results[0] if results else 'nothing'}")

results = brain.ask("programming language preference")  # Never said "programming language"
print(f"'programming language preference' → {results[0] if results else 'nothing'}")

results = brain.ask("career")  # Never said "career"
print(f"'career' → {results[0] if results else 'nothing'}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# POWER 5: Multi-user isolation — zero effort
# ═══════════════════════════════════════════════════════════════════════════
# Each Brain is a separate user. No namespace management. No tenant IDs.

print("═" * 60)
print("POWER 5: Multi-user isolation")
print("═" * 60)

alice = Brain("alice", url="http://localhost:8181")
bob = Brain("bob", url="http://localhost:8181")

alice.learn("I love hiking and photography")
bob.learn("I'm into competitive chess")

print(f"Alice's hobbies: {alice.ask('hobbies')}")
print(f"Bob's hobbies: {bob.ask('hobbies')}")
# Alice can never see Bob's memories and vice versa
print()


# ═══════════════════════════════════════════════════════════════════════════
# POWER 6: GDPR right-to-be-forgotten in 1 line
# ═══════════════════════════════════════════════════════════════════════════

print("═" * 60)
print("POWER 6: GDPR forget")
print("═" * 60)

bob.learn("My credit card number is 4242-4242-4242-4242")
print(f"Before forget: {bob.ask('credit card')}")
bob.forget("My credit card number is 4242-4242-4242-4242")
print(f"After forget: {bob.ask('credit card')}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# THE BOTTOM LINE
# ═══════════════════════════════════════════════════════════════════════════
print("═" * 60)
print("SUMMARY: What developers get")
print("═" * 60)
print("""
1 class:    Brain
4 methods:  learn(), ask(), context(), forget()
0 config:   No database, no embeddings, no vector store
∞ models:   Works with Claude, GPT-4, Gemini, Llama, any LLM

Compare to alternatives:
  mem0:     30 lines of setup, requires OpenAI key for embeddings
  Pinecone: 50 lines, need to manage indexes and namespaces
  ChromaDB: 20 lines, local only, no cross-model support
  CLS++:    brain = Brain("user") — done.
""")
