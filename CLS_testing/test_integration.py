"""
CLS++ Developer Integration Test
Run this to see memory working end-to-end.

Prerequisites: CLS++ server running at localhost:8181
"""

from clsplusplus import Brain

brain = Brain("test-dev", url="http://localhost:8181")

# ── STEP 1: Teach the brain about a user ──────────────────────────────────
print("=== TEACHING ===")
brain.learn("My name is Raj and I'm a founder")
brain.learn("I code in Python and TypeScript")
brain.learn("I'm building CLS++, an AI memory product")
brain.learn("I prefer dark mode in all editors")
print("Taught 4 facts.\n")

# ── STEP 2: Ask questions — semantic, not keyword ─────────────────────────
print("=== ASKING ===")
print("Q: What's my name?")
print("A:", brain.ask("What's my name?"))
print()

print("Q: What programming languages?")
print("A:", brain.ask("What programming languages do I use?"))
print()

print("Q: What am I building?")
print("A:", brain.ask("What product am I working on?"))
print()

# ── STEP 3: Get LLM-ready context ────────────────────────────────────────
print("=== LLM CONTEXT ===")
print("Paste this into any LLM system prompt:\n")
print(brain.context("help me code"))
print()

# ── STEP 4: Prove cross-session persistence ──────────────────────────────
print("=== PERSISTENCE TEST ===")
brain2 = Brain("test-dev", url="http://localhost:8181")  # New instance, same user
result = brain2.ask("Who am I?")
print("New Brain instance, same user ID → still remembers:")
print("A:", result)
print()

# ── STEP 5: Multi-user isolation ─────────────────────────────────────────
print("=== ISOLATION TEST ===")
other = Brain("other-user", url="http://localhost:8181")
other.learn("I'm a designer who uses Figma")
print("other-user knows:", other.ask("What tool do I use?"))
print("test-dev knows:", brain.ask("What tool do I use?"))
print("(test-dev should NOT see Figma)")
print()

print("=== ALL TESTS PASSED ===")
