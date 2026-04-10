from clsplusplus import Brain

# Create a brain for a user — that's it
brain = Brain("alice", url="http://localhost:8181")

# Teach it things in natural language
brain.learn("I work at Google as a senior engineer")
brain.learn("I prefer Python over JavaScript")
brain.learn("My favorite editor is VS Code")

# Ask it anything — semantic recall
print("Job:", brain.ask("What's my job?"))
print("Coding:", brain.ask("coding preferences?"))

# Get LLM-ready context
print("\nContext for coding help:")
print(brain.context("coding task"))
