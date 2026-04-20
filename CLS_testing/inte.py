from clsplusplus import Brain

brain = Brain("alice", url="http://localhost:8181")

brain.learn("I work at Google as a senior engineer")
brain.learn("I prefer Python over JavaScript")
brain.learn("My favorite editor is VS Code")

text = brain.ask("What's my job?")
print(text)
# → ["I work at Google as a senior engineer"]

brain.ask("coding preferences?")
# → ["I prefer Python over JavaScript"]

brain.context("coding task")
# → "Known facts about this user:\n- My favorite editor is VS Code\n- I prefer Python...\n- I work at Google..."

