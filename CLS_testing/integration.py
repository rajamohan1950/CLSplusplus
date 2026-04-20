pip install clsplusplus
from clsplusplus import CLS

client = CLS(api_key="cls_live_i6Ls0WGjNjOXYdkzdLMn8tZ1QzKJhp6lTqTjVnr_R1Q")

# Store a memory
client.memories.encode(content="User prefers dark mode", agent_id="a1")

# Retrieve memories
results = client.memories.retrieve(query="user preferences", agent_id="a1")
for item in results.items:
    print(item.text, item.confidence)
