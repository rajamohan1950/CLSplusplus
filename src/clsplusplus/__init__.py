"""CLS++ - Continuous Learning System++

Brain-inspired, model-agnostic persistent memory architecture for LLMs.

3-line integration:
    from clsplusplus import CLS
    client = CLS(api_key="cls_live_xxx")
    client.memories.encode(content="User prefers dark mode", agent_id="a1")
"""

__version__ = "0.1.0"

from clsplusplus.client import CLS, CLSClient, MemoriesClient

__all__ = ["CLS", "CLSClient", "MemoriesClient", "__version__"]
