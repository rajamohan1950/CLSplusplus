"""CLS++ — Memory that thinks like a brain.

    from clsplusplus import Brain

    brain = Brain("alice")
    brain.learn("I work at Google")
    brain.ask("Where do I work?")  # → ["I work at Google"]

Or as module-level functions:

    import clsplusplus as mem
    mem.learn("alice", "Prefers dark mode")
    mem.ask("alice", "What theme?")
"""

from __future__ import annotations

__version__ = "7.0.0"

__all__ = ["Brain", "learn", "ask", "context", "forget", "CLS", "CLSClient", "__version__"]


def __getattr__(name: str):
    from clsplusplus.client import Brain, learn, ask, context, forget, CLS, CLSClient, MemoriesClient
    _map = {
        "Brain": Brain, "learn": learn, "ask": ask, "context": context,
        "forget": forget, "CLS": CLS, "CLSClient": CLSClient, "MemoriesClient": MemoriesClient,
    }
    if name in _map:
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
