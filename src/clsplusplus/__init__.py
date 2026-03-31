"""CLS++ - Continuous Learning System++

Brain-inspired, model-agnostic persistent memory architecture for LLMs.

3-line integration (loads SDK on first use):
    from clsplusplus import CLS
    client = CLS(api_key="cls_live_xxx")
    client.memories.encode(content="User prefers dark mode", agent_id="a1")

Embedded installs (e.g. local server) can import the phase engine without pulling the HTTP client:
    from clsplusplus.memory_phase import PhaseMemoryEngine
"""

from __future__ import annotations

__version__ = "1.5.0"

__all__ = ["CLS", "CLSClient", "MemoriesClient", "__version__"]


def __getattr__(name: str):
    if name == "CLS":
        from clsplusplus.client import CLS

        return CLS
    if name == "CLSClient":
        from clsplusplus.client import CLSClient

        return CLSClient
    if name == "MemoriesClient":
        from clsplusplus.client import MemoriesClient

        return MemoriesClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
