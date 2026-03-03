"""Base store interface."""

from abc import ABC, abstractmethod
from typing import Optional

from clsplusplus.models import MemoryItem, StoreLevel


class BaseStore(ABC):
    """Abstract base for all memory stores."""

    level: StoreLevel

    @abstractmethod
    async def write(self, item: MemoryItem) -> MemoryItem:
        """Write a memory item."""
        pass

    @abstractmethod
    async def read(
        self,
        query_embedding: list[float],
        namespace: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[MemoryItem]:
        """Read memories by semantic similarity."""
        pass

    @abstractmethod
    async def get_by_id(self, item_id: str, namespace: str) -> Optional[MemoryItem]:
        """Get a single item by ID."""
        pass

    @abstractmethod
    async def delete(self, item_id: str, namespace: str) -> bool:
        """Delete an item."""
        pass

    @abstractmethod
    async def health(self) -> dict:
        """Health check."""
        pass
