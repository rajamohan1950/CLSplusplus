"""CLS++ data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class StoreLevel(str, Enum):
    """Memory store level (L0-L3)."""

    L0 = "L0"  # Working Buffer
    L1 = "L1"  # Indexing Store
    L2 = "L2"  # Schema Graph
    L3 = "L3"  # Deep Recess


class MemoryItem(BaseModel):
    """A memory item across all stores."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    namespace: str = "default"
    store_level: StoreLevel = StoreLevel.L0

    # Provenance
    source: str = "user"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 0.5
    version: int = 1
    checksum: Optional[str] = None
    lineage: list[str] = Field(default_factory=list)

    # Plasticity signals
    salience: float = 0.5
    usage_count: int = 0
    authority: float = 0.5
    conflict_score: float = 0.0
    surprise: float = 0.0
    promotion_score: float = 0.0

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None

    # For L2 graph
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for storage."""
        d = self.model_dump()
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryItem":
        """Create from dict."""
        if "timestamp" in d and isinstance(d["timestamp"], str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00"))
        return cls(**d)


class WriteRequest(BaseModel):
    """Request to write memory."""

    text: str
    namespace: str = "default"
    source: str = "user"
    salience: float = 0.5
    authority: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None


class ReadRequest(BaseModel):
    """Request to read memories."""

    query: str
    namespace: str = "default"
    limit: int = 10
    store_levels: Optional[list[StoreLevel]] = None
    min_confidence: float = 0.0


class ReadResponse(BaseModel):
    """Response from memory read."""

    items: list[MemoryItem]
    query: str
    namespace: str


class AdjudicateRequest(BaseModel):
    """Request to adjudicate conflicting facts."""

    new_fact: str
    evidence: list[str]
    namespace: str = "default"
    existing_item_id: Optional[str] = None


class DemoChatRequest(BaseModel):
    """Request for demo chat with real LLM."""

    model: str  # claude, openai, gemini
    message: str
    namespace: str = "demo-default"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    stores: dict[str, dict[str, Any]]
    version: str = "0.1.0"
