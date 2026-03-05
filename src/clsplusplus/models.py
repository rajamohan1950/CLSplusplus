"""CLS++ data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# Security: max lengths to prevent abuse
MAX_TEXT_LEN = 65536
MAX_NAMESPACE_LEN = 64
MAX_SOURCE_LEN = 64
MAX_QUERY_LEN = 4096
MAX_LIMIT = 100
MAX_METADATA_KEYS = 50
MAX_METADATA_VALUE_LEN = 1024


def _validate_namespace(v: str) -> str:
    if not v or len(v) > MAX_NAMESPACE_LEN:
        raise ValueError("namespace must be 1-64 chars")
    if not all(c.isalnum() or c in "-_" for c in v):
        raise ValueError("namespace: alphanumeric, dash, underscore only")
    return v


def _validate_item_id(v: str) -> str:
    if not v or len(v) > 64:
        raise ValueError("item_id must be 1-64 chars")
    if not all(c.isalnum() or c in "-_" for c in v):
        raise ValueError("item_id: alphanumeric, dash, underscore only")
    return v


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

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    namespace: str = Field(default="default", min_length=1, max_length=MAX_NAMESPACE_LEN)
    source: str = Field(default="user", max_length=MAX_SOURCE_LEN)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    authority: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    subject: Optional[str] = Field(default=None, max_length=256)
    predicate: Optional[str] = Field(default=None, max_length=256)
    object: Optional[str] = Field(default=None, max_length=256)

    @field_validator("namespace")
    @classmethod
    def ns_valid(cls, v: str) -> str:
        return _validate_namespace(v)


class ReadRequest(BaseModel):
    """Request to read memories."""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    namespace: str = Field(default="default", min_length=1, max_length=MAX_NAMESPACE_LEN)
    limit: int = Field(default=10, ge=1, le=MAX_LIMIT)
    store_levels: Optional[list[StoreLevel]] = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("namespace")
    @classmethod
    def ns_valid(cls, v: str) -> str:
        return _validate_namespace(v)


class ForgetRequest(BaseModel):
    """Request to forget (delete) a memory by ID."""

    item_id: str = Field(..., min_length=1, max_length=64)
    namespace: str = Field(default="default", min_length=1, max_length=MAX_NAMESPACE_LEN)

    @field_validator("item_id")
    @classmethod
    def id_valid(cls, v: str) -> str:
        return _validate_item_id(v)

    @field_validator("namespace")
    @classmethod
    def ns_valid(cls, v: str) -> str:
        return _validate_namespace(v)


class ReadResponse(BaseModel):
    """Response from memory read."""

    items: list[MemoryItem]
    query: str
    namespace: str


class AdjudicateRequest(BaseModel):
    """Request to adjudicate conflicting facts."""

    new_fact: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    evidence: list[str] = Field(..., max_length=20)  # max 20 evidence items
    namespace: str = Field(default="default", min_length=1, max_length=MAX_NAMESPACE_LEN)
    existing_item_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("namespace")
    @classmethod
    def ns_valid(cls, v: str) -> str:
        return _validate_namespace(v)


class DemoChatRequest(BaseModel):
    """Request for demo chat with real LLM."""

    model: str = Field(..., min_length=1, max_length=32)
    message: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    namespace: str = Field(default="demo-default", min_length=1, max_length=MAX_NAMESPACE_LEN)

    @field_validator("namespace")
    @classmethod
    def ns_valid(cls, v: str) -> str:
        return _validate_namespace(v)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    stores: dict[str, dict[str, Any]]
    version: str = "0.1.0"
