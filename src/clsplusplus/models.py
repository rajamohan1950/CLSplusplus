"""CLS++ data models."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class TemporalFilter:
    """Temporal constraints for a memory read query.

    Produced by ``parse_temporal_filter()`` from the query text, or supplied
    directly by the caller.  All fields have safe defaults that disable
    filtering / decay so the object is always safe to pass through.
    """

    start: Optional[datetime] = None          # inclusive lower bound on event_at
    end: Optional[datetime] = None            # inclusive upper bound on event_at
    recency_half_life_days: float = 90.0      # exponential-decay half-life
    temporal_signal: str = "none"             # "recent"|"historical"|"range"|"none"
    recency_alpha: float = 0.1               # blend weight: (1-α)·semantic + α·recency


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

    # Temporal provenance
    event_at: Optional[datetime] = None       # when the event HAPPENED (differs from write timestamp)
    superseded: bool = False                  # True = a newer fact on the same topic exists

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

    # When the conversation/event occurred. If provided, relative date references
    # in `text` ("yesterday", "last week", "in 3 days") are resolved to absolute
    # dates and appended inline so the memory remains queryable forever.
    # ISO-8601 string or datetime accepted.  Example: "2024-05-08T14:30:00"
    conversation_date: Optional[datetime] = Field(default=None)

    # Explicit event_at override.  If omitted, extract_event_date() is used to
    # auto-detect from the resolved text.  If that also fails, event_at is left None.
    event_at: Optional[datetime] = Field(default=None)

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

    # Temporal filtering. If None, auto-parsed from query text at read time.
    temporal_filter: Optional[TemporalFilter] = Field(default=None)
    # When False (default), superseded facts are hidden from results.
    include_superseded: bool = Field(default=False)

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
    trace_id: Optional[str] = None


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


# ============================================================================
# Integration models — Self-service API integration management
# ============================================================================

MAX_INTEGRATION_NAME_LEN = 128
MAX_URL_LEN = 2048
MAX_LABEL_LEN = 128


class IntegrationCreate(BaseModel):
    """Request to register a new integration."""

    name: str = Field(..., min_length=1, max_length=MAX_INTEGRATION_NAME_LEN)
    description: str = Field(default="", max_length=1024)
    namespace: str = Field(default="default", min_length=1, max_length=MAX_NAMESPACE_LEN)
    owner_email: Optional[str] = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("namespace")
    @classmethod
    def ns_valid(cls, v: str) -> str:
        return _validate_namespace(v)


class IntegrationResponse(BaseModel):
    """Integration details returned to client."""

    id: str
    name: str
    description: str
    namespace: str
    status: str
    owner_email: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    key_count: int = 0
    webhook_count: int = 0


class ApiKeyCreate(BaseModel):
    """Request to create a new API key for an integration."""

    scopes: list[str] = Field(
        default=["memories:read", "memories:write"],
        max_length=20,
    )
    label: str = Field(default="", max_length=MAX_LABEL_LEN)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class ApiKeyResponse(BaseModel):
    """API key returned to client. Full key shown only on creation."""

    id: str
    integration_id: str
    key_prefix: str
    key_hint: str
    scopes: list[str]
    label: str
    status: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    # Only populated on creation — never returned again
    key: Optional[str] = None


class WebhookCreate(BaseModel):
    """Request to subscribe to webhook events."""

    url: str = Field(..., min_length=10, max_length=MAX_URL_LEN)
    events: list[str] = Field(
        default=["*"],
        max_length=50,
    )
    description: str = Field(default="", max_length=1024)
    namespace_filter: Optional[str] = Field(default=None, max_length=MAX_NAMESPACE_LEN)


class WebhookResponse(BaseModel):
    """Webhook subscription returned to client."""

    id: str
    integration_id: str
    url: str
    events: list[str]
    description: str
    status: str
    failure_count: int = 0
    created_at: datetime
    namespace_filter: Optional[str] = None
    # Only populated on creation
    secret: Optional[str] = None


class IntegrationEventResponse(BaseModel):
    """Audit log entry."""

    id: str
    integration_id: str
    event_type: str
    actor: str
    description: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


# ============================================================================
# Memory Cycle — Multi-session LLM memory lifecycle test
# ============================================================================


class MemoryCycleRequest(BaseModel):
    """Request to run a full memory lifecycle test."""

    statements: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Facts to store as memories",
    )
    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Questions to ask each model",
    )
    models: list[str] = Field(
        default=["claude", "openai"],
        max_length=3,
        description="LLM models to test",
    )
    namespace: str = Field(
        default="cycle-test",
        min_length=1,
        max_length=MAX_NAMESPACE_LEN,
    )

    @field_validator("namespace")
    @classmethod
    def ns_valid(cls, v: str) -> str:
        return _validate_namespace(v)


class UsageResponse(BaseModel):
    """Tier-aware usage response for /v1/usage and /v1/billing/usage."""

    tier: str = Field(description="Current tier: free, pro, or unlimited")
    period: str = Field(description="Billing period YYYY-MM")
    operations: int = Field(description="Total operations this period")
    operations_limit: int = Field(description="Max operations for tier (-1 = unlimited)")
    writes: int = Field(description="Write operations this period")
    reads: int = Field(description="Read operations this period")
    namespaces_limit: int = Field(description="Max namespaces for tier (-1 = unlimited)")
    storage_limit: int = Field(description="Max L1 items per namespace for tier")
    rate_limit: int = Field(description="Requests per minute for tier")


# =========================================================================
# User auth models
# =========================================================================

class UserRegisterRequest(BaseModel):
    """Register a new user with email and password."""

    email: str = Field(..., max_length=256)
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(default="", max_length=128)


class UserLoginRequest(BaseModel):
    """Login with email and password."""

    email: str = Field(..., max_length=256)
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    """Public user profile (no password_hash)."""

    id: str
    email: str
    name: str
    tier: str
    is_admin: bool
    avatar_url: Optional[str] = None
    created_at: str


class TierUpgradeRequest(BaseModel):
    """Request to change user tier."""

    tier: str = Field(..., pattern="^(free|pro|business|enterprise)$")


class UserProfileUpdateRequest(BaseModel):
    """Request to update user profile fields."""

    name: Optional[str] = Field(default=None, max_length=128)
    email: Optional[str] = Field(default=None, max_length=256)
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)
    current_password: Optional[str] = Field(default=None, max_length=128)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Ingestion — Cross-LLM Context Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class PromptEntryModel(BaseModel):
    """A single prompt/response in a conversation."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    sequence_num: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptIngestRequest(BaseModel):
    """Batch ingest prompts from an LLM session.

    Called by Claude Code hooks, browser extension, SDKs.
    Supports batch ingestion for efficiency (single HTTP call per hook fire).
    """
    session_id: str = Field(..., min_length=1, max_length=128)
    entries: list[PromptEntryModel] = Field(..., min_length=1, max_length=200)
    llm_provider: str = Field(default="unknown", max_length=64)
    llm_model: Optional[str] = Field(default=None, max_length=128)
    client_type: str = Field(default="hook", max_length=32)
    namespace: str = Field(default="default", min_length=1, max_length=MAX_NAMESPACE_LEN)

    @field_validator("namespace")
    @classmethod
    def ns_valid(cls, v: str) -> str:
        return _validate_namespace(v)
