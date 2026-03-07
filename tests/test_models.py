"""Model validation, serialization, deserialization, edge cases, boundary tests."""

import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from clsplusplus.models import (
    MAX_LIMIT,
    MAX_METADATA_KEYS,
    MAX_NAMESPACE_LEN,
    MAX_QUERY_LEN,
    MAX_TEXT_LEN,
    AdjudicateRequest,
    DemoChatRequest,
    ForgetRequest,
    HealthResponse,
    MemoryItem,
    ReadRequest,
    ReadResponse,
    StoreLevel,
    WriteRequest,
    _validate_item_id,
    _validate_namespace,
)


# ---------------------------------------------------------------------------
# StoreLevel enum
# ---------------------------------------------------------------------------

class TestStoreLevel:

    def test_all_levels_exist(self):
        assert StoreLevel.L0 == "L0"
        assert StoreLevel.L1 == "L1"
        assert StoreLevel.L2 == "L2"
        assert StoreLevel.L3 == "L3"

    def test_level_is_string_enum(self):
        assert isinstance(StoreLevel.L0, str)
        assert StoreLevel.L0.value == "L0"

    def test_level_from_value(self):
        assert StoreLevel("L0") == StoreLevel.L0
        assert StoreLevel("L3") == StoreLevel.L3

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            StoreLevel("L4")

    def test_level_iteration(self):
        levels = list(StoreLevel)
        assert len(levels) == 4


# ---------------------------------------------------------------------------
# Namespace validation
# ---------------------------------------------------------------------------

class TestNamespaceValidation:

    def test_valid_namespace(self):
        assert _validate_namespace("default") == "default"
        assert _validate_namespace("user-123") == "user-123"
        assert _validate_namespace("agent_v2") == "agent_v2"
        assert _validate_namespace("a") == "a"

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError, match="1-64 chars"):
            _validate_namespace("")

    def test_too_long_namespace_raises(self):
        with pytest.raises(ValueError, match="1-64 chars"):
            _validate_namespace("a" * 65)

    def test_max_length_namespace_ok(self):
        assert _validate_namespace("a" * 64) == "a" * 64

    def test_special_chars_rejected(self):
        bad = ["bad name!", "ns with spaces", "ns/path", "ns@email", "ns.dot", "ns:colon", "ns;semi"]
        for ns in bad:
            with pytest.raises(ValueError, match="alphanumeric"):
                _validate_namespace(ns)

    def test_unicode_rejected(self):
        # Use a non-alphanumeric unicode character (em dash) since accented
        # chars like 'e' pass str.isalnum() in Python
        with pytest.raises(ValueError):
            _validate_namespace("ns\u2014")

    def test_sql_injection_in_namespace(self):
        with pytest.raises(ValueError):
            _validate_namespace("'; DROP TABLE--")

    def test_xss_in_namespace(self):
        with pytest.raises(ValueError):
            _validate_namespace("<script>alert(1)</script>")


# ---------------------------------------------------------------------------
# Item ID validation
# ---------------------------------------------------------------------------

class TestItemIdValidation:

    def test_valid_item_id(self):
        assert _validate_item_id("abc123") == "abc123"
        assert _validate_item_id("item-1") == "item-1"
        assert _validate_item_id("item_2") == "item_2"

    def test_uuid_as_item_id(self):
        uid = str(uuid4())
        # UUIDs have hyphens, which are allowed
        assert _validate_item_id(uid) == uid

    def test_empty_item_id_raises(self):
        with pytest.raises(ValueError, match="1-64 chars"):
            _validate_item_id("")

    def test_too_long_item_id_raises(self):
        with pytest.raises(ValueError, match="1-64 chars"):
            _validate_item_id("x" * 65)

    def test_path_traversal_rejected(self):
        with pytest.raises(ValueError):
            _validate_item_id("../../../etc/passwd")

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError):
            _validate_item_id("item\x00id")


# ---------------------------------------------------------------------------
# MemoryItem
# ---------------------------------------------------------------------------

class TestMemoryItem:

    def test_defaults(self):
        item = MemoryItem(text="hello")
        assert item.text == "hello"
        assert item.namespace == "default"
        assert item.store_level == StoreLevel.L0
        assert item.source == "user"
        assert item.confidence == 0.5
        assert item.salience == 0.5
        assert item.authority == 0.5
        assert item.usage_count == 0
        assert item.conflict_score == 0.0
        assert item.surprise == 0.0
        assert item.promotion_score == 0.0
        assert item.version == 1
        assert item.lineage == []
        assert item.metadata == {}
        assert item.embedding is None
        assert item.subject is None
        assert item.predicate is None
        assert item.object is None

    def test_auto_generated_id(self):
        item1 = MemoryItem(text="a")
        item2 = MemoryItem(text="b")
        assert item1.id != item2.id
        assert len(item1.id) == 36  # UUID format

    def test_custom_id(self):
        item = MemoryItem(id="custom-id", text="x")
        assert item.id == "custom-id"

    def test_to_dict(self):
        item = MemoryItem(text="test", namespace="ns1")
        d = item.to_dict()
        assert d["text"] == "test"
        assert d["namespace"] == "ns1"
        assert isinstance(d["timestamp"], str)  # ISO format

    def test_from_dict(self):
        d = {
            "id": "test-id",
            "text": "hello",
            "namespace": "ns1",
            "timestamp": "2024-01-01T00:00:00",
            "store_level": "L0",
        }
        item = MemoryItem.from_dict(d)
        assert item.id == "test-id"
        assert item.text == "hello"

    def test_from_dict_with_z_suffix(self):
        d = {"text": "x", "timestamp": "2024-01-01T00:00:00Z"}
        item = MemoryItem.from_dict(d)
        assert item.timestamp.year == 2024

    def test_roundtrip_serialization(self):
        item = MemoryItem(
            text="round trip",
            namespace="test",
            salience=0.9,
            embedding=[0.1] * 384,
            metadata={"key": "value"},
        )
        d = item.to_dict()
        json_str = json.dumps(d, default=str)
        restored = MemoryItem.from_dict(json.loads(json_str))
        assert restored.text == item.text
        assert restored.salience == item.salience
        assert len(restored.embedding) == 384

    def test_large_embedding(self):
        item = MemoryItem(text="x", embedding=[0.0] * 384)
        assert len(item.embedding) == 384

    def test_rdf_triple(self):
        item = MemoryItem(text="x", subject="France", predicate="capital", object="Paris")
        assert item.subject == "France"


# ---------------------------------------------------------------------------
# WriteRequest
# ---------------------------------------------------------------------------

class TestWriteRequest:

    def test_valid_request(self):
        req = WriteRequest(text="hello", namespace="default")
        assert req.text == "hello"

    def test_min_text_length(self):
        req = WriteRequest(text="x")
        assert req.text == "x"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            WriteRequest(text="")

    def test_max_text_length(self):
        req = WriteRequest(text="x" * MAX_TEXT_LEN)
        assert len(req.text) == MAX_TEXT_LEN

    def test_over_max_text_rejected(self):
        with pytest.raises(ValidationError):
            WriteRequest(text="x" * (MAX_TEXT_LEN + 1))

    def test_salience_bounds(self):
        WriteRequest(text="x", salience=0.0)
        WriteRequest(text="x", salience=1.0)
        with pytest.raises(ValidationError):
            WriteRequest(text="x", salience=-0.1)
        with pytest.raises(ValidationError):
            WriteRequest(text="x", salience=1.1)

    def test_authority_bounds(self):
        WriteRequest(text="x", authority=0.0)
        WriteRequest(text="x", authority=1.0)
        with pytest.raises(ValidationError):
            WriteRequest(text="x", authority=-0.1)
        with pytest.raises(ValidationError):
            WriteRequest(text="x", authority=1.1)

    def test_namespace_validation_in_request(self):
        with pytest.raises(ValidationError):
            WriteRequest(text="x", namespace="bad name!")

    def test_metadata_dict(self):
        req = WriteRequest(text="x", metadata={"key": "value"})
        assert req.metadata["key"] == "value"

    def test_subject_predicate_object(self):
        req = WriteRequest(text="x", subject="A", predicate="B", object="C")
        assert req.subject == "A"

    def test_source_max_length(self):
        req = WriteRequest(text="x", source="x" * 64)
        assert len(req.source) == 64


# ---------------------------------------------------------------------------
# ReadRequest
# ---------------------------------------------------------------------------

class TestReadRequest:

    def test_valid_request(self):
        req = ReadRequest(query="test")
        assert req.query == "test"
        assert req.limit == 10
        assert req.min_confidence == 0.0

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            ReadRequest(query="")

    def test_max_query_length(self):
        req = ReadRequest(query="x" * MAX_QUERY_LEN)
        assert len(req.query) == MAX_QUERY_LEN

    def test_over_max_query_rejected(self):
        with pytest.raises(ValidationError):
            ReadRequest(query="x" * (MAX_QUERY_LEN + 1))

    def test_limit_bounds(self):
        ReadRequest(query="x", limit=1)
        ReadRequest(query="x", limit=MAX_LIMIT)
        with pytest.raises(ValidationError):
            ReadRequest(query="x", limit=0)
        with pytest.raises(ValidationError):
            ReadRequest(query="x", limit=MAX_LIMIT + 1)

    def test_min_confidence_bounds(self):
        ReadRequest(query="x", min_confidence=0.0)
        ReadRequest(query="x", min_confidence=1.0)
        with pytest.raises(ValidationError):
            ReadRequest(query="x", min_confidence=-0.1)
        with pytest.raises(ValidationError):
            ReadRequest(query="x", min_confidence=1.1)

    def test_store_levels_filter(self):
        req = ReadRequest(query="x", store_levels=[StoreLevel.L1, StoreLevel.L2])
        assert len(req.store_levels) == 2

    def test_all_levels_none_default(self):
        req = ReadRequest(query="x")
        assert req.store_levels is None


# ---------------------------------------------------------------------------
# ForgetRequest
# ---------------------------------------------------------------------------

class TestForgetRequest:

    def test_valid_request(self):
        req = ForgetRequest(item_id="abc123", namespace="default")
        assert req.item_id == "abc123"

    def test_invalid_item_id(self):
        with pytest.raises(ValidationError):
            ForgetRequest(item_id="bad/id", namespace="default")

    def test_invalid_namespace(self):
        with pytest.raises(ValidationError):
            ForgetRequest(item_id="abc", namespace="bad space")


# ---------------------------------------------------------------------------
# AdjudicateRequest
# ---------------------------------------------------------------------------

class TestAdjudicateRequest:

    def test_valid_request(self):
        req = AdjudicateRequest(new_fact="Earth is round", evidence=["science"])
        assert req.new_fact == "Earth is round"

    def test_empty_fact_rejected(self):
        with pytest.raises(ValidationError):
            AdjudicateRequest(new_fact="", evidence=[])

    def test_namespace_validation(self):
        """Cover line 165: AdjudicateRequest namespace validator."""
        with pytest.raises(ValidationError):
            AdjudicateRequest(new_fact="test", evidence=[], namespace="bad name!")

    def test_valid_namespace(self):
        req = AdjudicateRequest(new_fact="test", evidence=[], namespace="valid-ns")
        assert req.namespace == "valid-ns"


# ---------------------------------------------------------------------------
# DemoChatRequest
# ---------------------------------------------------------------------------

class TestDemoChatRequest:

    def test_valid_request(self):
        req = DemoChatRequest(model="claude", message="Hello")
        assert req.model == "claude"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            DemoChatRequest(model="claude", message="")

    def test_namespace_default(self):
        req = DemoChatRequest(model="claude", message="hi")
        assert req.namespace == "demo-default"

    def test_namespace_validation(self):
        """Cover line 178: DemoChatRequest namespace validator."""
        with pytest.raises(ValidationError):
            DemoChatRequest(model="claude", message="hi", namespace="bad space!")

    def test_custom_namespace(self):
        req = DemoChatRequest(model="claude", message="hi", namespace="custom-ns")
        assert req.namespace == "custom-ns"


# ---------------------------------------------------------------------------
# ReadResponse
# ---------------------------------------------------------------------------

class TestReadResponse:

    def test_empty_response(self):
        resp = ReadResponse(items=[], query="test", namespace="default")
        assert len(resp.items) == 0

    def test_response_with_items(self):
        items = [MemoryItem(text=f"item{i}") for i in range(3)]
        resp = ReadResponse(items=items, query="test", namespace="default")
        assert len(resp.items) == 3


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------

class TestHealthResponse:

    def test_default_health(self):
        h = HealthResponse(stores={})
        assert h.status == "healthy"
        assert h.version == "0.1.0"

    def test_degraded_health(self):
        h = HealthResponse(status="degraded", stores={"L0": {"status": "unhealthy"}})
        assert h.status == "degraded"
