"""CLS++ Request Tracer — UUID-tagged call graph capture.

Every API request gets a trace_id (UUID). Every internal hop appends
a node to the tree. Query by UUID to see exactly what happened.
"""

import time
import uuid
from collections import deque, OrderedDict
from contextlib import contextmanager
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Hop:
    """One node in the call graph tree."""

    __slots__ = (
        "hop_id", "parent_id", "trace_id",
        "label", "module",
        "started_at", "ended_at",
        "metadata", "children",
    )

    def __init__(
        self,
        hop_id: str,
        parent_id: Optional[str],
        trace_id: str,
        label: str,
        module: str,
        metadata: Optional[dict] = None,
    ):
        self.hop_id = hop_id
        self.parent_id = parent_id
        self.trace_id = trace_id
        self.label = label
        self.module = module
        self.started_at = time.monotonic()
        self.ended_at: Optional[float] = None
        self.metadata: dict = metadata or {}
        self.children: list["Hop"] = []

    @property
    def duration_ms(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "hop_id": self.hop_id,
            "parent_id": self.parent_id,
            "label": self.label,
            "module": self.module,
            "started_at": round(self.started_at * 1000),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


class Trace:
    """Full call graph for one request."""

    __slots__ = ("trace_id", "operation", "created_at", "root", "_hops_by_id")

    def __init__(self, trace_id: str, operation: str):
        self.trace_id = trace_id
        self.operation = operation
        self.created_at = time.time()
        self.root: Optional[Hop] = None
        self._hops_by_id: dict[str, Hop] = {}

    @property
    def total_ms(self) -> Optional[float]:
        if self.root and self.root.duration_ms is not None:
            return self.root.duration_ms
        return None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "created_at": round(self.created_at * 1000),
            "total_ms": self.total_ms,
            "tree": self.root.to_dict() if self.root else None,
        }


# ---------------------------------------------------------------------------
# Tracer (singleton)
# ---------------------------------------------------------------------------

class Tracer:
    """
    Singleton tracer. Keeps the last MAX_TRACES in a ring buffer.
    Thread-safe for async via single-event-loop usage (no lock needed for asyncio).
    """

    MAX_TRACES = 2000

    def __init__(self):
        # OrderedDict acts as ordered ring buffer (pop oldest when full)
        self._traces: OrderedDict[str, Trace] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_trace(self, operation: str, trace_id: Optional[str] = None) -> str:
        """Create a new trace. Returns the trace_id."""
        tid = trace_id or str(uuid.uuid4())
        trace = Trace(tid, operation)
        self._store(tid, trace)
        return tid

    def ensure_trace(self, trace_id: str, operation: str) -> str:
        """Like new_trace but does NOT overwrite an existing trace.

        Use this when a trace may already have been started at a higher layer
        (e.g. the HTTP handler) so the service layer doesn't clobber the root span.
        """
        if trace_id not in self._traces:
            self._store(trace_id, Trace(trace_id, operation))
        return trace_id

    @contextmanager
    def span(self, trace_id: str, label: str, module: str, _parent: Optional[str] = None, **metadata):
        """
        Context manager that creates a hop, attaches it to the tree,
        and records duration.

        Usage:
            with tracer.span(trace_id, "memory_service.write", "memory_service", namespace=ns):
                ...
        """
        trace = self._traces.get(trace_id)
        hop_id = str(uuid.uuid4())

        if trace is None:
            # Trace not found — yield without recording
            yield hop_id
            return

        hop = Hop(
            hop_id=hop_id,
            parent_id=_parent if _parent is not None else self._current_parent(trace),
            trace_id=trace_id,
            label=label,
            module=module,
            metadata=dict(metadata),
        )

        # Attach to tree
        if trace.root is None:
            trace.root = hop
        elif hop.parent_id and hop.parent_id in trace._hops_by_id:
            trace._hops_by_id[hop.parent_id].children.append(hop)
        elif trace.root:
            trace.root.children.append(hop)

        trace._hops_by_id[hop_id] = hop

        # Track current open hop per trace (stack via metadata)
        prev_open = trace._hops_by_id.get("__open__")
        trace._hops_by_id["__open__"] = hop

        try:
            yield hop_id
        except Exception as exc:
            # Record the exception on the span so failures are visible in the trace
            hop.metadata["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            raise
        finally:
            hop.ended_at = time.monotonic()
            # Restore previous open hop
            if prev_open is not None:
                trace._hops_by_id["__open__"] = prev_open
            elif "__open__" in trace._hops_by_id:
                del trace._hops_by_id["__open__"]

    def add_metadata(self, trace_id: str, hop_id: str, **kwargs) -> None:
        """Add metadata to an existing hop (e.g. result counts after an operation)."""
        trace = self._traces.get(trace_id)
        if trace and hop_id in trace._hops_by_id:
            trace._hops_by_id[hop_id].metadata.update(kwargs)

    def get(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    def list_recent(self, limit: int = 50) -> list[dict]:
        items = list(self._traces.values())
        items.reverse()  # newest first
        return [
            {
                "trace_id": t.trace_id,
                "operation": t.operation,
                "created_at": round(t.created_at * 1000),
                "total_ms": t.total_ms,
            }
            for t in items[:limit]
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _store(self, trace_id: str, trace: Trace) -> None:
        if len(self._traces) >= self.MAX_TRACES:
            self._traces.popitem(last=False)  # evict oldest
        self._traces[trace_id] = trace

    def _current_parent(self, trace: Trace) -> Optional[str]:
        """Return the currently open hop id (deepest unclosed span)."""
        open_hop = trace._hops_by_id.get("__open__")
        if open_hop is not None:
            return open_hop.hop_id
        return trace.root.hop_id if trace.root else None


# Global singleton
tracer = Tracer()
