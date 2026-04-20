"""CLS++ cost model — per-operation cost constants for margin tracking."""

from __future__ import annotations

# Default costs in USD per operation.
# These represent infrastructure cost, not price charged.
COST_PER_OPERATION = {
    # API operations
    "write": 0.0001,
    "encode": 0.0001,
    "read": 0.00005,
    "retrieve": 0.00005,
    "search": 0.0001,
    "knowledge": 0.00005,
    "delete": 0.00001,
    "adjudication": 0.0005,
    "prewarm": 0.0001,
    "chat_message": 0.0005,
    # Compute
    "embedding": 0.0002,
    "hippocampal_replay": 0.0005,
    "l2_promotion": 0.0003,
    "consolidation": 0.001,
    "context_injection": 0.0001,
    # LLM tokens
    "llm_token_in": 0.000003,       # ~$3/1M tokens (Haiku-class)
    "llm_token_out": 0.000015,      # ~$15/1M tokens
    # Extension backend
    "ext_write": 0.00005,
    "ext_search": 0.00005,
    "ext_context_injection": 0.00008,
    "ext_memory_fetch": 0.00002,
    "ext_delete": 0.00001,
    "ext_download": 0.0,
    "ext_llm_proxy_openai": 0.0005,
    "ext_llm_proxy_anthropic": 0.0005,
    # Storage
    "storage_per_item_month": 0.00001,
}


def compute_cost(metrics: dict) -> float:
    """Given a metrics dict (field -> count), compute total cost in USD."""
    total = 0.0
    for field, count in metrics.items():
        unit_cost = COST_PER_OPERATION.get(field, 0)
        try:
            total += unit_cost * float(count)
        except (ValueError, TypeError):
            pass
    return round(total, 6)
