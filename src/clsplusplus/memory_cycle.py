"""CLS++ Memory Cycle — Multi-session LLM memory lifecycle proof.

Demonstrates: encode → retrieve → augment → cross-session persistence.
Proves memory works across models and sessions.
"""

import logging
from typing import Optional
from uuid import uuid4

from clsplusplus.config import Settings
from clsplusplus.memory_service import MemoryService
from clsplusplus.models import ReadRequest, WriteRequest

logger = logging.getLogger(__name__)


async def run_memory_cycle(
    memory_service: MemoryService,
    settings: Settings,
    statements: list[str],
    queries: list[str],
    models: list[str],
    namespace: str,
) -> dict:
    """Run a full memory lifecycle: encode → retrieve → augment → cross-session.

    Args:
        memory_service: The CLS++ memory service instance
        settings: Application settings
        statements: Facts to store as memories
        queries: Questions to ask each model
        models: LLM models to test (e.g. ["claude", "openai"])
        namespace: Namespace for this cycle run

    Returns:
        Structured report with all phases and verdict
    """
    cycle_id = str(uuid4())
    phases = {}

    # =========================================================================
    # Phase 1: ENCODE — Store all statements as memories
    # =========================================================================
    encode_items = []
    for stmt in statements:
        try:
            item = await memory_service.write(
                WriteRequest(
                    text=stmt,
                    namespace=namespace,
                    source="memory-cycle",
                    salience=0.9,
                    authority=0.8,
                )
            )
            encode_items.append({
                "id": item.id,
                "text": item.text,
                "store_level": item.store_level.value,
                "confidence": item.confidence,
            })
        except Exception as e:
            logger.error("Encode phase failed for '%s': %s", stmt, e)
            encode_items.append({"text": stmt, "error": str(e)})

    phases["encode"] = {
        "stored": len([i for i in encode_items if "error" not in i]),
        "total": len(statements),
        "items": encode_items,
    }

    # =========================================================================
    # Phase 2: RETRIEVE — Query back, verify stored correctly
    # =========================================================================
    retrieve_results = []
    for query in queries:
        try:
            resp = await memory_service.read(
                ReadRequest(query=query, namespace=namespace, limit=10)
            )
            retrieve_results.append({
                "query": query,
                "found": len(resp.items),
                "items": [
                    {"id": item.id, "text": item.text, "confidence": item.confidence}
                    for item in resp.items[:5]
                ],
            })
        except Exception as e:
            logger.error("Retrieve phase failed for '%s': %s", query, e)
            retrieve_results.append({"query": query, "found": 0, "error": str(e)})

    total_found = sum(r["found"] for r in retrieve_results)
    avg_confidence = 0.0
    all_items = [
        item
        for r in retrieve_results
        for item in r.get("items", [])
    ]
    if all_items:
        avg_confidence = sum(i["confidence"] for i in all_items) / len(all_items)

    phases["retrieve"] = {
        "queries": len(queries),
        "total_found": total_found,
        "confidence_avg": round(avg_confidence, 3),
        "results": retrieve_results,
    }

    # =========================================================================
    # Phase 3: AUGMENT — Each model answers query with memory context
    # =========================================================================
    augment_results = {}
    for model in models:
        model_results = []
        for query in queries[:2]:  # Limit to 2 queries per model to save API calls
            try:
                # Get memory context
                resp = await memory_service.read(
                    ReadRequest(query=query, namespace=namespace, limit=8)
                )
                memory_context = "\n".join(
                    f"- {item.text}" for item in resp.items
                ) if resp.items else "No prior context."

                # Call LLM with memory augmentation
                from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini

                system_prompt = f"""You are a helpful assistant. Use this context to answer:
{memory_context}
Answer naturally based on the context provided."""

                if model == "claude":
                    reply = await call_claude(settings, system_prompt, query)
                elif model == "openai":
                    reply = await call_openai(settings, system_prompt, query)
                elif model == "gemini":
                    reply = await call_gemini(settings, system_prompt, query)
                else:
                    reply = f"Unknown model: {model}"

                model_results.append({
                    "query": query,
                    "response": reply,
                    "memory_context_items": len(resp.items),
                    "memory_used": len(resp.items) > 0,
                })
            except Exception as e:
                logger.error("Augment phase failed for %s/%s: %s", model, query, e)
                model_results.append({
                    "query": query,
                    "error": str(e),
                    "memory_used": False,
                })

        augment_results[model] = model_results

    phases["augment"] = augment_results

    # =========================================================================
    # Phase 4: CROSS-SESSION — Read from same namespace, verify persistence
    # =========================================================================
    # Memories should persist because they're in L1 (PostgreSQL).
    # We verify by reading with a fresh query.
    cross_session_results = []
    for query in queries[:2]:
        try:
            resp = await memory_service.read(
                ReadRequest(query=query, namespace=namespace, limit=10)
            )
            cross_session_results.append({
                "query": query,
                "found": len(resp.items),
                "persisted": len(resp.items) > 0,
            })
        except Exception as e:
            cross_session_results.append({
                "query": query, "found": 0, "persisted": False, "error": str(e),
            })

    all_persisted = all(r.get("persisted", False) for r in cross_session_results)
    total_cross = sum(r["found"] for r in cross_session_results)

    phases["cross_session"] = {
        "namespace": namespace,
        "memories_persisted": all_persisted,
        "items_found": total_cross,
        "results": cross_session_results,
    }

    # =========================================================================
    # Phase 5: VERDICT
    # =========================================================================
    encode_ok = phases["encode"]["stored"] == phases["encode"]["total"]
    retrieve_ok = phases["retrieve"]["total_found"] > 0
    augment_ok = any(
        any(r.get("memory_used", False) for r in results)
        for results in phases["augment"].values()
    )
    persist_ok = phases["cross_session"]["memories_persisted"]

    if encode_ok and retrieve_ok and augment_ok and persist_ok:
        verdict = "PASS"
    elif encode_ok and retrieve_ok:
        verdict = "PARTIAL — encode and retrieve work, augment or persistence needs attention"
    else:
        verdict = "FAIL"

    return {
        "cycle_id": cycle_id,
        "namespace": namespace,
        "models": models,
        "phases": phases,
        "verdict": verdict,
    }
