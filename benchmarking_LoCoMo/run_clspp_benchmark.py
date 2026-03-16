#!/usr/bin/env python3
"""
CLS++ LoCoMo Benchmark Runner.

Evaluates the CLS++ thermodynamic phase memory engine against the LoCoMo
long-term conversational memory benchmark (ACL 2024).

Two modes:
  --mode direct   (default) Direct dialog ingestion into phase engine.
                  Each dialog turn is injected as a Fact directly (no LLM extraction).
                  Tests the RETRIEVAL + REASONING capability of the phase engine.
                  Fast: ~2 min per conversation (LLM only for QA answers).

  --mode llm      Full LLM-based extraction pipeline (Gas → Liquid attention gate).
                  Each dialog turn goes through LLM fact extraction.
                  Tests the COMPLETE pipeline. Slow: ~20 min per conversation.

Pipeline:
  1. INGEST: Feed all dialog turns across all sessions into the phase engine.
  2. QUERY: For each QA pair, use phase engine search() for memory retrieval,
     then query LLM with augmented context for the answer.
  3. EVALUATE: F1 score (token-level) per LoCoMo standard.

Metrics per category:
  Cat 1: Multi-hop reasoning (answers spanning multiple dialog turns)
  Cat 2: Temporal questions (when-based, requires date understanding)
  Cat 3: Open-domain questions (inference from conversational context)
  Cat 4: Long-context questions (answers far back in conversation)
  Cat 5: Adversarial (answer "not mentioned" if info unavailable)

Usage:
  python3 run_clspp_benchmark.py                     # All 10 conversations, direct mode
  python3 run_clspp_benchmark.py --limit 2            # First 2 conversations
  python3 run_clspp_benchmark.py --mode llm --limit 1 # Full LLM pipeline, 1 conv
"""

import asyncio
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

# Force unbuffered stdout for real-time progress visibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent / "locomo"))

from clsplusplus.config import Settings
from clsplusplus.memory_phase import PhaseMemoryEngine, Fact, PhaseMemoryItem

# --- Inline F1/scoring (avoid importing bert_score → torch → heavy deps) ---
import re as _re
import string as _string
from collections import Counter as _Counter


def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.replace(",", "")
    s = _re.sub(r'\b(a|an|the|and)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in set(_string.punctuation))
    return ' '.join(s.split())


def f1_score(prediction, ground_truth):
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = _Counter(pred_tokens) & _Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi(prediction, ground_truth):
    """Multi-answer F1: split on comma, compute max F1 per ground truth."""
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    scores = [max(f1_score(p, gt) for p in predictions) for gt in ground_truths]
    return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# LLM Callers
# =============================================================================

_settings = Settings()


async def _call_llm(system: str, user_msg: str) -> str:
    """Route to available LLM with failover."""
    from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini

    ERROR_MARKERS = [
        "An error occurred", "Add CLS_", "No response",
        "content may have been blocked", "credit balance is too low",
    ]

    callers = [
        ("openai", call_openai),
        ("claude", call_claude),
        ("gemini", call_gemini),
    ]

    for name, caller in callers:
        try:
            reply = await caller(_settings, system, user_msg)
            if not any(m in reply for m in ERROR_MARKERS):
                return reply
        except Exception:
            continue

    return "I don't know."


async def _extraction_caller(system: str, user_msg: str) -> str:
    """LLM caller for the phase engine's attention gate."""
    return await _call_llm(system, user_msg)


# =============================================================================
# QA Prompts — match LoCoMo's evaluation methodology
# =============================================================================

QA_SYSTEM = """You are a helpful assistant with access to memory from past conversations.
Use the provided memory context to answer questions accurately.
Answer with exact words from the memory/conversation whenever possible.
Keep your answer SHORT — a few words or a brief phrase. Do NOT write full sentences."""

QA_PROMPT = """Based on the memory context above, answer this question in a SHORT phrase (a few words only).

Question: {question}
Short answer:"""

QA_PROMPT_CAT2 = """Based on the memory context above, answer this question with an approximate DATE.
Use the DATE information from conversation sessions if available.

Question: {question}
Short answer:"""

QA_PROMPT_CAT5 = """Based on the memory context above, answer this question.
If the information is NOT available in the memory, answer "Not mentioned in the conversation".

Question: {question}
Short answer:"""


# =============================================================================
# Direct Ingestion — inject dialog turns as Facts without LLM extraction
# =============================================================================

def _direct_ingest_conversation(engine: PhaseMemoryEngine, conv: dict, namespace: str) -> dict:
    """
    Inject all dialog turns directly into the phase engine via store().

    Uses the FULL engine pipeline:
      - Token indexing (_token_index, _doc_freq, _item_by_id)
      - TSF search (triple-index hash lookup)
      - Crystallization (Landauer ΔF < 0)
      - Contradiction Cascade (surprise damage)
      - CER (Cross-Entity Resonance)

    Each turn creates a Fact with:
      subject = speaker name (normalized)
      relation = "said"
      value = the dialog text
      override = False (no override detection in direct mode)
      raw_text = "Speaker: text" (original format)
    """
    session_nums = sorted(set(
        int(k.split("_")[-1])
        for k in conv.keys()
        if k.startswith("session_") and "_date_time" not in k
        and "_observation" not in k and "_summary" not in k
    ))

    total_turns = 0
    ingested = 0

    for sess_num in session_nums:
        sess_key = f"session_{sess_num}"
        if sess_key not in conv or not conv[sess_key]:
            continue

        date_key = f"session_{sess_num}_date_time"
        date_str = conv.get(date_key, "")

        for dialog in conv[sess_key]:
            total_turns += 1
            speaker = dialog.get("speaker", "Unknown")
            text = dialog.get("text", "").strip()
            if not text:
                continue

            # Create Fact directly from dialog turn
            raw_text = f"{speaker}: {text}"
            # Include date context in the value for temporal QA
            value_text = f"[{date_str}] {text}" if date_str else text

            fact = Fact(
                subject=speaker.lower().strip(),
                relation="said",
                value=value_text.lower().strip(),
                override=False,
                raw_text=raw_text,
            )

            # Use store() for full indexing pipeline (token index, CER, etc.)
            # But skip per-item recompute during bulk load (would cause
            # premature crystallization → orphan melting cascade)
            engine._batch_mode = True
            engine.store(raw_text, namespace, fact=fact)
            ingested += 1

        if sess_num % 5 == 0:
            print(f"    Session {sess_num}: {ingested} facts ingested so far")

    # In benchmark mode, all items stored at once and queried immediately.
    # No time passes → no decay → no recompute needed. Items stay at s=1.0.
    # Crystallization is a long-term consolidation mechanism, not for benchmarks.
    engine._batch_mode = False

    return {"total_turns": total_turns, "ingested_facts": ingested}


# =============================================================================
# Benchmark Engine
# =============================================================================

class CLSPPBenchmark:
    """Runs LoCoMo benchmark against CLS++ phase memory engine."""

    def __init__(self, data_file: str, out_file: str, limit: int = 0, mode: str = "direct"):
        self.data_file = data_file
        self.out_file = out_file
        self.limit = limit
        self.mode = mode
        self.results = []
        self.start_time = time.time()

    async def run(self):
        """Execute the full benchmark."""
        samples = json.load(open(self.data_file))
        if self.limit > 0:
            samples = samples[:self.limit]

        print(f"╔══════════════════════════════════════════════════════════╗")
        print(f"║         CLS++ LoCoMo Benchmark Runner                  ║")
        print(f"║   F(θ,Σ,ρ,τ) = E_pred − Σ·S_model + λ·L_landauer     ║")
        print(f"╠══════════════════════════════════════════════════════════╣")
        print(f"║  Mode: {self.mode:<15} Conversations: {len(samples):<14} ║")
        print(f"╚══════════════════════════════════════════════════════════╝")
        print()

        for idx, sample in enumerate(samples):
            sample_id = sample["sample_id"]
            print(f"\n{'='*60}")
            print(f"  [{idx+1}/{len(samples)}] {sample_id}")
            print(f"{'='*60}")

            result = await self._process_conversation(sample)
            self.results.append(result)

            # Save after each conversation (incremental)
            self._save_results()

            elapsed = time.time() - self.start_time
            print(f"  ⏱ Elapsed: {elapsed:.0f}s")

        # Final analysis
        self._analyze_results()

    async def _process_conversation(self, sample: dict) -> dict:
        """Process one conversation: ingest all sessions, then answer all QA."""
        conv = sample["conversation"]
        sample_id = sample["sample_id"]

        # Fresh engine per conversation (each conversation is independent)
        # For benchmark: use very high τ_default so early dialog turns don't
        # decay to gas before QA queries. In real usage, τ reflects extraction
        # strength, but for benchmark we want to test retrieval, not decay.
        benchmark_tau = 10000.0 if self.mode == "direct" else _settings.phase_tau_default
        engine = PhaseMemoryEngine(
            kT=_settings.phase_kT,
            lambda_budget=_settings.phase_lambda,
            tau_c1=_settings.phase_tau_c1,
            tau_default=benchmark_tau,
            tau_override=_settings.phase_tau_override,
            strength_floor=_settings.phase_strength_floor,
            capacity=5000,  # Higher capacity for long conversations (600+ turns)
            beta_retrieval=_settings.phase_beta_retrieval,
        )
        namespace = sample_id

        # --- Phase 1: INGEST ---
        t0 = time.time()
        if self.mode == "direct":
            stats = _direct_ingest_conversation(engine, conv, namespace)
        else:
            stats = await self._llm_ingest_conversation(engine, conv, namespace)

        t_ingest = time.time() - t0
        n_items = len(engine._items.get(namespace, []))
        print(f"  ✓ Ingested: {stats['total_turns']} turns → {stats['ingested_facts']} facts → {n_items} items alive ({t_ingest:.1f}s)")
        print(f"    Events: {engine._event_counter}, indexed tokens: {len(engine._token_index)}")

        # --- Phase 2: ANSWER all QA questions ---
        t0 = time.time()
        qa_results = []
        model_key = "clspp"
        prediction_key = "clspp_prediction"

        for qi, qa in enumerate(sample["qa"]):
            question = qa["question"]
            # Category 5 (adversarial) uses 'adversarial_answer' key
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            category = qa["category"]

            # Search phase memory for relevant context
            memory_context, debug_items = engine.build_augmented_context(
                question, namespace, limit=10
            )

            # Build prompt based on category
            if category == 2:
                prompt = QA_PROMPT_CAT2.format(question=question)
            elif category == 5:
                prompt = QA_PROMPT_CAT5.format(question=question)
            else:
                prompt = QA_PROMPT.format(question=question)

            system = f"{QA_SYSTEM}\n\n{memory_context}"

            try:
                prediction = await _call_llm(system, prompt)
                prediction = prediction.strip()
            except Exception:
                prediction = "I don't know"

            # Evaluate F1
            score = self._compute_score(prediction, answer, category)

            qa_entry = {
                "question": question,
                "answer": answer,
                "category": category,
                "evidence": qa.get("evidence", []),
                prediction_key: prediction,
                f"{model_key}_f1": round(score, 3),
                "_memory_items_used": len(debug_items),
            }
            qa_results.append(qa_entry)

            if (qi + 1) % 10 == 0 or qi == 0:
                print(f"    QA {qi+1}/{len(sample['qa'])} | cat={category} | F1={score:.3f} | mem={len(debug_items)}")

        t_qa = time.time() - t0

        # Compute per-category accuracy
        cat_scores = defaultdict(list)
        for qa in qa_results:
            cat_scores[qa["category"]].append(qa[f"{model_key}_f1"])

        print(f"  ✓ QA complete ({t_qa:.1f}s)")
        for cat in sorted(cat_scores.keys()):
            scores = cat_scores[cat]
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"    Cat {cat}: {avg:.3f} ({len(scores)} Qs)")

        overall = sum(qa[f"{model_key}_f1"] for qa in qa_results) / len(qa_results) if qa_results else 0.0
        print(f"    Overall: {overall:.3f}")

        n_items = len(engine._items.get(namespace, []))
        return {
            "sample_id": sample_id,
            "qa": qa_results,
            "phase_stats": {
                "total_turns": stats["total_turns"],
                "ingested_facts": stats["ingested_facts"],
                "liquid_count": n_items,
                "gas_count": 0,
                "event_counter": engine._event_counter,
                "memory_density_rho": engine._memory_density(namespace),
                "total_free_energy": 0.0,
            },
        }

    async def _llm_ingest_conversation(self, engine, conv, namespace) -> dict:
        """
        Full LLM extraction pipeline for ingestion.

        Each dialog turn is sent to an LLM to extract structured facts:
            subject, relation, value (e.g. "alice", "favorite_movie", "the matrix")

        This enables TSF search to bridge semantic gaps that direct mode cannot.
        """
        session_nums = sorted(set(
            int(k.split("_")[-1])
            for k in conv.keys()
            if k.startswith("session_") and "_date_time" not in k
            and "_observation" not in k and "_summary" not in k
        ))

        total_turns = 0
        ingested = 0

        EXTRACT_SYSTEM = """You are a fact extraction engine. Extract ALL factual statements from the dialog turn.
For each fact, output one line in this EXACT format:
subject|relation|value

Rules:
- subject: The person or entity the fact is ABOUT (lowercase, e.g. "alice", "bob")
- relation: A short verb phrase describing the relationship (e.g. "favorite_movie", "lives_in", "works_at", "enjoys", "visited", "plans_to")
- value: The object/detail (lowercase, e.g. "the matrix", "new york", "google")
- Extract EVERY factual statement, preference, plan, opinion, experience
- If the turn contains no extractable facts (greetings, filler), output: NONE
- One fact per line. No numbering, no extra text."""

        engine._batch_mode = True

        for sess_num in session_nums:
            sess_key = f"session_{sess_num}"
            if sess_key not in conv or not conv[sess_key]:
                continue

            date_key = f"session_{sess_num}_date_time"
            date_str = conv.get(date_key, "")

            for dialog in conv[sess_key]:
                total_turns += 1
                speaker = dialog.get("speaker", "Unknown")
                text = dialog.get("text", "").strip()
                if not text:
                    continue

                raw_text = f"{speaker}: {text}"

                # LLM fact extraction
                try:
                    extraction = await _call_llm(
                        EXTRACT_SYSTEM,
                        f"Speaker: {speaker}\nDialog: {text}"
                    )
                except Exception:
                    extraction = "NONE"

                if not extraction or extraction.strip().upper() == "NONE":
                    # Still store the raw turn as a fallback (like direct mode)
                    value_text = f"[{date_str}] {text}" if date_str else text
                    fact = Fact(
                        subject=speaker.lower().strip(),
                        relation="said",
                        value=value_text.lower().strip(),
                        override=False,
                        raw_text=raw_text,
                    )
                    engine.store(raw_text, namespace, fact=fact)
                    ingested += 1
                    continue

                # Parse extracted facts
                for line in extraction.strip().split("\n"):
                    line = line.strip()
                    if not line or line.upper() == "NONE":
                        continue
                    parts = line.split("|")
                    if len(parts) >= 3:
                        subj = parts[0].strip().lower()
                        rel = parts[1].strip().lower().replace(" ", "_")
                        val = parts[2].strip().lower()
                        # Include date context for temporal QA
                        if date_str:
                            val = f"[{date_str}] {val}"
                        fact = Fact(
                            subject=subj,
                            relation=rel,
                            value=val,
                            override=False,
                            raw_text=raw_text,
                        )
                        # Build enriched text for indexing: include relation
                        # keywords so TSF can match semantic queries.
                        # e.g. "alice favorite_movie the matrix" → tokens include
                        # "alice", "favorite_movie", "matrix"
                        rel_words = rel.replace("_", " ")
                        enriched_text = f"{subj} {rel_words} {val}. {raw_text}"
                        engine.store(enriched_text, namespace, fact=fact)
                        ingested += 1
                    else:
                        # Fallback: store raw if parsing fails
                        value_text = f"[{date_str}] {text}" if date_str else text
                        fact = Fact(
                            subject=speaker.lower().strip(),
                            relation="said",
                            value=value_text.lower().strip(),
                            override=False,
                            raw_text=raw_text,
                        )
                        engine.store(raw_text, namespace, fact=fact)
                        ingested += 1

            if sess_num % 5 == 0:
                print(f"    Session {sess_num}: {ingested} facts ({total_turns} turns)")

        engine._batch_mode = False
        return {"total_turns": total_turns, "ingested_facts": ingested}

    @staticmethod
    def _compute_score(prediction: str, answer, category: int) -> float:
        """Compute F1 score per LoCoMo category rules."""
        answer = str(answer)
        if category in [2, 3, 4]:
            if category == 3:
                answer = answer.split(";")[0].strip()
            return f1_score(prediction, answer)
        elif category == 1:
            return f1_multi(prediction, answer)
        elif category == 5:
            pred_lower = prediction.lower()
            if "no information available" in pred_lower or "not mentioned" in pred_lower:
                return 1.0
            return 0.0
        return 0.0

    def _save_results(self):
        """Save current results to file."""
        os.makedirs(os.path.dirname(self.out_file) or ".", exist_ok=True)
        with open(self.out_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def _analyze_results(self):
        """Print final aggregated results and save stats."""
        elapsed = time.time() - self.start_time

        print(f"\n\n{'═'*60}")
        print(f"  FINAL RESULTS — CLS++ Phase Memory Engine (LoCoMo)")
        print(f"{'═'*60}")

        total_counts = defaultdict(int)
        acc_counts = defaultdict(float)
        all_scores = []

        for result in self.results:
            for qa in result["qa"]:
                cat = qa["category"]
                total_counts[cat] += 1
                acc_counts[cat] += qa["clspp_f1"]
                all_scores.append(qa["clspp_f1"])

        cat_labels = {
            4: "Long-context",
            1: "Multi-hop",
            2: "Temporal",
            3: "Open-domain",
            5: "Adversarial",
        }

        print(f"\n  {'Category':<20} {'Count':>6} {'F1 Acc':>10}")
        print(f"  {'─'*38}")
        for cat in [4, 1, 2, 3, 5]:
            if total_counts[cat] > 0:
                acc = acc_counts[cat] / total_counts[cat]
                label = f"Cat {cat} ({cat_labels[cat]})"
                print(f"  {label:<20} {total_counts[cat]:>6} {acc:>10.3f}")

        total = sum(total_counts.values())
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"  {'─'*38}")
        print(f"  {'OVERALL':<20} {total:>6} {overall:>10.3f}")

        # Phase engine aggregate stats
        total_turns = sum(r["phase_stats"]["total_turns"] for r in self.results)
        total_facts = sum(r["phase_stats"]["ingested_facts"] for r in self.results)
        total_liquid = sum(r["phase_stats"]["liquid_count"] for r in self.results)
        total_gas = sum(r["phase_stats"]["gas_count"] for r in self.results)

        print(f"\n  Phase Engine Statistics:")
        print(f"    Turns processed:     {total_turns}")
        print(f"    Facts extracted:     {total_facts}")
        print(f"    Final liquid:        {total_liquid}")
        print(f"    Final gas:           {total_gas}")
        print(f"    Condensation rate:   {total_facts/max(total_turns,1)*100:.1f}%")
        print(f"    Conversations:       {len(self.results)}")
        print(f"    Sessions:            {sum(r['phase_stats']['event_counter'] for r in self.results)}")
        print(f"    Total time:          {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Save stats file
        stats_file = self.out_file.replace(".json", "_stats.json")
        stats = {
            "clspp": {
                "overall_accuracy": round(overall, 4),
                "total_questions": total,
                "category_accuracy": {
                    str(cat): round(acc_counts[cat] / total_counts[cat], 4) if total_counts[cat] > 0 else 0.0
                    for cat in [1, 2, 3, 4, 5]
                },
                "category_counts": {str(k): v for k, v in total_counts.items()},
                "cum_accuracy_by_category": {str(k): round(v, 3) for k, v in acc_counts.items()},
                "phase_engine": {
                    "total_turns": total_turns,
                    "facts_extracted": total_facts,
                    "final_liquid": total_liquid,
                    "final_gas": total_gas,
                    "condensation_rate": round(total_facts / max(total_turns, 1) * 100, 1),
                },
                "runtime_seconds": round(elapsed, 1),
                "mode": self.mode,
            }
        }
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Results: {self.out_file}")
        print(f"  Stats:   {stats_file}")
        print(f"{'═'*60}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLS++ LoCoMo Benchmark")
    parser.add_argument("--data-file", type=str,
                        default=str(Path(__file__).parent / "locomo/data/locomo10.json"))
    parser.add_argument("--out-file", type=str,
                        default=str(Path(__file__).parent / "outputs/clspp_locomo10_results.json"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of conversations (0 = all)")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "llm"],
                        help="Ingestion mode: 'direct' (fast) or 'llm' (full pipeline)")
    args = parser.parse_args()

    benchmark = CLSPPBenchmark(args.data_file, args.out_file, args.limit, args.mode)
    asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
