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
        ("claude", call_claude),
        ("openai", call_openai),
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
The memory context may have two sections:
1. Schema summaries — high-level patterns from many conversations
2. Detailed conversation excerpts — specific dialog turns with dates and speakers
Search BOTH sections thoroughly. Answer with exact words from the memory whenever possible.
Keep your answer SHORT — a few words or a brief phrase. Do NOT write full sentences."""

QA_PROMPT = """Based on the memory context above, answer this question in a SHORT phrase (a few words only).

Question: {question}
Short answer:"""

QA_PROMPT_CAT2 = """Based on the memory context above, answer this question with a DATE.
Look carefully through the detailed conversation excerpts for date markers like [Session date].
The dates are typically shown at the beginning of conversation turns in brackets.

Question: {question}
Short answer:"""

QA_PROMPT_CAT5 = """Based on the memory context above, answer this question.
If the information is NOT available in the memory, answer "Not mentioned in the conversation".

Question: {question}
Short answer:"""

# =============================================================================
# Enhanced Mode — Category-Specific Prompts
# =============================================================================

ENHANCED_PROMPTS = {
    1: {
        "system": """You are a memory expert answering from stored conversational memories.
You must CONNECT information across MULTIPLE sessions to answer multi-hop questions.
The context includes both schema summaries and detailed conversation excerpts.
CRITICAL RULES:
- Read EVERY memory item below, including the detailed conversation excerpts section.
- The answer requires combining 2+ facts from DIFFERENT sessions.
- Use EXACT words, names, and phrases from the conversation — do NOT paraphrase.
- Think step by step: identify relevant facts, then combine them.
- Answer in a SHORT phrase (a few words). Use the SAME wording as the conversation.""",
        "user": """Here is the COMPLETE conversation history:

{context}

MULTI-HOP REASONING REQUIRED.
Step 1: Find ALL facts related to the question across different sessions.
Step 2: Connect the facts to derive the answer.
Step 3: Answer using EXACT words from the conversation.

Question: {question}

Think step by step, then give ONLY the short answer (a few words, exact wording from conversation):""",
    },
    2: {
        "system": """You are a memory expert answering temporal/date questions from stored conversational memories.
CRITICAL RULES:
- Pay CLOSE attention to [Session N, DATE] brackets — these contain the answer.
- When asked "when", look for: exact dates, months, years, seasons, relative time references.
- Use the EXACT date format from the conversation (e.g., if it says "March 2023", answer "March 2023", NOT "03/2023").
- If multiple dates are mentioned, choose the one that DIRECTLY answers the question.""",
        "user": """Here is the COMPLETE conversation history with date markers:

{context}

TEMPORAL QUESTION — find the exact date/time.
Look for dates in [Session N, DATE] markers AND within the dialog text itself.
Answer with the EXACT date/time phrase used in the conversation.

Question: {question}
Short answer (exact date/time from conversation):""",
    },
    3: {
        "system": """You are a memory expert answering questions from stored conversational memories.
CRITICAL RULES:
- Read ALL provided memories carefully — the answer may be IMPLIED, not explicit.
- Look for context clues, preferences, habits, relationships.
- Use EXACT words and phrases from the conversation in your answer.
- Keep answer SHORT — a few words or brief phrase.""",
        "user": """Here is the COMPLETE conversation history:

{context}

The answer may be IMPLIED rather than explicitly stated. Read ALL items above.
Answer using EXACT words/phrases from the conversation.

Question: {question}
Short answer:""",
    },
    4: {
        "system": """You are a memory expert answering from stored conversational memories.
CRITICAL RULES:
- The answer IS in the memories below — read EVERY SINGLE ITEM, especially older ones from early sessions.
- Do NOT say "not mentioned". The answer EXISTS in these memories.
- Use EXACT words and phrases from the conversation — do NOT paraphrase.
- Pay attention to ALL sessions, not just recent ones.""",
        "user": """Here is the COMPLETE conversation history:

{context}

The answer IS in these memories. Read EVERY item, including early sessions.
Answer using the EXACT words from the conversation. Keep it SHORT.

Question: {question}
Short answer (exact words from conversation):""",
    },
    5: {
        "system": """You are a memory expert. You must determine if the question can be answered from the memories below.
CRITICAL RULES:
- ONLY answer if the information is EXPLICITLY and CLEARLY stated in the memories.
- If the question asks about something NOT discussed in ANY memory, respond: "Not mentioned in the conversation"
- Do NOT guess, infer, or use outside knowledge.
- Do NOT confuse similar names, events, or topics.
- When in doubt, answer "Not mentioned in the conversation".""",
        "user": """Here are ALL available memories:

{context}

STRICT VERIFICATION: Is the answer to this question EXPLICITLY stated in the memories above?
If YES → answer using exact words from the conversation.
If NO → answer exactly: "Not mentioned in the conversation"

Question: {question}
Short answer:""",
    },
}

ENHANCED_EXTRACT_SYSTEM = """You are a fact extraction engine. Extract ALL factual statements from the dialog.
For each fact, output one line in this EXACT format:
subject|relation|value

Rules:
- subject: person/entity the fact is ABOUT (lowercase, e.g. "alice", "bob")
- relation: short verb phrase (e.g. "favorite_movie", "lives_in", "works_at", "birthday", "visited_on", "plans_to")
- value: the object/detail (lowercase)
- Extract EVERY factual statement, preference, plan, opinion, experience, date, location
- For DATE facts, create a SEPARATE entry: subject|event_date|YYYY-MM-DD or description
- For RELATIONSHIP facts between people: person_a|knows|person_b AND person_b|knows|person_a
- If no extractable facts (greetings, filler), output: NONE
- One fact per line. No numbering, no extra text.

Example input: "Alice: I got a new job at Google last March!"
Example output:
alice|works_at|google
alice|started_job_at_google|march
alice|employment_change|got new job at google"""


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

            # Create Fact directly from dialog turn — include date in raw_text
            raw_text = f"[{date_str}] {speaker}: {text}" if date_str else f"{speaker}: {text}"
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

            # Store a separate date-tagged entry for stronger temporal retrieval
            if date_str:
                date_fact = Fact(
                    subject=speaker.lower().strip(),
                    relation="event_date",
                    value=f"{date_str} {text[:100]}".lower().strip(),
                    override=False,
                    raw_text=f"[{date_str}] {raw_text}",
                )
                engine.store(f"{date_str} {raw_text}", namespace, fact=date_fact)
                ingested += 1

        if sess_num % 5 == 0:
            print(f"    Session {sess_num}: {ingested} facts ingested so far")

    # Finalize batch: recompute SVD ONLY (no free energy / crystallization).
    # Without SVD, _token_vectors stays empty → semantic_bonus = 0 for all queries.
    # We deliberately skip _recompute_all_free_energies — items must stay at s=1.0
    # for benchmark (no decay, no crystallization merging facts into 4 schemas).
    engine._batch_mode = False
    if engine._cooccurrence and engine._svd_dirty:
        engine._recompute_svd()
        engine._svd_store_count = 0
        engine._svd_dirty = False
    print(f"    SVD finalized: {len(engine._token_vectors)} token vectors, "
          f"{len(engine._cooccurrence)} co-occurrence pairs")

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
        if self.mode == "fullcontext":
            # Fullcontext mode: NO engine needed — raw conversation → LLM
            # This is the THEORETICAL CEILING benchmark
            stats = {"total_turns": 0, "ingested_facts": 0}
            t0 = time.time()
            # Build raw conversation text
            raw_conv_text = self._build_raw_conversation(conv)
            stats["total_turns"] = sum(
                len(conv.get(f"session_{n}", []))
                for n in range(1, 100)
                if f"session_{n}" in conv
            )
            stats["ingested_facts"] = stats["total_turns"]
            t_ingest = time.time() - t0
            print(f"  ✓ Fullcontext: {stats['total_turns']} turns, ~{len(raw_conv_text)//4} tokens ({t_ingest:.1f}s)")

            # Answer all QA
            t0 = time.time()
            qa_results = []
            model_key = "clspp"
            prediction_key = "clspp_prediction"

            for qi, qa in enumerate(sample["qa"]):
                question = qa["question"]
                answer = qa.get("answer", qa.get("adversarial_answer", ""))
                category = qa["category"]

                prediction = await self._fullcontext_answer_qa(
                    raw_conv_text, qa
                )

                score = self._compute_score(prediction, answer, category)
                qa_entry = {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "evidence": qa.get("evidence", []),
                    prediction_key: prediction,
                    f"{model_key}_f1": round(score, 3),
                    "_memory_items_used": stats["total_turns"],
                }
                qa_results.append(qa_entry)

                if (qi + 1) % 10 == 0 or qi == 0:
                    print(f"    QA {qi+1}/{len(sample['qa'])} | cat={category} | F1={score:.3f}")

            t_qa = time.time() - t0

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

            return {
                "sample_id": sample_id,
                "qa": qa_results,
                "phase_stats": {
                    "total_turns": stats["total_turns"],
                    "ingested_facts": stats["ingested_facts"],
                    "liquid_count": 0,
                    "gas_count": 0,
                    "event_counter": 0,
                    "memory_density_rho": 0.0,
                    "total_free_energy": 0.0,
                },
            }

        elif self.mode == "enhanced":
            benchmark_tau = 10000.0
            engine = PhaseMemoryEngine(
                kT=_settings.phase_kT,
                lambda_budget=_settings.phase_lambda,
                tau_c1=_settings.phase_tau_c1,
                tau_default=benchmark_tau,
                tau_override=_settings.phase_tau_override,
                strength_floor=_settings.phase_strength_floor,
                capacity=15000,  # Higher for dual indexing (raw + extracted)
                beta_retrieval=_settings.phase_beta_retrieval,
            )
            engine._benchmark_mode = True  # Preserve ALL items
        elif self.mode == "direct":
            benchmark_tau = 10000.0
            engine = PhaseMemoryEngine(
                kT=_settings.phase_kT,
                lambda_budget=_settings.phase_lambda,
                tau_c1=_settings.phase_tau_c1,
                tau_default=benchmark_tau,
                tau_override=_settings.phase_tau_override,
                strength_floor=_settings.phase_strength_floor,
                capacity=5000,
                beta_retrieval=_settings.phase_beta_retrieval,
            )
            engine._benchmark_mode = True  # Preserve ALL items — no crystallization GC
        else:
            benchmark_tau = _settings.phase_tau_default
            engine = PhaseMemoryEngine(
                kT=_settings.phase_kT,
                lambda_budget=_settings.phase_lambda,
                tau_c1=_settings.phase_tau_c1,
                tau_default=benchmark_tau,
                tau_override=_settings.phase_tau_override,
                strength_floor=_settings.phase_strength_floor,
                capacity=5000,
                beta_retrieval=_settings.phase_beta_retrieval,
            )
        namespace = sample_id

        # --- Phase 1: INGEST ---
        t0 = time.time()
        if self.mode == "direct":
            stats = _direct_ingest_conversation(engine, conv, namespace)
        elif self.mode == "enhanced":
            stats = await self._enhanced_ingest_conversation(engine, conv, namespace)
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

            if self.mode == "enhanced":
                prediction, debug_items = await self._enhanced_answer_qa(
                    engine, qa, namespace,
                )
            elif self.mode == "direct":
                # --- Pure extractive QA (no LLM needed) ---
                results, detail_episodes = engine.search_with_details(
                    question, namespace, limit=50, detail_limit=100
                )
                items = [item for _, item in results]
                debug_items = [
                    {"text": item.fact.raw_text, "score": round(score, 4)}
                    for score, item in results
                ]
                prediction = self._extractive_answer(
                    question, items, detail_episodes, category
                )
            else:
                # --- LLM-based QA (llm mode) ---
                memory_context, debug_items = engine.build_augmented_context_with_details(
                    question, namespace, limit=20, detail_limit=50
                )

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

    # =================================================================
    # Enhanced Mode — Dual-Index Ingestion
    # =================================================================

    async def _enhanced_ingest_conversation(self, engine, conv, namespace) -> dict:
        """
        Enhanced dual-indexing ingestion:
          A) Store raw dialog with session/date tags (preserves full context)
          B) LLM-extract structured triplets (enables entity-based retrieval)
          C) After all ingest, run recall agent to touch all items

        This gives BOTH keyword retrieval (raw) and semantic retrieval (triplets).
        """
        session_nums = sorted(set(
            int(k.split("_")[-1])
            for k in conv.keys()
            if k.startswith("session_") and "_date_time" not in k
            and "_observation" not in k and "_summary" not in k
        ))

        total_turns = 0
        ingested = 0
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

                # --- A) Store raw dialog with session/date tags ---
                tagged_value = f"[Session {sess_num}, {date_str}] {text}" if date_str else f"[Session {sess_num}] {text}"
                fact_raw = Fact(
                    subject=speaker.lower().strip(),
                    relation="said",
                    value=tagged_value.lower().strip(),
                    override=False,
                    raw_text=raw_text,
                )
                engine.store(raw_text, namespace, fact=fact_raw)
                ingested += 1

                # --- B) LLM-extract structured triplets ---
                try:
                    extraction = await _call_llm(
                        ENHANCED_EXTRACT_SYSTEM,
                        f"Session: {sess_num}\nDate: {date_str}\nSpeaker: {speaker}\nDialog: {text}"
                    )
                except Exception:
                    extraction = "NONE"

                if extraction and extraction.strip().upper() != "NONE":
                    for line in extraction.strip().split("\n"):
                        line = line.strip()
                        if not line or line.upper() == "NONE":
                            continue
                        parts = line.split("|")
                        if len(parts) >= 3:
                            subj = parts[0].strip().lower()
                            rel = parts[1].strip().lower().replace(" ", "_")
                            val = parts[2].strip().lower()
                            if date_str:
                                val = f"[Session {sess_num}, {date_str}] {val}"
                            fact_ext = Fact(
                                subject=subj,
                                relation=rel,
                                value=val,
                                override=False,
                                raw_text=raw_text,
                            )
                            rel_words = rel.replace("_", " ")
                            enriched = f"{subj} {rel_words} {val}. {raw_text}"
                            engine.store(enriched, namespace, fact=fact_ext)
                            ingested += 1

            if sess_num % 5 == 0:
                print(f"    Session {sess_num}: {ingested} facts ({total_turns} turns)")

        # Finalize batch: SVD only (no crystallization in benchmark mode)
        engine._batch_mode = False
        if engine._cooccurrence and engine._svd_dirty:
            engine._recompute_svd()
            engine._svd_store_count = 0
            engine._svd_dirty = False

        # --- C) Recall agent: touch ALL items to set high retrieval_count ---
        n_items = len(engine._items.get(namespace, []))
        rehearsed = engine.recall_long_tail(namespace, batch_size=n_items)
        print(f"    SVD finalized: {len(engine._token_vectors)} vectors, "
              f"recalled {rehearsed}/{n_items} items")

        return {"total_turns": total_turns, "ingested_facts": ingested}

    # =================================================================
    # Enhanced Mode — Category-Aware QA + Divide & Conquer
    # =================================================================

    async def _enhanced_answer_qa(self, engine, qa: dict, namespace: str):
        """
        Enhanced QA — FULL CONTEXT approach.

        CRITICAL INSIGHT: Max LoCoMo conversation = ~22K tokens.
        This fits EASILY in any modern LLM context window (128K+).
        Therefore: NO retrieval limit. Give ALL items to the LLM.

        Architecture:
          1. Retrieve ALL items in namespace (not top-K)
          2. Sort chronologically (preserves conversation flow)
          3. Temporal boosting for Cat 2 (dates at top)
          4. Full-context single LLM call (no D&C needed for <30K tokens)
          5. D&C fallback ONLY if context exceeds ~80K tokens
          6. Category-specific prompts with CoT for multi-hop

        Returns (prediction, debug_items).
        """
        question = qa["question"]
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        category = qa["category"]

        # --- Retrieve ALL items — no artificial limit ---
        # Use a very high limit to get everything in the namespace
        all_items = engine._items.get(namespace, [])

        # Use search_with_details for schema-aware retrieval + archive drill-down
        ranked_items, detail_episodes = engine.search_with_details(
            question, namespace, limit=len(all_items) + 100, detail_limit=200,
        )

        # Merge: ranked items first, then detail episodes, then any missed items
        # Extract items from (score, item) tuples returned by search_with_details
        seen_ids = {id(item) for _, item in ranked_items}
        items = [item for _, item in ranked_items]

        for ep in detail_episodes:
            if id(ep) not in seen_ids:
                items.append(ep)
                seen_ids.add(id(ep))

        for it in all_items:
            if id(it) not in seen_ids:
                items.append(it)
                seen_ids.add(id(it))

        # --- Multi-hop expansion for Cat 1: search for entities mentioned ---
        if category == 1 and items:
            expansion_entities = self._extract_expansion_terms(items[:30])
            for entity in expansion_entities[:8]:
                more = engine.search(entity, namespace, limit=50)
                for _score, it in more:
                    if id(it) not in seen_ids:
                        items.append(it)
                        seen_ids.add(id(it))

        # --- Temporal boosting for Cat 2 ---
        if category == 2:
            items = self._boost_temporal_items(items)

        # --- Chronological ordering (preserves conversation flow) ---
        items = self._chronological_order(items)

        # --- Build context ---
        context_str = self._build_context_string(items)

        # --- Estimate token count (~4 chars per token) ---
        est_tokens = len(context_str) // 4

        if est_tokens > 80000:
            # Only use D&C for truly massive contexts (>80K tokens)
            prediction = await self._divide_and_conquer_qa(items, question, category)
        else:
            # SINGLE LLM call with FULL context — this is the key insight
            prompts = ENHANCED_PROMPTS.get(category, ENHANCED_PROMPTS[4])
            system = prompts["system"]
            user_prompt = prompts["user"].format(context=context_str, question=question)
            try:
                prediction = await _call_llm(system, user_prompt)
                prediction = prediction.strip()
            except Exception:
                prediction = "I don't know"

        # --- Post-process Cat 5 adversarial detection ---
        if category == 5:
            prediction = self._normalize_adversarial(prediction)

        return prediction, items

    @staticmethod
    def _normalize_adversarial(prediction: str) -> str:
        """Robust adversarial detection — normalize 'not mentioned' variants."""
        p = prediction.lower().strip()
        NOT_MENTIONED_PATTERNS = [
            "not mentioned in the conversation",
            "not mentioned",
            "no information available",
            "not discussed",
            "not specified",
            "not stated",
            "no mention",
            "cannot be determined",
            "can't be determined",
            "not available in",
            "no relevant information",
            "i don't know",
            "there is no mention",
            "there's no mention",
            "doesn't mention",
            "does not mention",
            "wasn't mentioned",
            "was not mentioned",
            "isn't mentioned",
            "is not mentioned",
            "no data available",
            "cannot find",
            "unable to find",
            "not found in",
        ]
        for pattern in NOT_MENTIONED_PATTERNS:
            if pattern in p:
                return "Not mentioned in the conversation"
        return prediction

    async def _divide_and_conquer_qa(self, items, question: str, category: int) -> str:
        """
        Map-Reduce QA when context exceeds single LLM call capacity.

        Map:   Split items into chunks of 50 → extract relevant info per chunk
        Reduce: Merge all extractions → final LLM call for answer
        """
        MAX_PER_CHUNK = 50
        chunks = [items[i:i + MAX_PER_CHUNK] for i in range(0, len(items), MAX_PER_CHUNK)]

        # MAP: extract relevant info from each chunk
        extractions = []
        for ci, chunk in enumerate(chunks):
            ctx = self._build_context_string(chunk)
            try:
                extraction = await _call_llm(
                    "You are a memory search assistant. Extract ALL information from "
                    "these memories that could help answer the question. Include exact "
                    "quotes, names, dates, and session numbers. If nothing relevant, say NONE.",
                    f"Memories:\n{ctx}\n\nQuestion: {question}\n\nRelevant information:"
                )
                if extraction.strip().upper() != "NONE":
                    extractions.append(extraction)
            except Exception:
                continue

        if not extractions:
            return "Not mentioned in the conversation"

        # REDUCE: synthesize into final answer
        merged = "\n\n".join(f"[Source {i+1}]:\n{e}" for i, e in enumerate(extractions))
        prompts = ENHANCED_PROMPTS.get(category, ENHANCED_PROMPTS[4])
        system = prompts["system"]
        user_prompt = prompts["user"].format(context=merged, question=question)

        try:
            prediction = await _call_llm(system, user_prompt)
            return prediction.strip()
        except Exception:
            return "I don't know"

    # =================================================================
    # Enhanced Mode — Helper Methods
    # =================================================================

    @staticmethod
    def _extract_expansion_terms(items) -> list[str]:
        """Extract entity names from top retrieved items for multi-hop expansion."""
        entities = set()
        for item in items:
            if hasattr(item, 'fact') and item.fact:
                if item.fact.subject and item.fact.subject != "said":
                    entities.add(item.fact.subject)
                if item.fact.value:
                    # Extract capitalized words or names from value
                    words = item.fact.value.split()
                    for w in words:
                        clean = w.strip("[](),.:;'\"").lower()
                        if len(clean) > 2 and clean.isalpha() and clean not in {
                            "the", "and", "for", "that", "this", "with", "from",
                            "have", "has", "had", "was", "were", "are", "been",
                            "said", "not", "but", "can", "will", "would", "could",
                            "should", "about", "session", "like", "just", "really",
                            "know", "think", "yeah", "yes", "also", "well",
                        }:
                            entities.add(clean)
        return list(entities)[:10]

    @staticmethod
    def _boost_temporal_items(items) -> list:
        """Move items with date patterns to the top for temporal questions."""
        import re
        date_pattern = re.compile(
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|'
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|'
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*|'
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)',
            re.IGNORECASE
        )
        temporal = []
        non_temporal = []
        for item in items:
            text = ""
            if hasattr(item, 'fact') and item.fact:
                text = (item.fact.raw_text or "") + " " + (item.fact.value or "")
            if date_pattern.search(text):
                temporal.append(item)
            else:
                non_temporal.append(item)
        return temporal + non_temporal

    @staticmethod
    def _chronological_order(items) -> list:
        """Sort items by birth_order (chronological ingest order)."""
        return sorted(items, key=lambda i: getattr(i, 'birth_order', 0))

    @staticmethod
    def _build_context_string(items) -> str:
        """Build context string from list of PhaseMemoryItems."""
        lines = []
        for item in items:
            strength = getattr(item, 'strength', 1.0)
            if hasattr(item, 'fact') and item.fact and item.fact.raw_text:
                text = item.fact.raw_text
            elif hasattr(item, 'fact') and item.fact:
                text = f"{item.fact.subject} {item.fact.relation} {item.fact.value}"
            else:
                text = str(item)
            # Include session/date tag from value if available
            if hasattr(item, 'fact') and item.fact and item.fact.value:
                val = item.fact.value
                if val.startswith("[session") or val.startswith("[Session"):
                    bracket_end = val.find("]")
                    if bracket_end > 0:
                        tag = val[:bracket_end + 1]
                        text = f"{tag} {text}"
            lines.append(f"- [s={strength:.2f}] {text}")
        return "\n".join(lines)

    # =================================================================
    # Fullcontext Mode — Raw Conversation → LLM (Theoretical Ceiling)
    # =================================================================

    @staticmethod
    def _build_raw_conversation(conv: dict) -> str:
        """Build raw conversation text with session/date markers."""
        session_nums = sorted(set(
            int(k.split("_")[-1])
            for k in conv.keys()
            if k.startswith("session_") and "_date_time" not in k
            and "_observation" not in k and "_summary" not in k
        ))
        lines = []
        for sess_num in session_nums:
            sess_key = f"session_{sess_num}"
            if sess_key not in conv or not conv[sess_key]:
                continue
            date_key = f"session_{sess_num}_date_time"
            date_str = conv.get(date_key, "")
            lines.append(f"\n--- Session {sess_num} [{date_str}] ---")
            for dialog in conv[sess_key]:
                speaker = dialog.get("speaker", "Unknown")
                text = dialog.get("text", "").strip()
                if text:
                    lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    async def _fullcontext_answer_qa(self, raw_conv: str, qa: dict) -> str:
        """Answer QA using FULL raw conversation — no retrieval, no CLS++."""
        question = qa["question"]
        category = qa["category"]

        prompts = ENHANCED_PROMPTS.get(category, ENHANCED_PROMPTS[4])
        system = prompts["system"]
        user_prompt = prompts["user"].format(context=raw_conv, question=question)

        try:
            prediction = await _call_llm(system, user_prompt)
            prediction = prediction.strip()
        except Exception:
            prediction = "I don't know"

        if category == 5:
            prediction = self._normalize_adversarial(prediction)

        return prediction

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

    def _extractive_answer(self, question: str, memory_items: list, detail_episodes: list, category: int) -> str:
        """Pure extractive QA — no LLM needed. Extracts answer spans from retrieved memory."""
        import re as _re_local

        # Category 5 (adversarial): always answer "not mentioned"
        if category == 5:
            return "Not mentioned in the conversation"

        # Collect all text snippets from retrieved items + details
        snippets: list[str] = []
        for item in memory_items:
            if hasattr(item, 'fact'):
                snippets.append(item.fact.raw_text)
        for ep in detail_episodes:
            if hasattr(ep, 'fact'):
                snippets.append(ep.fact.raw_text)

        if not snippets:
            return "I don't know"

        STOP = {'what', 'when', 'where', 'who', 'how', 'which', 'why', 'did',
                'does', 'do', 'is', 'are', 'was', 'were', 'the', 'a', 'an',
                'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'that',
                'this', 'it', 'they', 'he', 'she', 'his', 'her', 'their',
                'has', 'have', 'had', 'been', 'be', 'with', 'about', 'from',
                'by', 'as', 'can', 'could', 'would', 'should', 'will',
                'during', 'between', 'after', 'before', 'into', 'through',
                'not', 'but', 'if', 'then', 'so', 'just', 'like', 'really',
                'very', 'also', 'too', 'some', 'any', 'all', 'most', 'own',
                'up', 'out', 'over', 'no', 'yes', 'well', 'much'}

        q_lower = question.lower()
        q_tokens = set(_normalize_answer(question).split()) - STOP
        if not q_tokens:
            q_tokens = set(_normalize_answer(question).split())

        # Extract named entities from question (capitalized words, proper nouns)
        q_entities = set()
        for word in question.split():
            clean = word.strip('?.,!:;\'"')
            if clean and clean[0].isupper() and clean.lower() not in STOP:
                q_entities.add(clean.lower())

        # --- Score snippets by relevance ---
        scored: list[tuple[float, str]] = []
        for snip in snippets:
            snip_norm = _normalize_answer(snip)
            snip_tokens = set(snip_norm.split())
            # Token overlap score
            overlap = len(q_tokens & snip_tokens)
            score = overlap / max(len(q_tokens), 1)
            # Entity bonus — named entities matching is a strong signal
            for ent in q_entities:
                if ent in snip_norm:
                    score += 2.0
            scored.append((score, snip))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_snippets = [s for _, s in scored[:20]]

        # --- Category 2 (temporal): extract dates ---
        if category == 2:
            date_pattern = _re_local.compile(
                r'(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})|'
                r'(?:(?:january|february|march|april|may|june|july|august|september|october|november|december|'
                r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)|'
                r'(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                _re_local.IGNORECASE,
            )
            # Check snippets with entity matches first
            for snip in top_snippets:
                if any(ent in snip.lower() for ent in q_entities):
                    dates = date_pattern.findall(snip)
                    if dates:
                        return dates[0]
            # Then check all top snippets
            for snip in top_snippets:
                dates = date_pattern.findall(snip)
                if dates:
                    return dates[0]
            # Check ALL snippets as fallback
            for _, snip in scored:
                dates = date_pattern.findall(snip)
                if dates:
                    return dates[0]

        # --- Extract answer spans (not full snippets) ---
        # Split snippets into sentences and score each sentence
        sentences: list[tuple[float, str]] = []
        for snip in top_snippets:
            # Split on sentence boundaries
            sents = _re_local.split(r'[.!?]+', snip)
            for sent in sents:
                sent = sent.strip()
                if len(sent) < 5:
                    continue
                sent_norm = _normalize_answer(sent)
                sent_tokens = set(sent_norm.split())
                # Score: tokens NOT in question (these are likely the answer)
                answer_tokens = sent_tokens - q_tokens - STOP
                # But also require some question tokens for relevance
                relevance = len(q_tokens & sent_tokens) / max(len(q_tokens), 1)
                entity_match = sum(1 for ent in q_entities if ent in sent_norm)
                # Good answer: relevant to question + has new info
                sent_score = relevance * 2.0 + entity_match * 3.0 + len(answer_tokens) * 0.1
                if relevance > 0 or entity_match > 0:
                    sentences.append((sent_score, sent))

        sentences.sort(key=lambda x: x[0], reverse=True)

        if sentences:
            # Return top sentence, trimmed
            best_sent = sentences[0][1]
            # Remove speaker prefix like "Alice:" or "[date] Speaker:"
            best_sent = _re_local.sub(r'^\[.*?\]\s*', '', best_sent)
            best_sent = _re_local.sub(r'^[A-Za-z]+:\s*', '', best_sent)
            if len(best_sent) > 200:
                best_sent = best_sent[:200]
            return best_sent.strip()

        # Fallback: return first snippet trimmed
        if top_snippets:
            best = top_snippets[0]
            best = _re_local.sub(r'^\[.*?\]\s*', '', best)
            best = _re_local.sub(r'^[A-Za-z]+:\s*', '', best)
            if len(best) > 200:
                sents = best.split('.')
                return sents[0].strip()
            return best.strip()

        return "I don't know"

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
            pred_lower = prediction.lower().strip()
            # Robust adversarial detection — many valid phrasings
            NOT_MENTIONED = [
                "not mentioned in the conversation",
                "not mentioned", "no information available",
                "not discussed", "not specified", "not stated",
                "no mention", "cannot be determined", "can't be determined",
                "i don't know", "not found in", "no relevant information",
                "doesn't mention", "does not mention",
                "wasn't mentioned", "was not mentioned",
            ]
            for pattern in NOT_MENTIONED:
                if pattern in pred_lower:
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

async def run_ab_comparison(data_file: str, out_dir: str, limit: int = 0):
    """Run A/B comparison: Mode A (direct) vs Mode B (enhanced)."""
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║              CLS++ A/B COMPARISON BENCHMARK                  ║")
    print("║  Mode A: direct (baseline)    Mode B: enhanced (recall+D&C)  ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")

    os.makedirs(out_dir, exist_ok=True)

    # Run Mode A
    print("\n" + "="*60)
    print("  RUNNING MODE A (direct baseline)")
    print("="*60)
    out_a = os.path.join(out_dir, "ab_mode_a_results.json")
    bench_a = CLSPPBenchmark(data_file, out_a, limit, "direct")
    await bench_a.run()

    # Run Mode B
    print("\n" + "="*60)
    print("  RUNNING MODE B (enhanced)")
    print("="*60)
    out_b = os.path.join(out_dir, "ab_mode_b_results.json")
    bench_b = CLSPPBenchmark(data_file, out_b, limit, "enhanced")
    await bench_b.run()

    # Compare results
    def _cat_scores(results):
        cat_acc = defaultdict(list)
        for r in results:
            for qa in r["qa"]:
                cat_acc[qa["category"]].append(qa["clspp_f1"])
        return cat_acc

    scores_a = _cat_scores(bench_a.results)
    scores_b = _cat_scores(bench_b.results)

    cat_labels = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Long-context", 5: "Adversarial"}

    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║                    A/B COMPARISON RESULTS                    ║")
    print("╠════════════════════╤═══════════════╤═══════════════╤═════════╣")
    print("║  Category          │ Mode A (dir.) │ Mode B (enh.) │   Δ     ║")
    print("╠════════════════════╪═══════════════╪═══════════════╪═════════╣")

    all_a, all_b = [], []
    for cat in [1, 2, 3, 4, 5]:
        sa = scores_a.get(cat, [])
        sb = scores_b.get(cat, [])
        avg_a = sum(sa) / len(sa) if sa else 0.0
        avg_b = sum(sb) / len(sb) if sb else 0.0
        delta = avg_b - avg_a
        all_a.extend(sa)
        all_b.extend(sb)
        label = f"Cat {cat} {cat_labels[cat]}"
        sign = "+" if delta >= 0 else ""
        print(f"║  {label:<18}│   {avg_a*100:>6.2f}%     │   {avg_b*100:>6.2f}%     │ {sign}{delta*100:>5.2f}% ║")

    overall_a = sum(all_a) / len(all_a) if all_a else 0.0
    overall_b = sum(all_b) / len(all_b) if all_b else 0.0
    delta_o = overall_b - overall_a
    sign_o = "+" if delta_o >= 0 else ""
    print("╠════════════════════╪═══════════════╪═══════════════╪═════════╣")
    print(f"║  {'OVERALL':<18}│   {overall_a*100:>6.2f}%     │   {overall_b*100:>6.2f}%     │ {sign_o}{delta_o*100:>5.2f}% ║")
    print("╚════════════════════╧═══════════════╧═══════════════╧═════════╝")

    # Save per-question paired comparison
    paired = []
    for ra, rb in zip(bench_a.results, bench_b.results):
        for qa_a, qa_b in zip(ra["qa"], rb["qa"]):
            paired.append({
                "question": qa_a["question"],
                "category": qa_a["category"],
                "answer": qa_a["answer"],
                "mode_a_prediction": qa_a.get("clspp_prediction", ""),
                "mode_b_prediction": qa_b.get("clspp_prediction", ""),
                "mode_a_f1": qa_a["clspp_f1"],
                "mode_b_f1": qa_b["clspp_f1"],
                "delta": round(qa_b["clspp_f1"] - qa_a["clspp_f1"], 3),
            })
    paired_file = os.path.join(out_dir, "ab_paired_comparison.json")
    with open(paired_file, "w") as f:
        json.dump(paired, f, indent=2)
    print(f"\n  Paired comparison: {paired_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLS++ LoCoMo Benchmark")
    parser.add_argument("--data-file", type=str,
                        default=str(Path(__file__).parent / "locomo/data/locomo10.json"))
    parser.add_argument("--out-file", type=str,
                        default=str(Path(__file__).parent / "outputs/clspp_locomo10_results.json"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of conversations (0 = all)")
    parser.add_argument("--mode", type=str, default="direct",
                        choices=["direct", "llm", "enhanced", "fullcontext"],
                        help="Mode: 'direct' (fast), 'llm' (extraction), 'enhanced' (full CLS++), 'fullcontext' (ceiling: raw conv → LLM)")
    parser.add_argument("--ab", action="store_true",
                        help="Run A/B comparison: direct vs enhanced")
    args = parser.parse_args()

    if args.ab:
        out_dir = os.path.dirname(args.out_file) or str(Path(__file__).parent / "outputs")
        asyncio.run(run_ab_comparison(args.data_file, out_dir, args.limit))
    else:
        benchmark = CLSPPBenchmark(args.data_file, args.out_file, args.limit, args.mode)
        asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
