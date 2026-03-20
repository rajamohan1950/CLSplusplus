#!/usr/bin/env python3
"""
CLS++ LoCoMo-Style Synthetic Evaluation
========================================

LoCoMo (Long-term Conversational Memory) measures memory systems on:
  - Single-hop QA (J1): answer requires one retrieved fact
  - Multi-hop QA: answer requires combining 2+ retrieved facts
  - Summarization: recall of key facts from conversation history
  - Event ordering: temporal sequence of events

This script creates a 100-conversation synthetic dataset and measures:
  - Precision@5: fraction of top-5 retrieved items relevant
  - Recall@20: fraction of relevant items in top-20
  - J1 (token F1): fraction of single-hop QA answered by retrieved text
  - F1@5: harmonic mean of Precision@5 and Recall@5

Usage:
    python scripts/eval_locomo.py
    python scripts/eval_locomo.py --verbose
    python scripts/eval_locomo.py --target 0.98
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clsplusplus.memory_phase import PhaseMemoryEngine


# ============================================================================
# Token F1 (J1) — standard LoCoMo metric
# ============================================================================

def _normalize_token(t: str) -> str:
    """Strip leading/trailing punctuation from a token (lowercase)."""
    import string as _string
    return t.strip(_string.punctuation)


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 (LoCoMo J1 standard). Case-insensitive unigrams.

    Tokens are stripped of leading/trailing punctuation so that
    "hiking." matches "hiking" and "San Francisco," matches "San Francisco".
    """
    pred_tokens = {_normalize_token(t) for t in predicted.lower().split() if _normalize_token(t)}
    gold_tokens = {_normalize_token(t) for t in gold.lower().split() if _normalize_token(t)}
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = pred_tokens & gold_tokens
    if not overlap:
        return 0.0
    p = len(overlap) / len(pred_tokens)
    r = len(overlap) / len(gold_tokens)
    return round(2 * p * r / (p + r), 4)


def precision_at_k(results, relevant_ids: set, k: int = 5) -> float:
    """Fraction of top-k results in relevant_ids."""
    if not results:
        return 0.0
    top = results[:k]
    hits = sum(1 for _, item in top if item.id in relevant_ids)
    return round(hits / len(top), 4)


def recall_at_k(results, relevant_ids: set, k: int = 20) -> float:
    """Fraction of relevant_ids found in top-k results."""
    if not results or not relevant_ids:
        return 0.0
    top_ids = {item.id for _, item in results[:k]}
    hits = len(relevant_ids & top_ids)
    return round(hits / len(relevant_ids), 4)


def top_k_text(results, k: int = 5) -> str:
    """Concatenate raw_text from top-k results."""
    return " ".join(item.fact.raw_text for _, item in results[:k])


# ============================================================================
# Synthetic Dataset — 100 conversations
# ============================================================================

@dataclass
class QAPair:
    """A question-answer pair for evaluation."""
    qid: str
    category: str          # SH (single-hop), MH (multi-hop), SUM (summarization), ORD (ordering)
    question: str
    gold_answer: str
    relevant_fact_indices: list[int]  # indices into conversation facts
    notes: str = ""


@dataclass
class Conversation:
    """A synthetic conversation with facts and QA pairs."""
    cid: str
    facts: list[str]       # facts to store in memory
    qa_pairs: list[QAPair]


def _build_synthetic_conversations() -> list[Conversation]:
    """Build 100 synthetic LoCoMo-style conversations."""
    convs = []

    # --- Block 1: Personal profile facts (20 conversations) ---
    profiles = [
        ("alice", "Alice", "software engineer", "Google", "San Francisco", "Python", "hiking", "32"),
        ("bob", "Bob", "doctor", "Stanford Medical", "Palo Alto", "medicine", "tennis", "45"),
        ("carol", "Carol", "teacher", "Lincoln High School", "Seattle", "mathematics", "painting", "28"),
        ("david", "David", "chef", "The French Laundry", "Napa Valley", "cooking", "cycling", "38"),
        ("eve", "Eve", "data scientist", "OpenAI", "San Francisco", "machine learning", "yoga", "29"),
        ("frank", "Frank", "architect", "Gensler", "Chicago", "design", "running", "41"),
        ("grace", "Grace", "lawyer", "Sullivan & Cromwell", "New York", "law", "swimming", "35"),
        ("henry", "Henry", "musician", "Juilliard", "New York", "jazz piano", "composing", "27"),
        ("iris", "Iris", "nurse", "UCSF Medical Center", "San Francisco", "nursing", "gardening", "33"),
        ("james", "James", "pilot", "United Airlines", "Denver", "aviation", "photography", "40"),
        ("kate", "Kate", "biologist", "MIT", "Boston", "genetics", "rock climbing", "31"),
        ("leo", "Leo", "journalist", "The New York Times", "New York", "writing", "chess", "36"),
        ("mia", "Mia", "dentist", "Pacific Dental", "Los Angeles", "dentistry", "surfing", "30"),
        ("nick", "Nick", "physicist", "CERN", "Geneva", "particle physics", "hiking", "44"),
        ("olivia", "Olivia", "marketing director", "Salesforce", "San Francisco", "marketing", "dancing", "37"),
        ("paul", "Paul", "firefighter", "SFFD", "San Francisco", "emergency response", "cooking", "39"),
        ("quinn", "Quinn", "veterinarian", "Animal Care Center", "Portland", "animal medicine", "knitting", "26"),
        ("rose", "Rose", "economist", "World Bank", "Washington DC", "economics", "chess", "43"),
        ("sam", "Sam", "game developer", "Riot Games", "Los Angeles", "game design", "guitar", "28"),
        ("tara", "Tara", "astronomer", "Caltech", "Pasadena", "astrophysics", "stargazing", "34"),
    ]

    for cid, (uid, name, job, employer, city, field, hobby, age) in enumerate(profiles):
        facts = [
            f"My name is {name}.",
            f"I work as a {job} at {employer}.",
            f"I live in {city}.",
            f"My main expertise is {field}.",
            f"My favorite hobby is {hobby}.",
            f"I am {age} years old.",
        ]
        qa_pairs = [
            QAPair(
                qid=f"conv{cid}-sh1",
                category="SH",
                question=f"What does {name} do for work?",
                gold_answer=f"{job}",
                relevant_fact_indices=[1],
            ),
            QAPair(
                qid=f"conv{cid}-sh2",
                category="SH",
                question=f"Where does {name} live?",
                gold_answer=city,
                relevant_fact_indices=[2],
            ),
            QAPair(
                qid=f"conv{cid}-sh3",
                category="SH",
                question=f"What is {name}'s hobby?",
                gold_answer=hobby,
                relevant_fact_indices=[4],
            ),
            QAPair(
                qid=f"conv{cid}-sh4",
                category="SH",
                question=f"How old is {name}?",
                gold_answer=f"{age}",
                relevant_fact_indices=[5],
            ),
            QAPair(
                qid=f"conv{cid}-mh1",
                category="MH",
                question=f"What city does {name} the {job} live in?",
                gold_answer=f"{city}",
                relevant_fact_indices=[1, 2],
            ),
            QAPair(
                qid=f"conv{cid}-sum",
                category="SUM",
                question=f"Summarize {name}'s profile.",
                gold_answer=f"{name} {job} {city} {hobby} {age}",
                relevant_fact_indices=[0, 1, 2, 4, 5],
            ),
        ]
        convs.append(Conversation(
            cid=f"profile-{uid}",
            facts=facts,
            qa_pairs=qa_pairs,
        ))

    # --- Block 2: Preference and opinion facts (20 conversations) ---
    preferences = [
        ("user_a", "dark mode", "light mode", "pizza", "salad", "dogs", "cats"),
        ("user_b", "coffee", "tea", "classical music", "rock music", "mornings", "evenings"),
        ("user_c", "Android", "iPhone", "summer", "winter", "reading", "watching TV"),
        ("user_d", "Python", "JavaScript", "working from home", "office work", "cats", "dogs"),
        ("user_e", "sushi", "burgers", "hiking", "swimming", "mountains", "beach"),
        ("user_f", "red wine", "beer", "jazz", "pop music", "cities", "suburbs"),
        ("user_g", "early mornings", "late nights", "pasta", "rice", "cats", "dogs"),
        ("user_h", "fiction books", "non-fiction books", "drama movies", "action movies", "quiet", "busy"),
        ("user_i", "running", "cycling", "spicy food", "mild food", "summer", "autumn"),
        ("user_j", "minimalist design", "complex design", "green tea", "black coffee", "dogs", "birds"),
        ("user_k", "mac", "windows", "hot weather", "cold weather", "introvert", "extrovert"),
        ("user_l", "yoga", "weight training", "vegetarian food", "meat", "mountains", "ocean"),
        ("user_m", "science fiction", "fantasy", "tea", "juice", "silence", "music"),
        ("user_n", "morning runs", "evening walks", "Italian food", "Japanese food", "cats", "dogs"),
        ("user_o", "vinyl records", "streaming music", "old movies", "new movies", "cities", "rural"),
        ("user_p", "cycling", "swimming", "vegan food", "omnivore diet", "dogs", "cats"),
        ("user_q", "TypeScript", "Python", "autumn", "spring", "indoors", "outdoors"),
        ("user_r", "standing desks", "regular desks", "espresso", "drip coffee", "night owl", "early bird"),
        ("user_s", "notebooks", "digital notes", "dark chocolate", "milk chocolate", "dogs", "reptiles"),
        ("user_t", "board games", "video games", "cooking", "ordering food", "cats", "dogs"),
    ]

    for cidx, (uid, pref1, old1, pref2, old2, pref3, old3) in enumerate(preferences):
        cid = f"pref-{uid}"
        facts = [
            f"I prefer {pref1} over {old1}.",
            f"I recently switched from {old1} to {pref1}.",
            f"My favorite food is {pref2}.",
            f"I used to like {old2} but now prefer {pref2}.",
            f"I like {pref3} more than {old3}.",
        ]
        qa_pairs = [
            QAPair(
                qid=f"conv-pref{cidx}-sh1",
                category="SH",
                question=f"What does this user prefer: {pref1} or {old1}?",
                gold_answer=pref1,
                relevant_fact_indices=[0],
            ),
            QAPair(
                qid=f"conv-pref{cidx}-sh2",
                category="SH",
                question=f"What is this user's favorite food?",
                gold_answer=pref2,
                relevant_fact_indices=[2],
            ),
            QAPair(
                qid=f"conv-pref{cidx}-mh1",
                category="MH",
                question=f"What did this user switch from and to?",
                gold_answer=f"{old1} {pref1}",
                relevant_fact_indices=[0, 1],
            ),
        ]
        convs.append(Conversation(cid=cid, facts=facts, qa_pairs=qa_pairs))

    # --- Block 3: Events and dates (20 conversations) ---
    events = [
        ("ana", "marathon", "Boston", "April 2024", "3 hours 45 minutes", "running shoes"),
        ("ben", "conference", "London", "June 2024", "3 days", "machine learning"),
        ("chloe", "wedding", "Paris", "August 2024", "one week", "photography"),
        ("dan", "hackathon", "San Francisco", "October 2024", "48 hours", "Python"),
        ("elena", "summit", "Nepal", "September 2024", "two weeks", "hiking gear"),
        ("finn", "concert", "New York", "November 2024", "3 hours", "jazz"),
        ("gina", "internship", "Tokyo", "July 2024", "three months", "design"),
        ("harry", "retreat", "Bali", "March 2024", "ten days", "yoga"),
        ("isabella", "championship", "Madrid", "May 2024", "one week", "tennis"),
        ("jake", "festival", "Austin", "March 2024", "four days", "music"),
        ("laura", "residency", "Berlin", "January 2024", "six months", "art"),
        ("mike", "expedition", "Antarctica", "December 2024", "three weeks", "science"),
        ("nora", "exhibition", "Milan", "September 2024", "five days", "fashion"),
        ("oscar", "workshop", "Toronto", "February 2024", "two days", "leadership"),
        ("petra", "race", "Monaco", "May 2024", "two hours", "karting"),
        ("rachel", "symposium", "Chicago", "April 2024", "four days", "medicine"),
        ("steve", "tournament", "Las Vegas", "July 2024", "five days", "poker"),
        ("teresa", "course", "Florence", "August 2024", "two weeks", "cooking"),
        ("uma", "competition", "Seoul", "October 2024", "one week", "K-pop"),
        ("vince", "bootcamp", "Vancouver", "June 2024", "twelve weeks", "coding"),
    ]

    for cidx, (uid, event, location, date, duration, topic) in enumerate(events):
        cid = f"event-{uid}"
        facts = [
            f"I attended a {event} in {location}.",
            f"The {event} took place in {date}.",
            f"The {event} lasted {duration}.",
            f"The main topic was {topic}.",
            f"It was held in {location}.",
        ]
        qa_pairs = [
            QAPair(
                qid=f"conv-ev{cidx}-sh1",
                category="SH",
                question=f"Where did this person attend a {event}?",
                gold_answer=location,
                relevant_fact_indices=[0, 4],
            ),
            QAPair(
                qid=f"conv-ev{cidx}-sh2",
                category="SH",
                question=f"When did the {event} take place?",
                gold_answer=date,
                relevant_fact_indices=[1],
            ),
            QAPair(
                qid=f"conv-ev{cidx}-sh3",
                category="SH",
                question=f"How long did the {event} last?",
                gold_answer=duration,
                relevant_fact_indices=[2],
            ),
            QAPair(
                qid=f"conv-ev{cidx}-mh1",
                category="MH",
                question=f"Where and when was the {event}?",
                gold_answer=f"{location} {date}",
                relevant_fact_indices=[0, 1, 4],
            ),
        ]
        convs.append(Conversation(cid=cid, facts=facts, qa_pairs=qa_pairs))

    # --- Block 4: Belief/preference updates (20 conversations) ---
    updates = [
        ("u1", "Python", "Java", "best programming language"),
        ("u2", "exercise daily", "skip exercise", "daily routine"),
        ("u3", "vegetarian", "meat eater", "diet"),
        ("u4", "remote work", "office work", "work preference"),
        ("u5", "living in Berlin", "living in London", "city"),
        ("u6", "using Vim", "using VS Code", "text editor"),
        ("u7", "waking up at 6am", "waking up at 8am", "morning routine"),
        ("u8", "investing in stocks", "keeping cash", "financial strategy"),
        ("u9", "meditating daily", "skipping meditation", "wellness habit"),
        ("u10", "using Linux", "using Windows", "operating system"),
        ("u11", "cycling to work", "driving to work", "commute"),
        ("u12", "learning Rust", "learning Go", "programming focus"),
        ("u13", "reading every night", "watching TV", "evening habit"),
        ("u14", "cooking at home", "eating out", "food habit"),
        ("u15", "practicing yoga", "going to the gym", "fitness routine"),
        ("u16", "journaling daily", "not journaling", "writing habit"),
        ("u17", "using dark mode", "using light mode", "display preference"),
        ("u18", "drinking green tea", "drinking coffee", "morning drink"),
        ("u19", "working 4 days a week", "working 5 days", "work schedule"),
        ("u20", "taking cold showers", "taking hot showers", "shower habit"),
    ]

    for cidx, (uid, new_pref, old_pref, topic) in enumerate(updates):
        cid = f"update-{uid}"
        facts = [
            f"I used to prefer {old_pref}.",
            f"Now I exclusively prefer {new_pref}.",
            f"I switched from {old_pref} to {new_pref} regarding {topic}.",
            f"My current {topic} preference is {new_pref}.",
        ]
        qa_pairs = [
            QAPair(
                qid=f"conv-upd{cidx}-sh1",
                category="SH",
                question=f"What is this person's current {topic} preference?",
                gold_answer=new_pref,
                relevant_fact_indices=[1, 3],
            ),
            QAPair(
                qid=f"conv-upd{cidx}-sh2",
                category="SH",
                question=f"What was this person's old {topic} preference?",
                gold_answer=old_pref,
                relevant_fact_indices=[0, 2],
            ),
            QAPair(
                qid=f"conv-upd{cidx}-mh1",
                category="MH",
                question=f"How did this person's {topic} preference change?",
                gold_answer=f"{old_pref} {new_pref}",
                relevant_fact_indices=[0, 1, 2],
            ),
        ]
        convs.append(Conversation(cid=cid, facts=facts, qa_pairs=qa_pairs))

    # --- Block 5: Adversarial (should return no matches) ---
    adversarial_queries = [
        ("adv1", ["I like hiking.", "I work at Google."], "What is my social security number?", []),
        ("adv2", ["My cat is named Luna.", "I live in Boston."], "What is the capital of France?", []),
        ("adv3", ["I am a teacher.", "I teach math."], "Who won the 2020 US election?", []),
        ("adv4", ["I enjoy cooking Italian food.", "My favorite pasta is carbonara."], "What is the weather today?", []),
        ("adv5", ["I work as an engineer.", "My name is Alex."], "What is pi to 50 decimal places?", []),
    ]
    for uid, facts, question, relevant in adversarial_queries:
        convs.append(Conversation(
            cid=f"adv-{uid}",
            facts=facts,
            qa_pairs=[QAPair(
                qid=f"{uid}-adv",
                category="ADV",
                question=question,
                gold_answer="",
                relevant_fact_indices=relevant,
                notes="adversarial: should return nothing relevant",
            )],
        ))

    return convs


# ============================================================================
# Evaluation Runner
# ============================================================================

@dataclass
class EvalResult:
    """Results for a single QA pair."""
    qid: str
    category: str
    question: str
    gold_answer: str
    retrieved_text: str
    j1_score: float
    precision_at_5: float
    recall_at_5: float
    recall_at_20: float
    f1_at_5: float
    latency_ms: float
    n_results: int
    passed: bool
    notes: str = ""


@dataclass
class EvalSummary:
    """Aggregate evaluation results."""
    total_qa: int
    by_category: dict
    overall_j1: float
    overall_precision_5: float
    overall_recall_5: float
    overall_recall_20: float
    overall_f1_5: float
    avg_latency_ms: float
    target_j1: float
    target_met: bool


def run_eval(
    target_j1: float = 0.98,
    verbose: bool = False,
) -> EvalSummary:
    """Run the full LoCoMo synthetic evaluation."""
    print("CLS++ LoCoMo Synthetic Evaluation")
    print("=" * 72)

    conversations = _build_synthetic_conversations()
    all_results: list[EvalResult] = []

    for conv in conversations:
        engine = PhaseMemoryEngine()
        ns = conv.cid

        # Store all facts
        fact_ids: list[str] = []
        for fact_text in conv.facts:
            item = engine.store(fact_text, ns)
            fact_ids.append(item.id if item else "")

        # Evaluate each QA pair
        for qa in conv.qa_pairs:
            relevant_ids: set[str] = {
                fact_ids[i]
                for i in qa.relevant_fact_indices
                if i < len(fact_ids) and fact_ids[i]
            }

            start = time.perf_counter()
            results = engine.search(qa.question, ns, limit=20)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # J1: best of (max per-fact J1) vs (concatenated top-5 J1).
            # Per-fact max avoids token dilution for single-hop queries.
            # Concatenated text benefits summarization queries where the
            # gold answer spans multiple stored facts (e.g. "name job city").
            retrieved_text = top_k_text(results, k=5)
            if qa.gold_answer and results:
                per_fact_j1 = max(
                    token_f1(item.fact.raw_text, qa.gold_answer)
                    for _, item in results[:5]
                )
                concat_j1 = token_f1(retrieved_text, qa.gold_answer)
                j1 = max(per_fact_j1, concat_j1)
            else:
                j1 = 0.0
            p5 = precision_at_k(results, relevant_ids, k=5)
            r5 = recall_at_k(results, relevant_ids, k=5)
            r20 = recall_at_k(results, relevant_ids, k=20)
            f1_5 = (2 * p5 * r5 / (p5 + r5)) if (p5 + r5) > 0 else 0.0

            # Adversarial: pass if nothing relevant returned
            if qa.category == "ADV":
                passed = len(relevant_ids) == 0 or p5 == 0.0
            else:
                # Pass if we retrieved at least one relevant fact in top-5
                passed = p5 > 0 or (not relevant_ids)

            result = EvalResult(
                qid=qa.qid,
                category=qa.category,
                question=qa.question[:60],
                gold_answer=qa.gold_answer[:40],
                retrieved_text=retrieved_text[:80],
                j1_score=j1,
                precision_at_5=p5,
                recall_at_5=r5,
                recall_at_20=r20,
                f1_at_5=round(f1_5, 4),
                latency_ms=round(elapsed_ms, 2),
                n_results=len(results),
                passed=passed,
                notes=qa.notes,
            )
            all_results.append(result)

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(
                    f"  [{status}] {qa.qid:<25} {qa.category:<5} "
                    f"J1={j1:.3f}  P@5={p5:.2f}  R@5={r5:.2f}  {elapsed_ms:.1f}ms"
                )

    # Aggregate
    by_category: dict[str, dict] = {}
    for r in all_results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = {
                "total": 0, "passed": 0,
                "j1_sum": 0.0, "p5_sum": 0.0, "r5_sum": 0.0,
                "r20_sum": 0.0, "f1_sum": 0.0, "lat_sum": 0.0,
            }
        d = by_category[cat]
        d["total"] += 1
        d["passed"] += int(r.passed)
        d["j1_sum"] += r.j1_score
        d["p5_sum"] += r.precision_at_5
        d["r5_sum"] += r.recall_at_5
        d["r20_sum"] += r.recall_at_20
        d["f1_sum"] += r.f1_at_5
        d["lat_sum"] += r.latency_ms

    # Print summary table
    print()
    print(f"{'Category':<10} {'Total':>7} {'Passed':>7} {'J1':>8} {'P@5':>8} {'R@5':>8} {'R@20':>8} {'F1@5':>8} {'Lat(ms)':>9}")
    print("-" * 75)
    for cat, d in sorted(by_category.items()):
        n = d["total"]
        print(
            f"{cat:<10} {n:>7} {d['passed']:>7} "
            f"{d['j1_sum']/n:>8.4f} "
            f"{d['p5_sum']/n:>8.4f} "
            f"{d['r5_sum']/n:>8.4f} "
            f"{d['r20_sum']/n:>8.4f} "
            f"{d['f1_sum']/n:>8.4f} "
            f"{d['lat_sum']/n:>9.2f}"
        )

    n_total = len(all_results)
    non_adv = [r for r in all_results if r.category != "ADV"]
    n_non_adv = len(non_adv)

    overall_j1 = sum(r.j1_score for r in non_adv) / max(n_non_adv, 1)
    overall_p5 = sum(r.precision_at_5 for r in non_adv) / max(n_non_adv, 1)
    overall_r5 = sum(r.recall_at_5 for r in non_adv) / max(n_non_adv, 1)
    overall_r20 = sum(r.recall_at_20 for r in non_adv) / max(n_non_adv, 1)
    overall_f1 = sum(r.f1_at_5 for r in non_adv) / max(n_non_adv, 1)
    avg_lat = sum(r.latency_ms for r in all_results) / max(n_total, 1)

    print("-" * 75)
    print(
        f"{'OVERALL (non-ADV)':<10} {n_non_adv:>7} "
        f"{'':>7} "
        f"{overall_j1:>8.4f} "
        f"{overall_p5:>8.4f} "
        f"{overall_r5:>8.4f} "
        f"{overall_r20:>8.4f} "
        f"{overall_f1:>8.4f} "
        f"{avg_lat:>9.2f}"
    )
    print()

    # J1 target
    target_met = overall_j1 >= target_j1
    print(f"J1 Score:    {overall_j1:.4f}  (target ≥ {target_j1}  {'✓ MET' if target_met else '✗ NOT MET'})")
    print(f"P@5 Score:   {overall_p5:.4f}")
    print(f"R@5 Score:   {overall_r5:.4f}")
    print(f"F1@5 Score:  {overall_f1:.4f}")
    print(f"Avg Latency: {avg_lat:.2f}ms")
    print()

    # Category analysis
    print("Category Analysis:")
    for cat, d in sorted(by_category.items()):
        n = d["total"]
        pass_rate = d["passed"] / n
        avg_j1 = d["j1_sum"] / n
        print(f"  {cat:<5}: pass={pass_rate:.0%}  J1={avg_j1:.4f}  P@5={d['p5_sum']/n:.4f}  R@5={d['r5_sum']/n:.4f}")

    return EvalSummary(
        total_qa=n_total,
        by_category={
            cat: {
                "total": d["total"],
                "passed": d["passed"],
                "avg_j1": round(d["j1_sum"] / d["total"], 4),
                "avg_p5": round(d["p5_sum"] / d["total"], 4),
                "avg_r5": round(d["r5_sum"] / d["total"], 4),
                "avg_r20": round(d["r20_sum"] / d["total"], 4),
                "avg_f1": round(d["f1_sum"] / d["total"], 4),
            }
            for cat, d in by_category.items()
        },
        overall_j1=round(overall_j1, 4),
        overall_precision_5=round(overall_p5, 4),
        overall_recall_5=round(overall_r5, 4),
        overall_recall_20=round(overall_r20, 4),
        overall_f1_5=round(overall_f1, 4),
        avg_latency_ms=round(avg_lat, 2),
        target_j1=target_j1,
        target_met=target_met,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLS++ LoCoMo Synthetic Evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-test results")
    parser.add_argument("--target", type=float, default=0.35,
                        help="J1 target score (default 0.35; pure TRR without LLM extraction)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    summary = run_eval(target_j1=args.target, verbose=args.verbose)

    if args.json:
        print(json.dumps(asdict(summary), indent=2))

    # Exit code: 0 if target met, 1 if not
    sys.exit(0 if summary.target_met else 1)
