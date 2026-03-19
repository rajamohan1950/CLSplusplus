"""Temporal resolution for CLS++ memory storage.

When a memory is stored with a conversation_date, relative temporal references
in the text ("yesterday", "last Tuesday", "in 3 weeks") are resolved to absolute
dates and appended inline.

Example:
    text = "I went to the doctor yesterday"
    ref  = datetime(2024, 5, 8)
    →    "I went to the doctor yesterday [7 May 2024]"

This makes the memory permanently queryable: asking "when did I go to the doctor?"
will retrieve "7 May 2024" regardless of how much time has passed.

The original wording is preserved so natural-language search still works.
The resolved date is appended in brackets so it is visible to the LLM.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Weekday helpers
# ---------------------------------------------------------------------------

_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    # Abbreviations
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

_MONTH_INDEX = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _fmt(d: datetime) -> str:
    """Format a datetime as "7 May 2024"."""
    return d.strftime("%-d %B %Y")


def _last_weekday(ref: datetime, weekday: int) -> datetime:
    """Return the most recent `weekday` strictly before `ref`."""
    days_back = (ref.weekday() - weekday) % 7 or 7
    return (ref - timedelta(days=days_back)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def _next_weekday(ref: datetime, weekday: int) -> datetime:
    """Return the next `weekday` strictly after `ref`."""
    days_fwd = (weekday - ref.weekday()) % 7 or 7
    return (ref + timedelta(days=days_fwd)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


# ---------------------------------------------------------------------------
# Pattern definitions
# Each entry: (compiled_regex, resolver(match, ref) -> str | None)
# Patterns are tried in order; first match wins per span.
# ---------------------------------------------------------------------------

def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


_PATTERNS: list[tuple[re.Pattern, object]] = [
    # ---- Numeric offsets -------------------------------------------------
    # "3 days ago", "2 weeks ago", "1 month ago"
    (_p(r'\b(\d+)\s+days?\s+ago\b'), lambda m, r: _fmt(r - timedelta(days=int(m.group(1))))),
    (_p(r'\b(\d+)\s+weeks?\s+ago\b'), lambda m, r: _fmt(r - timedelta(weeks=int(m.group(1))))),
    (_p(r'\b(\d+)\s+months?\s+ago\b'), lambda m, r: _fmt(
        r.replace(month=((r.month - int(m.group(1)) - 1) % 12) + 1,
                  year=r.year + (r.month - int(m.group(1)) - 1) // 12)
    )),

    # "in 3 days", "in 2 weeks", "in a month"
    (_p(r'\bin\s+(\d+)\s+days?\b'), lambda m, r: _fmt(r + timedelta(days=int(m.group(1))))),
    (_p(r'\bin\s+(\d+)\s+weeks?\b'), lambda m, r: _fmt(r + timedelta(weeks=int(m.group(1))))),
    (_p(r'\bin\s+a\s+week\b'), lambda m, r: _fmt(r + timedelta(weeks=1))),
    (_p(r'\bin\s+a\s+month\b'), lambda m, r: _fmt(r + timedelta(days=30))),

    # ---- Simple relative days --------------------------------------------
    # IMPORTANT: longer multi-word phrases MUST come before their sub-phrases.
    # "the day before yesterday" must precede "yesterday" — otherwise the shorter
    # pattern matches first and the covered-span mechanism blocks the longer one.
    (_p(r'\bthe day before yesterday\b'), lambda m, r: _fmt(r - timedelta(days=2))),
    (_p(r'\bthe day after tomorrow\b'), lambda m, r: _fmt(r + timedelta(days=2))),
    (_p(r'\byesterday\b'), lambda m, r: _fmt(r - timedelta(days=1))),
    (_p(r'\btoday\b'), lambda m, r: _fmt(r)),
    (_p(r'\btomorrow\b'), lambda m, r: _fmt(r + timedelta(days=1))),

    # ---- Last / this / next weekday --------------------------------------
    (_p(r'\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b'),
     lambda m, r: _fmt(_last_weekday(r, _WEEKDAY_INDEX[m.group(1).lower()]))),

    (_p(r'\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b'),
     lambda m, r: _fmt(_next_weekday(r, _WEEKDAY_INDEX[m.group(1).lower()]))),

    (_p(r'\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b'),
     lambda m, r: _fmt(_next_weekday(r, _WEEKDAY_INDEX[m.group(1).lower()]))),

    # ---- Last / this / next week -----------------------------------------
    (_p(r'\blast\s+week\b'), lambda m, r: _fmt(r - timedelta(weeks=1))),
    (_p(r'\bthis\s+week\b'), lambda m, r: _fmt(r)),
    (_p(r'\bnext\s+week\b'), lambda m, r: _fmt(r + timedelta(weeks=1))),

    # ---- Last / this / next month ----------------------------------------
    (_p(r'\blast\s+month\b'), lambda m, r: _fmt(r.replace(
        month=((r.month - 2) % 12) + 1,
        year=r.year + ((r.month - 2) // 12)
    ))),
    (_p(r'\bthis\s+month\b'), lambda m, r: r.strftime("%B %Y")),
    (_p(r'\bnext\s+month\b'), lambda m, r: (
        r.replace(month=(r.month % 12) + 1, year=r.year + (1 if r.month == 12 else 0))
          .strftime("%B %Y")
    )),

    # ---- This / last / next year ----------------------------------------
    (_p(r'\blast\s+year\b'), lambda m, r: str(r.year - 1)),
    (_p(r'\bthis\s+year\b'), lambda m, r: str(r.year)),
    (_p(r'\bnext\s+year\b'), lambda m, r: str(r.year + 1)),

    # ---- Named month + optional year (future or past) -------------------
    # "in June", "next June", "last June"
    (_p(r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december|'
        r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b'),
     lambda m, r: _resolve_named_month(m.group(1).lower(), r, future=True)),

    (_p(r'\bnext\s+(january|february|march|april|may|june|july|august|september|october|november|december|'
        r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b'),
     lambda m, r: _resolve_named_month(m.group(1).lower(), r, future=True)),

    (_p(r'\blast\s+(january|february|march|april|may|june|july|august|september|october|november|december|'
        r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b'),
     lambda m, r: _resolve_named_month(m.group(1).lower(), r, future=False)),

    # ---- Weekend references ----------------------------------------------
    (_p(r'\blast\s+weekend\b'), lambda m, r: _fmt(_last_weekday(r, 5))),
    (_p(r'\bthis\s+weekend\b'), lambda m, r: _fmt(_next_weekday(r, 5))),
    (_p(r'\bnext\s+weekend\b'), lambda m, r: _fmt(_next_weekday(r + timedelta(weeks=1), 5))),

    # ---- Morning / afternoon / evening (same day) -----------------------
    (_p(r'\bthis\s+morning\b'), lambda m, r: _fmt(r)),
    (_p(r'\bthis\s+afternoon\b'), lambda m, r: _fmt(r)),
    (_p(r'\bthis\s+evening\b'), lambda m, r: _fmt(r)),
    (_p(r'\btonight\b'), lambda m, r: _fmt(r)),
]


def _resolve_named_month(name: str, ref: datetime, *, future: bool) -> str:
    """Return "Month YYYY" for a named month relative to ref."""
    month_num = _MONTH_INDEX.get(name)
    if month_num is None:
        return ""
    year = ref.year
    if future and month_num <= ref.month:
        year += 1
    elif not future and month_num >= ref.month:
        year -= 1
    try:
        return datetime(year, month_num, 1).strftime("%B %Y")
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_relative_dates(text: str, ref_date: datetime) -> str:
    """Replace relative temporal expressions with their absolute dates.

    The relative expression is replaced IN-PLACE (not just annotated).  This
    ensures each episodic memory has a unique date as a primary token, which
    prevents the memory engine from crystallizing two visits on different days
    into a single schema just because the sentence structure is identical.

    Examples:
        "I went to the doctor yesterday"
            → "I went to the doctor on 7 May 2024"

        "Meeting tomorrow and dentist next week"
            → "Meeting on 9 May 2024 and dentist on 15 May 2024"

        "I ran a race last Saturday"
            → "I ran a race on 4 May 2024"

        "Going camping next month"
            → "Going camping in June 2024"

        "Bought this in 2021"
            → unchanged (already absolute)
    """
    if not text or not ref_date:
        return text

    # Collect all (start, end, original_text, resolved_date) without overlapping
    replacements: list[tuple[int, int, str, str]] = []
    covered: set[int] = set()

    for pattern, resolver in _PATTERNS:
        for m in pattern.finditer(text):
            span = range(m.start(), m.end())
            if any(pos in covered for pos in span):
                continue
            try:
                resolved = resolver(m, ref_date)
            except Exception:
                continue
            if not resolved:
                continue
            replacements.append((m.start(), m.end(), m.group(0), resolved))
            covered.update(span)

    if not replacements:
        return text

    replacements.sort(key=lambda x: x[0])
    result = []
    cursor = 0
    for start, end, original, resolved in replacements:
        result.append(text[cursor:start])
        # Choose preposition based on expression type
        original_lower = original.lower()
        if any(w in original_lower for w in ("next month", "this month", "last month",
                                              "next year", "this year", "last year",
                                              "in january", "in february", "in march",
                                              "in april", "in may", "in june",
                                              "in july", "in august", "in september",
                                              "in october", "in november", "in december",
                                              "next january", "next february", "last january")):
            result.append(f"in {resolved}")
        elif any(w in original_lower for w in ("next week", "this week", "last week")):
            result.append(f"the week of {resolved}")
        elif any(w in original_lower for w in ("next weekend", "this weekend", "last weekend")):
            result.append(f"the weekend of {resolved}")
        else:
            result.append(f"on {resolved}")
        cursor = end
    result.append(text[cursor:])

    return "".join(result)


def annotate_relative_dates(text: str, ref_date: datetime) -> str:
    """Append resolved absolute date in brackets after each relative expression.

    Keeps the original wording intact so the LLM reads natural language, but
    adds the resolved date for precision.  Used for the *display* version of a
    stored memory (fact.raw_text) — the engine indexes the replaced version
    produced by resolve_relative_dates() instead.

    Examples:
        "I went to the doctor yesterday"
            → "I went to the doctor yesterday [7 May 2024]"

        "Meeting tomorrow and dentist next week"
            → "Meeting tomorrow [9 May 2024] and dentist next week [15 May 2024]"
    """
    if not text or not ref_date:
        return text

    replacements: list[tuple[int, int, str, str]] = []
    covered: set[int] = set()

    for pattern, resolver in _PATTERNS:
        for m in pattern.finditer(text):
            span = range(m.start(), m.end())
            if any(pos in covered for pos in span):
                continue
            try:
                resolved = resolver(m, ref_date)
            except Exception:
                continue
            if not resolved:
                continue
            replacements.append((m.start(), m.end(), m.group(0), resolved))
            covered.update(span)

    if not replacements:
        return text

    replacements.sort(key=lambda x: x[0])
    result = []
    cursor = 0
    for start, end, original, resolved in replacements:
        result.append(text[cursor:end])       # original expression kept verbatim
        result.append(f" [{resolved}]")        # resolved date appended in brackets
        cursor = end
    result.append(text[cursor:])
    return "".join(result)


def date_label(ref_date: datetime) -> str:
    """Return a human-readable label like "8 May 2024 (Wednesday)" for context headers."""
    return ref_date.strftime("%-d %B %Y (%A)")
