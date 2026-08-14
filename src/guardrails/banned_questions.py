"""Guardrail: block banned interview questions before they reach the candidate.

Two-layer design:

1. **Keyword/pattern pre-filter** (`keyword_screen`) — a fast, deterministic
   regex scan over the question text for direct mentions of each banned
   category (age, gender, marital status, religion, nationality,
   health/pregnancy, salary history, politics). Catches obvious cases in
   microseconds with zero external calls.

2. **LLM classifier** (`llm_screen`) — a Groq structured-output call (via
   `src.agents.groq_client.chat_structured`) for subtle/adversarial phrasings
   that imply a banned topic without naming it (e.g. "how do you balance work
   with family responsibilities" implying marital/parental status, or "where
   are you originally from" implying nationality).

`check_question` runs layer 1 first and short-circuits on a hit; otherwise it
falls through to layer 2. Returns a `BannedQuestionCheck` with the category
matched and which layer caught it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from src.agents.groq_client import chat_structured

BANNED_CATEGORIES = [
    "age",
    "gender",
    "marital_status",
    "religion",
    "nationality",
    "health_pregnancy",
    "salary_history",
    "politics",
]

# Keyword pre-filter: category -> compiled regexes. Kept deliberately broad;
# false positives here are safer than a banned question reaching a candidate.
_KEYWORD_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "age": [
        re.compile(r"\bage\b|\bage\s+(?:group|range)\b", re.IGNORECASE),
        re.compile(r"\bhow old\b|\byear born\b|\bbirth\s*(?:year|date)\b", re.IGNORECASE),
        re.compile(r"\b(?:young|old)\s+enough\b|\bunderage\b|\bminor\b", re.IGNORECASE),
    ],
    "gender": [
        re.compile(r"\bgender\b|\bsex\b|\bmale\b|\bfemale\b|\bman\s+or\s+woman\b", re.IGNORECASE),
        re.compile(r"\btransgender\b|\bcisgender\b|\bnontraditional\s+gender\b", re.IGNORECASE),
    ],
    "marital_status": [
        re.compile(r"\bmarried\b|\bmarital\b|\bsingle\b|\bspouse\b|\bhusband\b|\bwife\b", re.IGNORECASE),
        re.compile(r"\bdivorced\b|\bseparated\b|\bwidow(?:er)?\b|\bpartner\b|\bchildren\b", re.IGNORECASE),
        re.compile(r"\bfamily\s+plans?\b|\bplan(?:s)?\s+to\s+(?:have\s+)?kids\b|\bparental\s+status\b", re.IGNORECASE),
    ],
    "religion": [
        re.compile(r"\breligion\b|\breligious\b|\bfaith\b|\bbelief(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bchristian\b|\bmuslim\b|\bhindu\b|\bbuddhist\b|\bjewish\b|\bsikh\b|\bcatholic\b", re.IGNORECASE),
        re.compile(r"\bmosque\b|\bchurch\b|\btemple\b|\bsynagogue\b|\bpray(?:er|ers)?\b|\bquran\b|\bbible\b", re.IGNORECASE),
    ],
    "nationality": [
        re.compile(r"\bnationality\b|\bnational\s+origin\b|\bcitizen(?:ship)?\b|\bimmigrant\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+(?:are|were)\s+you\s+(?:originally\s+)?from\b|\bwhere\s+do\s+you\s+come\s+from\b", re.IGNORECASE),
        re.compile(r"\bpassport\b|\bvisa\s+(?:status|type)\b|\bethnicity\b|\brace\b", re.IGNORECASE),
    ],
    "health_pregnancy": [
        re.compile(r"\bhealth\s+(?:condition|issues?|problems?|status)\b|\bdisability\b|\bmedical\s+history\b", re.IGNORECASE),
        re.compile(r"\bpregnan(?:t|cy)\b|\bexpecting\b|\bmaternity\b|\bpaternity\b", re.IGNORECASE),
        re.compile(r"\bsick\s+(?:leave|days)\b|\bmental\s+health\b|\bphysical\s+fitness\b|\bweight\b", re.IGNORECASE),
    ],
    "salary_history": [
        re.compile(r"\bsalary\s+history\b|\bcurrent\s+salary\b|\bprevious\s+salary\b|\bpast\s+salary\b", re.IGNORECASE),
        re.compile(r"\bhow\s+much\s+(?:do|did|are)\s+you\s+(?:make|earn|paid|getting)\b|\bcurrent\s+pay\b", re.IGNORECASE),
        re.compile(r"\bcompensation\s+history\b|\bwhat\s+are\s+you\s+currently\s+paid\b|\bsalary\s+range\s+at\s+(?:your\s+)?(?:last|previous|current)\b", re.IGNORECASE),
    ],
    "politics": [
        re.compile(r"\bpolitics\b|\bpolitical\s+(?:views?|affiliation|leanings?|party)\b", re.IGNORECASE),
        re.compile(r"\bvote(?:d|s)?\b|\bvoting\b|\bcandidate\s+(?:for|in)\b|\belection(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bdemocrat\b|\brepublican\b|\bliberal\b|\bconservative\b|\bsocialist\b", re.IGNORECASE),
    ],
}


@dataclass
class BannedQuestionCheck:
    """Result of screening one interview question."""

    question: str = ""
    blocked: bool = False
    category: str = ""
    layer: str = ""
    reason: str = ""


class _LLMDecision(BaseModel):
    blocked: bool = False
    category: str = ""
    reason: str = ""


def keyword_screen(question: str) -> BannedQuestionCheck | None:
    """Layer 1: deterministic keyword/pattern scan. Returns a check on a hit."""
    for category, patterns in _KEYWORD_PATTERNS.items():
        for pat in patterns:
            if pat.search(question):
                return BannedQuestionCheck(
                    question=question,
                    blocked=True,
                    category=category,
                    layer="keyword",
                    reason=f"matched banned topic '{category}' via pattern {pat.pattern!r}",
                )
    return None


_LLM_SYSTEM = """You are a strict interview-question safety filter. A question is
BANNED if it asks about or probes any of these protected topics, directly OR by
implication:
- age
- gender
- marital status (including family/parental status or plans)
- religion
- nationality / national origin / ethnicity / race
- health, pregnancy, disability, or medical history
- salary history (past/current pay)
- politics / political views

Indirect phrasings count too. Examples:
- "How do you balance work with family responsibilities?" -> blocked (marital_status)
- "Where are you originally from?" -> blocked (nationality)
- "Do you have any commitments that would limit your hours?" -> blocked (marital_status/family)
- "Tell me about a technical project you shipped." -> NOT blocked

Reply with JSON: {"blocked": bool, "category": one of age, gender, marital_status,
religion, nationality, health_pregnancy, salary_history, politics (empty string if
not blocked), "reason": short explanation}."""


def llm_screen(
    question: str,
    model: str = "openai/gpt-oss-120b",
    caller: Callable[..., _LLMDecision] | None = None,
) -> BannedQuestionCheck:
    """Layer 2: Groq structured-output classifier for subtle phrasings."""
    call = caller if caller is not None else _default_llm_call
    decision = call(question, model)
    if not decision.blocked:
        return BannedQuestionCheck(question=question, blocked=False)
    return BannedQuestionCheck(
        question=question,
        blocked=True,
        category=decision.category,
        layer="llm",
        reason=decision.reason or f"classified as '{decision.category}' by LLM",
    )


def _default_llm_call(question: str, model: str) -> _LLMDecision:
    return chat_structured(
        _LLMDecision,
        _LLM_SYSTEM,
        f"Screen this interview question:\n{question}",
        model=model,
    )


def check_question(
    question: str,
    use_llm: bool = True,
    llm_caller: Callable[..., _LLMDecision] | None = None,
) -> BannedQuestionCheck:
    """Screen a question with keyword pre-filter, then LLM if still clean."""
    hit = keyword_screen(question)
    if hit is not None:
        return hit
    if not use_llm:
        return BannedQuestionCheck(question=question, blocked=False)
    return llm_screen(question, caller=llm_caller)
