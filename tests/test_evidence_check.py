"""Tests for the evidence-check guardrail (Phase 3b).

Covers: empty quote rejected, non-matching quote rejected, verbatim matching
quote passes, and a fuzzy (slightly-different) quote passes.
"""

from __future__ import annotations

from src.guardrails.evidence_check import check_evidence
from src.schemas import CompetencyScore

TRANSCRIPT = (
    "Agent: Can you tell me about the RAG system you built?\n"
    "Candidate: I built a retrieval-augmented generation system using "
    "Chroma as the vector store and FastAPI for the service layer.\n"
    "Agent: How did you evaluate retrieval quality?\n"
    "Candidate: I measured hit rate and used human review of the top five "
    "results for each query.\n"
)


def _entry(quote: str, name: str = "RAG") -> CompetencyScore:
    return CompetencyScore(name=name, score=4, confidence=0.8, evidence_quote=quote)


def test_empty_evidence_quote_is_rejected():
    check = check_evidence(_entry("   "), TRANSCRIPT)
    assert not check.passed
    assert "empty" in check.reason.lower()


def test_quote_not_in_transcript_is_rejected():
    fabricated = "I single-handedly built a distributed database from scratch."
    check = check_evidence(_entry(fabricated), TRANSCRIPT)
    assert not check.passed
    assert "does not appear" in check.reason


def test_verbatim_quote_passes():
    quote = "I built a retrieval-augmented generation system using Chroma as the vector store"
    check = check_evidence(_entry(quote), TRANSCRIPT)
    assert check.passed
    assert "verbatim" in check.reason


def test_quote_with_punctuation_differences_passes():
    quote = "I measured hit rate, and used human review of the top five results"
    check = check_evidence(_entry(quote), TRANSCRIPT)
    assert check.passed
    assert "close fuzzy match" in check.reason


def test_fuzzy_match_with_stray_filler_word_passes():
    quote = "measured hit rate and used a human review of the top results for each query"
    check = check_evidence(_entry(quote), TRANSCRIPT)
    assert check.passed
