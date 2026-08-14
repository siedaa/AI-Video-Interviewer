"""Guardrail: reject scorecard competency entries with no real transcript evidence.

The scoring rule is "no quote, no score" — every competency in a scorecard must
back its score with an `evidence_quote` that actually appears in the interview
transcript. This module enforces that before a scorecard is written.

`check_evidence(entry, transcript)` returns an `EvidenceCheck`:
- fails if `evidence_quote` is empty/whitespace, or
- fails if the quote does not appear in the transcript as a matching substring
  or a close fuzzy match (SequenceMatcher ratio over the whole transcript,
  after normalizing whitespace/case). A close match tolerates transcription
  artifacts like dropped punctuation or a stray filler word, while still
  rejecting quotes that were invented or copied from another source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.schemas import CompetencyScore

FUZZY_THRESHOLD = 0.90


@dataclass
class EvidenceCheck:
    """Result of validating one competency entry's evidence quote."""

    name: str = ""
    passed: bool = False
    reason: str = ""


def _normalize(text: str) -> str:
    """Lowercase and collapse runs of whitespace for tolerant matching."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip().lower()


def _contains_substring(quote_norm: str, transcript_norm: str) -> bool:
    return quote_norm in transcript_norm


def _close_match(quote_norm: str, transcript_norm: str) -> bool:
    """True if the quote closely matches some window of the transcript.

    Sliding-window comparison tolerates transcription artifacts — dropped
    punctuation, stray filler words, minor rephrasing — while still rejecting
    quotes that were invented. Window lengths bracket the quote length by a
    few characters so small insertions/deletions don't break alignment.
    """
    qlen = len(quote_norm)
    tlen = len(transcript_norm)
    if qlen == 0 or tlen == 0 or qlen > tlen:
        return False

    for wlen in range(max(1, qlen - 5), min(tlen, qlen + 6) + 1):
        step = max(1, wlen // 4)
        for start in range(0, tlen - wlen + 1, step):
            window = transcript_norm[start : start + wlen]
            ratio = SequenceMatcher(None, quote_norm, window).ratio()
            if ratio >= FUZZY_THRESHOLD:
                return True
    return False


def check_evidence(entry: CompetencyScore, transcript: str) -> EvidenceCheck:
    """Validate a competency entry against the transcript text."""
    quote = (entry.evidence_quote or "").strip()
    name = entry.name or ""

    if not quote:
        return EvidenceCheck(name=name, passed=False, reason="evidence_quote is empty")

    transcript_norm = _normalize(transcript)
    quote_norm = _normalize(quote)

    if _contains_substring(quote_norm, transcript_norm):
        return EvidenceCheck(name=name, passed=True, reason="quote found verbatim in transcript")

    if _close_match(quote_norm, transcript_norm):
        return EvidenceCheck(name=name, passed=True, reason="quote found as close fuzzy match in transcript")

    return EvidenceCheck(
        name=name,
        passed=False,
        reason="evidence_quote does not appear in the transcript",
    )
