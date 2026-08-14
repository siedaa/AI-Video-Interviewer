"""Tests for the Phase 6 scorer (src/agents/scorer.py).

The scorer hits Groq, so every test injects a fake LLM caller and exercises the
pure pipeline: schema-valid output, deterministic metadata, and — critically —
the evidence guardrail rejecting a deliberately-bad entry instead of writing it.
"""

from __future__ import annotations

import datetime
import json

from src.agents.scorer import (
    github_grounded_questions_asked,
    run,
)
from src.guardrails.evidence_check import check_evidence
from src.schemas import CompetencyScore, QuestionPlan, Scorecard, Transcript, Turn

# Anchored to a real date so interview_date/duration_seconds are deterministic:
# spans 1000ms..601000ms => duration 600s, date 2026-08-14 (UTC).
_BASE_MS = int(
    datetime.datetime(2026, 8, 14, tzinfo=datetime.timezone.utc).timestamp() * 1000
)

TRANSCRIPT_TURNS = [
    Turn(
        speaker="agent",
        text="Hello Sieda Khan, welcome to your interview for the Junior AI Engineer role at Northwind Labs. First, describe a RAG system you have built.",
        timestamp_ms=_BASE_MS + 1000,
        node="intro",
    ),
    Turn(
        speaker="candidate",
        text="I built a retrieval-augmented generation system using Chroma as the vector store and FastAPI for the service layer, and I evaluated it with hit rate metrics.",
        timestamp_ms=_BASE_MS + 200000,
        node="resume_probe",
    ),
    Turn(
        speaker="agent",
        text="Walk me through your flyrank-capstone repo schema.",
        timestamp_ms=_BASE_MS + 400000,
        node="github_deepdive",
    ),
    Turn(
        speaker="candidate",
        text="Sure, I structured the database schema with ORM models and added an image-tagging schema, and I wrote unit tests with pytest.",
        timestamp_ms=_BASE_MS + 601000,
        node="github_deepdive",
    ),
]

TRANSCRIPT_TEXT = "\n".join(
    f"{'Agent' if t.speaker == 'agent' else 'Candidate'}: {t.text}"
    for t in TRANSCRIPT_TURNS
)

QUESTION_PLAN = QuestionPlan(
    questions=[
        {
            "id": "q1",
            "text": "First, describe a RAG system you have built.",
            "competency": "Technical depth",
            "source": "jd",
            "difficulty": "medium",
        },
        {
            "id": "q2",
            "text": "Walk me through your flyrank-capstone repo schema.",
            "competency": "Technical depth",
            "source": "github",
            "difficulty": "medium",
        },
        {
            "id": "q3",
            "text": "Explain the concept of RAG to a product manager.",
            "competency": "Communication",
            "source": "scenario",
            "difficulty": "easy",
        },
    ]
)

GOOD_ENTRY = CompetencyScore(
    name="Technical depth",
    score=4,
    confidence=0.85,
    evidence_quote="I built a retrieval-augmented generation system using Chroma as the vector store",
    reasoning="Candidate described a concrete RAG system with a vector store and service layer.",
)

BAD_ENTRY = CompetencyScore(
    name="Debugging",
    score=2,
    confidence=0.6,
    evidence_quote="I single-handedly built a distributed database from scratch",
    reasoning="Candidate did not actually say this.",
)


def _write_inputs(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    plan_path = tmp_path / "question_plan.json"
    transcript_path.write_text(
        Transcript(turns=TRANSCRIPT_TURNS).model_dump_json(), encoding="utf-8"
    )
    plan_path.write_text(
        json.dumps(QUESTION_PLAN.model_dump(), indent=2), encoding="utf-8"
    )
    return transcript_path, plan_path


def _base_scorecard(**overrides) -> Scorecard:
    defaults = dict(
        candidate_name="Sieda Khan",
        role="Junior AI Engineer",
        interview_date="2026-08-14",
        duration_seconds=600,
        competencies=[GOOD_ENTRY],
        overall_score=4.0,
        recommendation="hire",
        recommendation_reasoning="Strong technical depth with concrete evidence.",
        strengths=["Grounded answers about a real RAG system"],
        concerns=["Communication competency not probed"],
    )
    defaults.update(overrides)
    return Scorecard(**defaults)


def test_scorer_produces_valid_schema_with_grounded_quotes(tmp_path):
    transcript_path, plan_path = _write_inputs(tmp_path)
    out_path = tmp_path / "scorecard.json"

    def fake_llm(model_cls, system, user):
        assert model_cls is Scorecard
        return _base_scorecard()

    result = run(transcript_path, plan_path, out_path, caller=fake_llm)

    loaded = Scorecard.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert loaded.candidate_name == "Sieda Khan"
    assert loaded.role == "Junior AI Engineer"
    # duration from timestamps: (601000 - 1000) // 1000 = 600s; date 2026-08-14
    assert loaded.duration_seconds == 600
    assert loaded.interview_date == "2026-08-14"
    assert loaded.recommendation in {"hire", "no_hire", "borderline"}
    assert loaded.guardrail_flags == []
    # one github question ("Walk me through your flyrank-capstone repo schema.")
    # appears verbatim in an agent turn, so it counts as asked.
    assert github_grounded_questions_asked(QUESTION_PLAN, Transcript(turns=TRANSCRIPT_TURNS)) == 1
    assert loaded.github_grounded_questions_asked == 1
    # every written entry must pass the evidence guardrail
    for entry in loaded.competencies:
        assert check_evidence(entry, TRANSCRIPT_TEXT).passed, entry.name
    assert result.model_dump() == loaded.model_dump()


def test_bad_entry_is_excluded_and_flagged_not_written(tmp_path):
    transcript_path, plan_path = _write_inputs(tmp_path)
    out_path = tmp_path / "scorecard.json"

    def fake_llm(model_cls, system, user):
        return _base_scorecard(competencies=[GOOD_ENTRY, BAD_ENTRY], overall_score=3.0)

    # max_attempts=1 forces the exclusion path (no regeneration).
    run(transcript_path, plan_path, out_path, caller=fake_llm, max_attempts=1)

    loaded = Scorecard.model_validate_json(out_path.read_text(encoding="utf-8"))
    names = [e.name for e in loaded.competencies]
    assert "Technical depth" in names
    assert "Debugging" not in names, "failed entry must be excluded, not silently written"
    assert any("Debugging" in flag for flag in loaded.guardrail_flags)
    # overall recomputed from the accepted entry only (excluded from scoring)
    assert loaded.overall_score == 4.0
    for entry in loaded.competencies:
        assert check_evidence(entry, TRANSCRIPT_TEXT).passed


def test_bad_entry_triggers_regeneration_with_stricter_prompt(tmp_path):
    transcript_path, plan_path = _write_inputs(tmp_path)
    out_path = tmp_path / "scorecard.json"
    calls: list[str] = []

    def fake_llm(model_cls, system, user):
        calls.append(user)
        if len(calls) == 1:
            return _base_scorecard(competencies=[BAD_ENTRY])
        return _base_scorecard(competencies=[GOOD_ENTRY])

    run(transcript_path, plan_path, out_path, caller=fake_llm)

    assert len(calls) == 2, "bad entry should trigger exactly one regeneration"
    # corrective prompt must demand real transcript quotes and name the failure
    assert "evidence guardrail" in calls[1]
    assert "Debugging" in calls[1]

    loaded = Scorecard.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert loaded.guardrail_flags == [], "all entries passed after regeneration"
    for entry in loaded.competencies:
        assert check_evidence(entry, TRANSCRIPT_TEXT).passed
