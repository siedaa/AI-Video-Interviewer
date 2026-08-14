"""Phase 6 agent: score a live interview transcript into output/scorecard.json.

Takes output/transcript.json + output/prep/question_plan.json and uses Groq
structured output (chat_structured, same pattern as jd_parser / resume_parser /
question_planner) to score every competency in the question plan against the
transcript. The hard rule is "no quote, no score": every competency entry is
run through src.guardrails.evidence_check before the file is written. Entries
that fail are regenerated with a stricter corrective prompt (up to MAX_ATTEMPTS
calls); anything still failing is excluded from scoring and recorded in
guardrail_flags — never silently written.

Deterministic fields (computed in Python, overriding whatever the LLM emits):
- duration_seconds / interview_date  from the transcript timestamps
- github_grounded_questions_asked   from question-plan matches in agent turns
- overall_score                     recomputed as the mean of accepted scores
                                     whenever any entry is excluded
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from src.agents.groq_client import DEFAULT_MODEL, chat_structured
from src.guardrails.evidence_check import check_evidence
from src.schemas import CompetencyScore, QuestionPlan, Scorecard, Transcript

MATCH_THRESHOLD = 0.3
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a rigorous hiring evaluator for the Junior AI Engineer role
at Northwind Labs. You are given a real interview transcript and the approved question
plan. Produce a scorecard where EVERY score is grounded in a REAL quote from the
transcript.

Hard rules:
- evidence_quote MUST be a real, verbatim (or near-verbatim) substring of the
  candidate's spoken text in the transcript. Copy the candidate's actual words.
  NEVER fabricate, paraphrase into new wording, or quote the wrong speaker.
- score: integer 1-5. 1 = clearly failed, 3 = acceptable, 5 = exceptional.
- confidence: float 0.0-1.0 = how much direct evidence supports the score (lower
  when evidence is thin or indirect).
- Score EVERY competency from the provided list, at least one entry per competency.
- overall_score: float 0.0-5.0 summarizing overall performance across competencies.
- recommendation: one of "hire", "no_hire", "borderline".
- strengths and concerns: 1-4 concrete, specific bullet strings each, each traceable
  to the transcript.
- candidate_name and role: read from the transcript (the interviewer greets the
  candidate by name and states the role). Do not invent.
- interview_date and duration_seconds are provided in the prompt — use them.
- Do not emit empty or invented quotes. If a competency has genuinely no supporting
  answer in the transcript, still score it but set evidence_quote to the closest
  REAL quote you can find and keep confidence low."""


def _llm_call(
    model_cls: type[Scorecard],
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> Scorecard:
    return chat_structured(model_cls, system, user, model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# deterministic metadata
# ---------------------------------------------------------------------------


def _transcript_text(transcript: Transcript) -> str:
    lines = []
    for turn in transcript.turns:
        speaker = "Agent" if turn.speaker == "agent" else "Candidate"
        lines.append(f"{speaker}: {turn.text}")
    return "\n".join(lines)


def _duration_seconds(transcript: Transcript) -> int:
    ts = [t.timestamp_ms for t in transcript.turns if t.timestamp_ms]
    if not ts:
        return 0
    return max(0, (max(ts) - min(ts)) // 1000)


def _interview_date(transcript: Transcript) -> str:
    ts = [t.timestamp_ms for t in transcript.turns if t.timestamp_ms]
    if not ts:
        return ""
    return datetime.fromtimestamp(max(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def github_grounded_questions_asked(plan: QuestionPlan, transcript: Transcript) -> int:
    """Count question-plan github questions that appear in agent turns.

    Mirrors the TranscriptTracker heuristic in src/realtime/agent.py:
    SequenceMatcher token-overlap against the question text, threshold 0.3.
    """
    agent_texts = [t.text for t in transcript.turns if t.speaker == "agent"]
    count = 0
    for q in plan.questions:
        if q.source != "github":
            continue
        for text in agent_texts:
            if SequenceMatcher(None, text, q.text).ratio() >= MATCH_THRESHOLD:
                count += 1
                break
    return count


def _stamp_metadata(
    scorecard: Scorecard, transcript: Transcript, plan: QuestionPlan
) -> None:
    """Override LLM-produced metadata with deterministic values where we have them."""
    date_str = _interview_date(transcript)
    if date_str:
        scorecard.interview_date = date_str
    scorecard.duration_seconds = _duration_seconds(transcript)
    scorecard.github_grounded_questions_asked = github_grounded_questions_asked(plan, transcript)
    if not (scorecard.candidate_name or "").strip():
        scorecard.candidate_name = "Unknown"


# ---------------------------------------------------------------------------
# evidence guardrail (no quote, no score)
# ---------------------------------------------------------------------------


def evidence_failures(
    scorecard: Scorecard, transcript_text: str
) -> list[tuple[CompetencyScore, str]]:
    """Every competency whose evidence_quote fails check_evidence, with the reason."""
    failures: list[tuple[CompetencyScore, str]] = []
    for entry in scorecard.competencies:
        check = check_evidence(entry, transcript_text)
        if not check.passed:
            failures.append((entry, check.reason))
    return failures


def finalize_scorecard(scorecard: Scorecard, transcript_text: str) -> Scorecard:
    """Exclude failed entries from scoring; flag them instead of writing them.

    Recomputes overall_score as the mean of the accepted scores (the failed
    entries are excluded from scoring, per spec). Returns the mutated scorecard.
    """
    failures = evidence_failures(scorecard, transcript_text)
    if not failures:
        return scorecard

    failed_names = {entry.name for entry, _ in failures}
    accepted = [e for e in scorecard.competencies if e.name not in failed_names]

    flags = list(scorecard.guardrail_flags)
    for entry, reason in failures:
        flags.append(f"competency '{entry.name}' rejected: {reason}")
    scorecard.guardrail_flags = flags

    scorecard.competencies = accepted
    if accepted:
        scorecard.overall_score = round(sum(e.score for e in accepted) / len(accepted), 2)
    else:
        scorecard.overall_score = 0.0
    return scorecard


# ---------------------------------------------------------------------------
# scoring pipeline
# ---------------------------------------------------------------------------


def _build_user_prompt(transcript_text: str, plan: QuestionPlan) -> str:
    competencies = sorted({q.competency for q in plan.questions if q.competency})
    plan_lines = "\n".join(f"- [{q.source}] {q.text}" for q in plan.questions)
    return f"""=== INTERVIEW TRANSCRIPT ===
{transcript_text}

=== APPROVED QUESTION PLAN ===
{plan_lines}

=== COMPETENCIES TO SCORE ===
{", ".join(competencies) if competencies else "(none specified)"}

Score every competency listed above. Extract candidate_name and role from the
transcript text. Produce the scorecard now."""


def score_models(
    transcript: Transcript,
    plan: QuestionPlan,
    *,
    caller: Callable[..., Scorecard] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> Scorecard:
    """Core pipeline: LLM-draft the scorecard, then enforce the evidence guardrail.

    Retries with a corrective prompt whenever the evidence guardrail rejects an
    entry; after max_attempts, still-failing entries are excluded and flagged.
    """
    call = caller if caller is not None else _llm_call
    transcript_text = _transcript_text(transcript)
    base_user = _build_user_prompt(transcript_text, plan)

    draft: Scorecard | None = None
    last_err: str | None = None

    for attempt in range(1, max_attempts + 1):
        extra = ""
        if draft is not None:
            failures = evidence_failures(draft, transcript_text)
            if failures:
                feedback = "\n".join(f"- '{e.name}': {reason}" for e, reason in failures)
                extra = (
                    "\n\nThe previous scorecard was rejected by the evidence guardrail."
                    "\nEvery evidence_quote MUST be a real substring of the transcript — "
                    "copy the candidate's actual words, never fabricate or paraphrase."
                    "\n\nRejected entries:"
                    f"\n{feedback}"
                    "\n\nReturn the complete corrected scorecard."
                )
            elif last_err:
                extra = (
                    "\n\nThe previous response failed to parse into the scorecard schema"
                    f" ({last_err}). Fix it and return the complete scorecard."
                )

        try:
            draft = call(Scorecard, SYSTEM_PROMPT, base_user + extra)
            _stamp_metadata(draft, transcript, plan)
            if not evidence_failures(draft, transcript_text):
                return draft
            last_err = None
        except Exception as exc:  # parse or API error -> plain retry next iteration
            last_err = f"{type(exc).__name__}: {exc}"
            print(f"[scorer] LLM call failed on attempt {attempt}: {last_err}")
            draft = None

    if draft is None:
        raise RuntimeError(f"[scorer] LLM failed after {max_attempts} attempts: {last_err}")

    for entry, reason in evidence_failures(draft, transcript_text):
        print(f"[scorer] excluding competency '{entry.name}' after {max_attempts} attempts: {reason}")
    return finalize_scorecard(draft, transcript_text)


def run(
    transcript_path: str | Path,
    plan_path: str | Path,
    output_path: str | Path,
    *,
    caller: Callable[..., Scorecard] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> Scorecard:
    transcript_path = Path(transcript_path)
    plan_path = Path(plan_path)
    output_path = Path(output_path)

    transcript = Transcript.model_validate_json(
        transcript_path.read_text(encoding="utf-8")
    )
    plan = QuestionPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))

    print(
        f"[scorer] scoring {transcript_path} against {plan_path} "
        f"({len(transcript.turns)} turns, {len(plan.questions)} questions)..."
    )
    scorecard = score_models(transcript, plan, caller=caller, max_attempts=max_attempts)

    # defense in depth: the written file must round-trip through the schema
    Scorecard.model_validate(scorecard.model_dump())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(scorecard.model_dump(), indent=2), encoding="utf-8"
    )
    print(
        f"[scorer] wrote {output_path} "
        f"({len(scorecard.competencies)} competencies, "
        f"overall={scorecard.overall_score}, "
        f"recommendation={scorecard.recommendation})"
    )
    return scorecard


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(__file__).resolve().parents[2]
    transcript_path = Path(argv[0]) if len(argv) >= 1 else root / "output" / "transcript.json"
    plan_path = Path(argv[1]) if len(argv) >= 2 else root / "output" / "prep" / "question_plan.json"
    out_path = Path(argv[2]) if len(argv) >= 3 else root / "output" / "scorecard.json"
    report_path = Path(argv[3]) if len(argv) >= 4 else root / "output" / "report.pdf"
    run(transcript_path, plan_path, out_path)
    from src.agents import report_generator

    report_generator.run(out_path, transcript_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
