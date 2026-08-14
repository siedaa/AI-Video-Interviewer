"""FirstRound MCP server — stdio transport, exposed to Claude Desktop.

Tools (exactly five, per PRD §4 Phase 6):
- get_candidate(candidate_id_or_name)   parsed resume + JD match info
- get_question_plan()                   approved question plan
- save_score(competency_name, score, evidence_quote, reasoning)
                                        append/update one competency, enforced
                                        through the evidence-check guardrail
- get_scorecard()                       full output/scorecard.json
- list_interviews()                     available interview outputs

Candidate identity: the project currently tracks ONE candidate — the flat files
in output/prep/ (resume.json, jd.json, question_plan.json) plus
output/{transcript,scorecard}.json. There is no multi-candidate ID system yet,
so get_candidate treats output/prep/ as "the" candidate and matches the argument
against the current candidate's name (case-insensitive substring). list_interviews
is written to scale if output/ ever gains multiple interviews' data.

Every tool returns a structured dict with a "status" field:
  {"status": "ok", ...}          success
  {"status": "not_found", ...}   missing file / unknown ID (never a crash)
  {"status": "error", ...}       rejected by a guardrail (save_score)

Run (stdio):
    python mcp_server/server.py            # absolute path — works from any cwd
    python -m mcp_server.server            # needs repo root on sys.path / cwd
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Resolve the repo root from this file so tools work no matter how the process
# is launched (Claude Desktop launches the script by absolute path).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastmcp import FastMCP  # noqa: E402

from src.guardrails.evidence_check import check_evidence  # noqa: E402
from src.schemas import CompetencyScore  # noqa: E402

# ---------------------------------------------------------------------------
# paths (module-level so the smoke tests can monkeypatch them)
# ---------------------------------------------------------------------------
OUTPUT = ROOT / "output"
PREP = OUTPUT / "prep"
RESUME_PATH = PREP / "resume.json"
JD_PATH = PREP / "jd.json"
PLAN_PATH = PREP / "question_plan.json"
TRANSCRIPT_PATH = OUTPUT / "transcript.json"
SCORECARD_PATH = OUTPUT / "scorecard.json"

mcp = FastMCP(
    name="firstround-mcp",
    instructions=(
        "FirstRound AI interviewer data server. Exposes the candidate's parsed "
        "resume and JD match, the approved question plan, the generated scorecard, "
        "and an evidence-checked save_score tool. All data lives in the repo's "
        "output/ directory."
    ),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    """Return parsed JSON, or None if the file is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _not_found(message: str) -> dict:
    return {"status": "not_found", "message": message}


def _transcript_text(data: dict) -> str:
    lines = []
    for turn in data.get("turns", []):
        speaker = "Agent" if turn.get("speaker") == "agent" else "Candidate"
        lines.append(f"{speaker}: {turn.get('text', '')}")
    return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _skills_overlap(must_have: list[str], skills: list[str]) -> list[str]:
    """Must-have skills that are covered by the resume's skill list.

    Soft substring matching (both directions) so compound requirements like
    "LangChain or LangGraph" match a resume listing either framework.
    """
    haystacks = [_norm(s) for s in skills]
    matched = []
    for req in must_have:
        req_norm = _norm(req)
        if not req_norm:
            continue
        if any(req_norm in h or h in req_norm for h in haystacks):
            matched.append(req)
    return matched


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_candidate(candidate_id_or_name: str) -> dict:
    """Return the parsed resume + JD match info for a candidate.

    Accepts a candidate id or name. With a single candidate in output/prep/, any
    value matching the current candidate's name (case-insensitive substring)
    returns that candidate; a non-matching value returns a not_found response.
    An empty string returns the current candidate.
    """
    resume = _read_json(RESUME_PATH)
    jd = _read_json(JD_PATH)
    if not resume:
        return _not_found(
            "no candidate found: output/prep/resume.json is missing or unreadable"
        )

    candidate_name = str(resume.get("name", "") or "")
    query = (candidate_id_or_name or "").strip()
    if query and _norm(query) not in _norm(candidate_name) and _norm(candidate_name) not in _norm(query):
        return _not_found(
            f"candidate '{query}' not found; current candidate is '{candidate_name}'"
        )

    resume_skills = [str(s) for s in resume.get("skills", [])]
    must_have = [str(s) for s in (jd or {}).get("must_have_skills", [])]
    matched = _skills_overlap(must_have, resume_skills)
    missing = [s for s in must_have if s not in matched]

    candidate = {
        "candidate_name": candidate_name,
        "github_url": resume.get("github_url"),
        "linkedin_url": resume.get("linkedin_url"),
        "roles": [
            {"title": r.get("title"), "company": r.get("company"),
             "years": r.get("years"), "description": r.get("description")}
            for r in resume.get("roles", [])
        ],
        "skills": resume_skills,
        "claims": [str(c) for c in resume.get("claims", [])],
        "jd_match": {
            "jd_role": (jd or {}).get("role"),
            "jd_company": (jd or {}).get("company"),
            "jd_location": (jd or {}).get("location"),
            "must_have_skills": must_have,
            "matched_skills": matched,
            "missing_skills": missing,
            "jd_competencies": [str(c) for c in (jd or {}).get("competencies", [])],
            "must_have_coverage": f"{len(matched)}/{len(must_have)} must-have skills present",
        },
    }
    return {"status": "ok", "candidate": candidate}


@mcp.tool()
def get_question_plan() -> dict:
    """Return the approved question plan (output/prep/question_plan.json)."""
    plan = _read_json(PLAN_PATH)
    if not plan:
        return _not_found(
            "no question plan found: output/prep/question_plan.json is missing or "
            "unreadable (run the prep pipeline + HITL approval first)"
        )
    return {
        "status": "ok",
        "approved_by_human": bool(plan.get("approved_by_human")),
        "question_count": len(plan.get("questions", [])),
        "questions": [
            {
                "id": q.get("id"),
                "text": q.get("text"),
                "competency": q.get("competency"),
                "source": q.get("source"),
                "source_reference": q.get("source_reference"),
                "difficulty": q.get("difficulty"),
            }
            for q in plan.get("questions", [])
        ],
    }


@mcp.tool()
def save_score(
    competency_name: str,
    score: int,
    evidence_quote: str,
    reasoning: str,
) -> dict:
    """Append or update one competency entry in output/scorecard.json.

    The evidence_quote is validated against the transcript via the evidence-check
    guardrail (no quote, no score) BEFORE anything is written. A quote that is
    empty or does not appear in the transcript is rejected and returned as an
    error, not saved. Updates an existing competency with the same name, appends
    otherwise, then recomputes overall_score as the mean of all scores.
    """
    transcript = _read_json(TRANSCRIPT_PATH)
    if not transcript:
        return _not_found(
            "no transcript found: output/transcript.json is missing or unreadable; "
            "cannot validate an evidence quote"
        )

    scorecard = _read_json(SCORECARD_PATH)
    if not scorecard:
        return _not_found(
            "no scorecard found: output/scorecard.json is missing or unreadable "
            "(run the scorer first)"
        )

    name = (competency_name or "").strip()
    quote = (evidence_quote or "").strip()
    if not name:
        return {"status": "error", "message": "competency_name is required"}
    if not quote:
        return {"status": "error", "message": "evidence_quote is required"}
    if not isinstance(score, int) or score < 1 or score > 5:
        return {"status": "error", "message": "score must be an integer 1-5"}

    existing = None
    for entry in scorecard.get("competencies", []):
        if (entry.get("name") or "").lower() == name.lower():
            existing = entry
            break

    entry = CompetencyScore(
        name=name,
        score=score,
        confidence=float(existing.get("confidence", 0.0)) if existing else 0.0,
        evidence_quote=quote,
        reasoning=reasoning or "",
    )

    check = check_evidence(entry, _transcript_text(transcript))
    if not check.passed:
        return {
            "status": "error",
            "message": "evidence_quote rejected by the evidence guardrail",
            "reason": check.reason,
        }

    competencies = scorecard.setdefault("competencies", [])
    if existing is not None:
        competencies[competencies.index(existing)] = entry.model_dump()
    else:
        competencies.append(entry.model_dump())

    scorecard["overall_score"] = round(
        sum(float(c.get("score", 0)) for c in competencies) / max(1, len(competencies)),
        2,
    )

    try:
        SCORECARD_PATH.write_text(
            json.dumps(scorecard, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        return {"status": "error", "message": f"could not write scorecard: {exc}"}

    return {
        "status": "ok",
        "saved": entry.model_dump(),
        "competencies_count": len(competencies),
        "overall_score": scorecard["overall_score"],
    }


@mcp.tool()
def get_scorecard() -> dict:
    """Return the full output/scorecard.json."""
    scorecard = _read_json(SCORECARD_PATH)
    if not scorecard:
        return _not_found(
            "no scorecard found: output/scorecard.json is missing or unreadable "
            "(run the scorer first)"
        )
    return {"status": "ok", "scorecard": scorecard}


@mcp.tool()
def list_interviews() -> dict:
    """List available interview outputs.

    Currently a single candidate's flat files live in output/ (one transcript +
    one scorecard). Discovery is written as a loop over the well-known artifacts
    so it scales if output/ ever holds multiple interviews' data; today it yields
    the one interview when its files exist.
    """
    transcripts = [TRANSCRIPT_PATH] if TRANSCRIPT_PATH.exists() else []
    scorecards = [SCORECARD_PATH] if SCORECARD_PATH.exists() else []

    interviews = []
    for transcript_path in transcripts:
        data = _read_json(transcript_path) or {}
        turns = data.get("turns", [])
        ts = [t.get("timestamp_ms", 0) for t in turns if t.get("timestamp_ms")]
        candidate_name = "Unknown"
        scorecard_path = None
        if scorecards:
            sc = _read_json(scorecards[0]) or {}
            candidate_name = sc.get("candidate_name") or candidate_name
            scorecard_path = str(scorecards[0])
        interviews.append(
            {
                "candidate_name": candidate_name,
                "interview_date": (
                    datetime.fromtimestamp(max(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                    if ts else ""
                ),
                "duration_seconds": (max(ts) - min(ts)) // 1000 if ts else 0,
                "turns": len(turns),
                "transcript_path": str(transcript_path),
                "scorecard_path": scorecard_path,
                "scorecard_exists": scorecard_path is not None,
            }
        )

    if not interviews:
        return {
            "status": "ok",
            "interviews": [],
            "message": "no interview outputs found in output/",
        }
    return {"status": "ok", "interviews": interviews}


if __name__ == "__main__":
    mcp.run(transport="stdio")
