"""Pydantic schemas copied verbatim from FirstRound-Final-Test.pdf §6.

Extra fields are allowed by the grader; missing fields are not. We mirror the
spec exactly and keep the field sets deliberately minimal.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# output/prep/jd.json
# ---------------------------------------------------------------------------
class JDInfo(BaseModel):
    role: str = ""
    seniority: str = ""
    company: Optional[str] = ""
    location: Optional[str] = ""
    must_have_skills: List[str] = Field(default_factory=list)
    competencies: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# output/prep/resume.json
# ---------------------------------------------------------------------------
class ResumeRole(BaseModel):
    title: str = ""
    company: str = ""
    years: str = ""
    description: str = ""


class ResumeInfo(BaseModel):
    name: str = ""
    roles: List[ResumeRole] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None


# ---------------------------------------------------------------------------
# output/prep/github.json
# ---------------------------------------------------------------------------
class CommitInfo(BaseModel):
    message: str = ""
    date: str = ""


class FileEvidence(BaseModel):
    path: str = ""
    excerpt: str = ""


class RepoInfo(BaseModel):
    name: str = ""
    full_name: str = ""
    language: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    description: str = ""
    updated_at: str = ""
    pushed_at: str = ""
    stars: int = 0
    top_repository: bool = False
    recent_commits: List[CommitInfo] = Field(default_factory=list)
    readme_excerpt: Optional[str] = None
    files_read: List[FileEvidence] = Field(default_factory=list)


class GithubInfo(BaseModel):
    username: str = ""
    profile_url: str = ""
    repos: List[RepoInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# output/prep/question_plan.json
# ---------------------------------------------------------------------------
class Question(BaseModel):
    id: str = ""
    text: str = ""
    competency: str = ""
    source: Literal["jd", "resume", "github", "scenario"] = "jd"
    source_reference: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    follow_up_triggers: List[str] = Field(default_factory=list)


class QuestionPlan(BaseModel):
    questions: List[Question] = Field(default_factory=list)
    approved_by_human: bool = False
    edits_made: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# output/transcript.json
# ---------------------------------------------------------------------------
class Turn(BaseModel):
    speaker: Literal["agent", "candidate"] = "agent"
    text: str = ""
    timestamp_ms: int = 0
    node: str = ""
    interrupted: bool = False


class Transcript(BaseModel):
    turns: List[Turn] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# output/scorecard.json
# ---------------------------------------------------------------------------
class CompetencyScore(BaseModel):
    name: str = ""
    score: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_quote: str = ""
    reasoning: str = ""


class Scorecard(BaseModel):
    candidate_name: str = ""
    role: str = ""
    interview_date: str = ""
    duration_seconds: int = 0
    competencies: List[CompetencyScore] = Field(default_factory=list)
    overall_score: float = 0.0
    recommendation: Literal["hire", "no_hire", "borderline"] = "borderline"
    recommendation_reasoning: str = ""
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    guardrail_flags: List[str] = Field(default_factory=list)
    github_grounded_questions_asked: int = 0