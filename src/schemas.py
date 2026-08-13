"""Pydantic schemas copied verbatim from FirstRound-Final-Test.pdf §6.

Extra fields are allowed by the grader; missing fields are not. We mirror the
spec exactly and keep the field sets deliberately minimal. GraphState extends
the schema set with the LangGraph runtime state (not part of the graded files).
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

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


# ---------------------------------------------------------------------------
# LangGraph runtime state (Phase 2, not part of the graded output files)
# ---------------------------------------------------------------------------
class ApprovalEdit(BaseModel):
    """A single recruiter edit applied to the question plan."""

    question_id: str = ""
    field: str = ""
    old_value: str = ""
    new_value: str = ""
    timestamp: str = ""


class GraphState(TypedDict, total=False):
    """State flowing through the FirstRound LangGraph (Phase 2+).

    TypedDict (not a Pydantic model) so LangGraph can apply its update /
    reducer semantics out of the box. All fields optional so partial inputs
    are valid.

    Fields:
      inputs        - raw file paths (jd, resume) handed to the graph
      jd            - JDInfo model as parsed by parse_jd
      resume        - ResumeInfo model as parsed by parse_resume
      github_findings - GithubInfo model from github_agent
      question_plan - QuestionPlan model from question_planner
      thread_id     - stable id keying the checkpointer / resumability
      approval_status - "pending" until the recruiter decides (Phase 2 HITL)
      edits_made    - list of recruiter edits (see ApprovalEdit)
    """

    inputs: dict[str, str]
    jd: JDInfo
    resume: ResumeInfo
    github_findings: GithubInfo
    question_plan: QuestionPlan
    thread_id: str
    approval_status: Literal["pending", "approved", "edited", "rejected"]
    edits_made: List[ApprovalEdit]