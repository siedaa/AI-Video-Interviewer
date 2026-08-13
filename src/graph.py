"""Phase 2: LangGraph wrapping the Phase 1 prep pipeline + HITL gate.

Graph structure (plain-text version for ARCHITECTURE.md):

    START
      |
      v
    [parse_jd]        node  : read inputs/jd.txt -> state["jd"]
      |
      v
    [parse_resume]    node  : read inputs/resume.pdf -> state["resume"]
      |
      v
    CONDITIONAL EDGE #1  ("route_github")  <-- real, data-dependent
      | resume.github_url present?
      |   yes -> [github_agent]
      |   no  -> [request_github_manually]  (interrupt: CLI asks for username)
      |                    |
      |                    v
      |              (override merged into resume.github_url, edge to github_agent)
      v
    [github_agent]    node  : state["resume"].github_url + GITHUB_PAT
      |                     -> state["github_findings"] (repos, languages, commits,
      |                        README + real file excerpts for top 3)
      v
    [question_planner] node : state["jd"] + resume + github_findings
      |                     -> state["question_plan"] (12 questions)
      |                        state["approval_status"] = "pending"
      v
    [hitl_gate]       node  : interrupt() -- graph SUSPENDS, returns plan to CLI.
      |   CLI returns Command(resume={"action": ...}) via the same thread_id.
      |   action == "approve" : writes output/prep/question_plan.json,
      |                         approval_status = "approved"
      |   action == "edit"    : applies edits -> edits_made += [...],
      |                         approval_status = "edited"
      |   action == "reject"  : approval_status = "rejected" (no file written)
      v
    CONDITIONAL EDGE #2  ("route_approval")
      | approval_status == "approved" -> END
      | approval_status == "edited"   -> [hitl_gate] again  (re-review loop)
      | approval_status == "rejected" -> END
      v
    END

Interrupts (HITL):
  - [request_github_manually] interrupts with
        {"type": "request_github", ...}; CLI resumes with
        Command(resume={"github_url": "https://github.com/<username>"}).
  - [hitl_gate] interrupts with
        {"type": "question_plan_review", "plan": {...}, ...}; CLI resumes with
        Command(resume={"action": "approve" | "edit" | "reject", ...}).

Checkpointer: SqliteSaver on checkpoint.db, keyed by thread_id. A dropped
call / restart can resume this thread's exact state by re-invoking with the
same thread_id.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.schemas import (
    ApprovalEdit,
    CommitInfo,
    CompetencyScore,
    FileEvidence,
    GraphState,
    GithubInfo,
    JDInfo,
    Question,
    QuestionPlan,
    RepoInfo,
    ResumeInfo,
    ResumeRole,
    Scorecard,
    Transcript,
    Turn,
)
from src.agents import github_agent, jd_parser, question_planner, resume_parser

ROOT = Path(__file__).resolve().parents[1]
PREP_DIR = ROOT / "output" / "prep"
CHECKPOINT_DB = ROOT / "checkpoint.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dump(model, name: str) -> None:
    """Write a Pydantic model to output/prep/<name>.json."""
    path = PREP_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.model_dump(), indent=2), encoding="utf-8")
    print(f"[graph] wrote {path}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def parse_jd(state: GraphState) -> dict[str, Any]:
    """Read the JD text file and parse it into state["jd"]."""
    jd_path = Path(state["inputs"]["jd"])
    text = jd_path.read_text(encoding="utf-8", errors="replace")
    print(f"[graph] parse_jd: {jd_path.name}")
    info = jd_parser.parse_jd(text)
    _dump(info, "jd")
    return {"jd": info}


def parse_resume(state: GraphState) -> dict[str, Any]:
    """Extract resume text with pdfplumber and parse into state["resume"]."""
    pdf_path = Path(state["inputs"]["resume"])
    print(f"[graph] parse_resume: {pdf_path.name}")
    text = resume_parser.extract_pdf_text(pdf_path)
    info = resume_parser.parse_resume(text)
    _dump(info, "resume")
    return {"resume": info}


def route_github(state: GraphState) -> str:
    """Conditional edge #1: does the parsed resume have a GitHub URL?"""
    resume: ResumeInfo = state["resume"]
    if resume.github_url:
        return "github_agent"
    return "request_github_manually"


def request_github_manually(state: GraphState) -> dict[str, Any]:
    """HITL pause: ask the CLI for the candidate's GitHub username.

    Interrupts with a payload; the CLI prompts the user and resumes with
    Command(resume={"github_url": "https://github.com/<username>"}).
    """
    print("[graph] request_github_manually: no GitHub URL in resume")
    override = interrupt(
        {
            "type": "request_github",
            "message": "No GitHub URL found in the resume. "
            "Please provide the candidate's GitHub username.",
        }
    )
    github_url = (override or {}).get("github_url", "")
    if not github_url:
        raise ValueError("CLI resumed request_github_manually without github_url")
    resume = state["resume"].model_copy(update={"github_url": github_url})
    _dump(resume, "resume")
    return {"resume": resume}


def run_github_agent(state: GraphState) -> dict[str, Any]:
    """Pull real GitHub evidence and store it in state["github_findings"]."""
    resume: ResumeInfo = state["resume"]
    jd: JDInfo | None = state.get("jd")
    if not resume.github_url:
        raise ValueError("No github_url available in github_agent node")
    print(f"[graph] github_agent: pulling data for {resume.github_url}")
    info = github_agent.gather_github(resume.github_url, resume, jd)
    _dump(info, "github")
    return {"github_findings": info}


def run_question_planner(state: GraphState) -> dict[str, Any]:
    """Build the 12-question plan; set approval_status to pending."""
    print("[graph] question_planner: generating plan")
    plan = question_planner.generate_plan(
        state["jd"],
        state["resume"],
        state["github_findings"],
    )
    plan.approved_by_human = False
    plan.edits_made = []
    return {"question_plan": plan, "approval_status": "pending"}


def hitl_gate(state: GraphState) -> dict[str, Any]:
    """HITL gate: suspend and let the recruiter approve/edit/reject the plan.

    The graph genuinely pauses at interrupt(). The CLI resumes the SAME
    thread (same thread_id / checkpointer) with Command(resume=decision).
    """
    plan: QuestionPlan = state["question_plan"]
    decision: dict[str, Any] = interrupt(
        {
            "type": "question_plan_review",
            "plan": plan.model_dump(),
            "approval_status": state.get("approval_status", "pending"),
        }
    )
    action = decision.get("action")

    if action == "approve":
        plan.approved_by_human = True
        plan.edits_made = [
            r.model_dump() if isinstance(r, ApprovalEdit) else r
            for r in state.get("edits_made", [])
        ]
        _dump(plan, "question_plan")
        return {"approval_status": "approved", "question_plan": plan}

    if action == "edit":
        existing = state.get("edits_made", [])
        new_plan, new_edits = _apply_edits(plan, decision.get("edits", []), existing)
        return {
            "approval_status": "edited",
            "question_plan": new_plan,
            "edits_made": existing + new_edits,
        }

    if action == "reject":
        return {"approval_status": "rejected"}

    raise ValueError(f"Unknown HITL action: {action!r}")


def route_approval(state: GraphState) -> str:
    """Conditional edge #2: route on the recruiter's decision."""
    status = state.get("approval_status", "pending")
    if status == "edited":
        return "hitl_gate"  # loop back for re-review
    return "END"  # approved -> END (file written), rejected -> END (nothing written)


def _apply_edits(
    plan: QuestionPlan, edits: list[dict[str, Any]], existing: list[ApprovalEdit] | None = None
) -> tuple[QuestionPlan, list[ApprovalEdit]]:
    """Apply recruiter edits to the plan; return (new_plan, new_records)."""
    new_plan = plan.model_copy(deep=True)
    records: list[ApprovalEdit] = []
    now = datetime.now(timezone.utc).isoformat()

    for edit in edits:
        qid = edit.get("question_id")
        field = edit.get("field")
        new_value = str(edit.get("new_value", ""))
        target = next((q for q in new_plan.questions if q.id == qid), None)
        if target is None:
            print(f"[graph] edit ignored: unknown question id {qid}")
            continue
        old_value = str(getattr(target, field, ""))
        if hasattr(target, field):
            setattr(target, field, new_value)
        records.append(
            ApprovalEdit(
                question_id=qid,
                field=field,
                old_value=old_value,
                new_value=new_value,
                timestamp=now,
            )
        )
        print(f"[graph] edit: {qid}.{field} {old_value!r} -> {new_value!r}")

    new_plan.edits_made = [
        r.model_dump() if isinstance(r, ApprovalEdit) else r
        for r in (existing or [])
    ]
    return new_plan, records


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph(checkpointer: SqliteSaver) -> Callable[..., Any]:
    """Assemble and compile the FirstRound StateGraph."""
    builder = StateGraph(GraphState)

    builder.add_node("parse_jd", parse_jd)
    builder.add_node("parse_resume", parse_resume)
    builder.add_node("request_github_manually", request_github_manually)
    builder.add_node("github_agent", run_github_agent)
    builder.add_node("question_planner", run_question_planner)
    builder.add_node("hitl_gate", hitl_gate)

    builder.add_edge(START, "parse_jd")
    builder.add_edge("parse_jd", "parse_resume")
    builder.add_conditional_edges(
        "parse_resume",
        route_github,
        {"github_agent": "github_agent", "request_github_manually": "request_github_manually"},
    )
    builder.add_edge("request_github_manually", "github_agent")
    builder.add_edge("github_agent", "question_planner")
    builder.add_edge("question_planner", "hitl_gate")
    builder.add_conditional_edges(
        "hitl_gate",
        route_approval,
        {"hitl_gate": "hitl_gate", "END": END},
    )

    return builder.compile(checkpointer=checkpointer)


@contextmanager
def get_checkpointer() -> Iterator[SqliteSaver]:
    """Context-manager checkpointer on checkpoint.db (call inside `with`).

    The msgpack serde is constructed with an explicit allowlist of the
    src.schemas Pydantic models that live in GraphState, so they round-trip
    through the checkpoint without warnings today AND are explicitly allowed
    when LangGraph's strict-msgpack mode is enabled downstream.

    Why this shape (not with_msgpack_allowlist): the default allowlist is
    the permissive sentinel ``True``, and ``with_msgpack_allowlist`` returns
    ``self`` unchanged in that case, so nothing would actually be registered.
    Passing ``allowed_msgpack_modules=[...]`` in the constructor registers the
    classes immediately (jsonplus normalizes each class -> ("src.schemas",
    "JDInfo") etc.).
    """
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            JDInfo,
            ResumeRole,
            ResumeInfo,
            CommitInfo,
            FileEvidence,
            RepoInfo,
            GithubInfo,
            Question,
            QuestionPlan,
            Turn,
            Transcript,
            CompetencyScore,
            Scorecard,
            ApprovalEdit,
        ]
    )
    with closing(
        sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    ) as conn:
        yield SqliteSaver(conn, serde=serde)


if __name__ == "__main__":
    with get_checkpointer() as cp:
        app = build_graph(cp)
    print("graph compiled OK; run via run_graph.py --help")
    sys.exit(0)
