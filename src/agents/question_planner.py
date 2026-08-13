"""Phase 1 agent: turn JD + resume + GitHub evidence into 12 interview questions.

Combines output/prep/{jd,resume,github}.json via Groq structured output into
output/prep/question_plan.json. Enforces: exactly 12 questions, at least 3 with
source="github" whose source_reference points at a real repo/file/commit listed
in github.json (retried with feedback until satisfied).

Question schema (PRD §6):
  id, text, competency, source (jd|resume|github|scenario),
  source_reference, difficulty (easy|medium|hard), follow_up_triggers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.agents.groq_client import chat_structured
from src.schemas import GithubInfo, JDInfo, QuestionPlan, ResumeInfo

SYSTEM_PROMPT = """You are an expert technical interviewer building a question
plan for a live AI video interview of a real candidate. You will be given:
1) the parsed job description, 2) the parsed resume, 3) real GitHub evidence
for the candidate's actual repositories (with file excerpts).

Produce EXACTLY 12 questions. Rules:
- Every question must be answerable by the candidate and grounded in what is
  actually given — never invent facts, repos, files or claims.
- "source" is one of: jd (from the job description), resume (from a resume
  claim), github (grounded in the candidate's real code), scenario (a
  hypothetical work scenario, often competency-focused).
- AT LEAST 3 questions MUST have source="github". Their source_reference must
  name a real repository, file path, or commit message exactly as it appears in
  the GitHub evidence (e.g. "repo:myapp", "file:myapp/src/main.py",
  "commit:fixed rate limiting"). Generic questions like "tell me about your
  GitHub" DO NOT count — reference a specific repo/file/commit.
- Mix difficulty levels (easy/medium/hard) across the set.
- follow_up_triggers: 2-4 short phrases describing candidate answers that should
  trigger a deeper probe (e.g. "mentions caching", "unclear on RAG chunking").
- id values: q1..q12."""

VALID_SOURCES = {"jd", "resume", "github", "scenario"}


def _build_user_prompt(jd: JDInfo, resume: ResumeInfo, github: GithubInfo) -> str:
    gh_lines = []
    for repo in github.repos:
        gh_lines.append(f"### repo: {repo.full_name}")
        gh_lines.append(f"  language: {repo.language} | stars: {repo.stars}")
        if repo.description:
            gh_lines.append(f"  description: {repo.description}")
        if repo.recent_commits:
            gh_lines.append("  recent commits:")
            for c in repo.recent_commits[:5]:
                gh_lines.append(f"    - {c.date[:10]} {c.message}")
        if repo.readme_excerpt:
            gh_lines.append(
                f"  readme excerpt: {repo.readme_excerpt[:400].strip()}"
            )
        for f in repo.files_read:
            gh_lines.append(
                f"  file: {f.path} (excerpt below, {len(f.excerpt)} chars)"
            )
            gh_lines.append(f"  file excerpt: {f.excerpt[:600].strip()}")
        if repo.top_repository:
            gh_lines.append("  [TOP-RELEVANT REPO]")
    gh_text = "\n".join(gh_lines)

    return f"""=== JOB DESCRIPTION (parsed) ===
Role: {jd.role}
Seniority: {jd.seniority}
Company: {jd.company} ({jd.location})
Must-have skills: {", ".join(jd.must_have_skills)}
Competencies to assess: {", ".join(jd.competencies)}

=== RESUME (parsed) ===
Name: {resume.name}
Roles:
{json.dumps([r.model_dump() for r in resume.roles], indent=2)}
Skills: {", ".join(resume.skills)}
Claims:
{json.dumps(resume.claims, indent=2)}

=== GITHUB EVIDENCE (real) ===
{gh_text}

Produce the question plan now."""


def _github_refs(github: GithubInfo) -> set[str]:
    refs = set()
    for repo in github.repos:
        refs.add(repo.full_name.lower())
        refs.add(repo.name.lower())
        for c in repo.recent_commits:
            refs.add(c.message.lower())
        for f in repo.files_read:
            refs.add(f.path.lower())
            refs.add(f"file:{f.path.lower()}")
            refs.add(f"repo:{repo.name.lower()}/" + f.path.lower())
    return refs


def _github_grounded(q, refs: set[str]) -> bool:
    """True if the question cites a real repo/file/commit from the evidence."""
    ref = (q.source_reference or "").lower()
    if not ref:
        return False
    # Must reference something concrete, not just the word "github".
    if ref in {"github", "github repo", "github profile", "my github"}:
        return False
    if ref in refs:
        return True
    # Allow "repo:name" / "file:path" / "commit:..." style references.
    for key in refs:
        if len(key) >= 3 and (key in ref or ref in key):
            return True
    return False


def _validate(plan: QuestionPlan, github: GithubInfo) -> list[str]:
    problems: list[str] = []
    if len(plan.questions) != 12:
        problems.append(f"expected exactly 12 questions, got {len(plan.questions)}")
    gh_qs = [q for q in plan.questions if q.source == "github"]
    if len(gh_qs) < 3:
        problems.append(f"expected >=3 github-sourced questions, got {len(gh_qs)}")
    else:
        refs = _github_refs(github)
        grounded = [q for q in gh_qs if _github_grounded(q, refs)]
        if len(grounded) < 3:
            problems.append(
                f"only {len(grounded)} of {len(gh_qs)} github questions reference "
                "a real repo/file/commit"
            )
    bad_ids = [q.id for q in plan.questions if not q.id]
    if bad_ids:
        problems.append("some questions are missing an id")
    return problems


def generate_plan(jd: JDInfo, resume: ResumeInfo, github: GithubInfo) -> QuestionPlan:
    user_prompt = _build_user_prompt(jd, resume, github)
    refs = _github_refs(github)

    plan = chat_structured(QuestionPlan, SYSTEM_PROMPT, user_prompt)

    for attempt in range(1, 4):
        problems = _validate(plan, github)
        if not problems:
            break

        feedback = "\n".join(f"- {p}" for p in problems)
        print(
            f"[question_planner] validation failed (attempt {attempt}): "
            f"{problems}; regenerating with feedback..."
        )
        corrective = user_prompt + f"""

The previous question plan was rejected for these reasons:
{feedback}

Real GitHub references available include: {", ".join(sorted(refs)[:40])}
Fix every problem and return the full corrected plan."""

        plan = chat_structured(QuestionPlan, SYSTEM_PROMPT, corrective)

    return plan


def run(
    prep_dir: str | Path,
    output_path: str | Path,
) -> QuestionPlan:
    prep_dir = Path(prep_dir)
    output_path = Path(output_path)

    jd = JDInfo.model_validate_json(
        (prep_dir / "jd.json").read_text(encoding="utf-8")
    )
    resume = ResumeInfo.model_validate_json(
        (prep_dir / "resume.json").read_text(encoding="utf-8")
    )
    github = GithubInfo.model_validate_json(
        (prep_dir / "github.json").read_text(encoding="utf-8")
    )

    print("[question_planner] generating 12-question plan...")
    plan = generate_plan(jd, resume, github)

    problems = _validate(plan, github)
    if problems:
        raise RuntimeError(
            "question plan did not pass validation after retries: "
            + "; ".join(problems)
        )

    plan.approved_by_human = False
    plan.edits_made = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan.model_dump(), indent=2), encoding="utf-8"
    )
    n_gh = sum(1 for q in plan.questions if q.source == "github")
    print(
        f"[question_planner] wrote {output_path} "
        f"({len(plan.questions)} questions, {n_gh} github-grounded)"
    )
    return plan


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(__file__).resolve().parents[2]
    prep_dir = Path(argv[0]) if len(argv) >= 1 else root / "output" / "prep"
    out_path = Path(argv[1]) if len(argv) >= 2 else prep_dir / "question_plan.json"
    run(prep_dir, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())