"""Phase 1 CLI: run the full offline prep pipeline.

Usage:
    python run_prep.py inputs/jd.txt inputs/resume.pdf

Outputs to output/prep/{jd.json,resume.json,github.json,question_plan.json}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agents import github_agent, jd_parser, question_planner, resume_parser  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FirstRound prep pipeline: JD + resume + GitHub -> question plan."
    )
    parser.add_argument("jd", nargs="?", default=str(ROOT / "inputs" / "jd.txt"))
    parser.add_argument("resume", nargs="?", default=str(ROOT / "inputs" / "resume.pdf"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "prep"),
        help="directory for prep JSON outputs",
    )
    args = parser.parse_args(argv)

    jd_path = Path(args.jd)
    resume_path = Path(args.resume)
    prep_dir = Path(args.output)

    if not jd_path.exists():
        print(f"error: JD file not found: {jd_path}")
        return 1
    if not resume_path.exists():
        print(f"error: resume file not found: {resume_path}")
        return 1

    print("=" * 60)
    print("FirstRound prep pipeline")
    print("=" * 60)

    jd_parser.run(jd_path, prep_dir / "jd.json")
    print()

    resume_parser.run(resume_path, prep_dir / "resume.json")
    print()

    github_agent.run(prep_dir / "resume.json", prep_dir / "github.json",
                     jd_path=prep_dir / "jd.json")
    print()

    question_planner.run(prep_dir, prep_dir / "question_plan.json")
    print()

    print("=" * 60)
    print("Prep complete. Review the plan and run the HITL gate in Phase 2.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
