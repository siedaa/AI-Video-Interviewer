"""Phase 2 CLI harness for the LangGraph prep pipeline + HITL gate.

Usage:
    python run_graph.py --jd inputs/jd.txt --resume inputs/resume.pdf
                         [--thread-id prep-<name>]

The graph runs to the first interrupt (either the "no GitHub URL" prompt or
the question-plan review gate), suspends, and this harness handles the
interaction. Everything is resumed on the SAME thread_id through the SQLite
checkpointer (checkpoint.db), so state survives restarts / crashes.

Interrupt handling:
  request_github       -> prompt for a GitHub username, resume with it
  question_plan_review -> pretty-print the 12 questions; prompt approve /
                          edit / reject
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langgraph.types import Command  # noqa: E402

from src.graph import build_graph, get_checkpointer  # noqa: E402


def _print_plan(plan: dict) -> None:
    print("\n" + "=" * 62)
    print(f"QUESTION PLAN  ({len(plan['questions'])} questions)  approval_status={plan.get('approval_status')}")
    print("=" * 62)
    for q in plan["questions"]:
        print(
            f"[{q['id']}] ({q['difficulty']:6s} / {q['source']:8s}) "
            f"[{q['competency']}]"
        )
        print(f"     {q['text']}")
        if q.get("source_reference"):
            print(f"     ref: {q['source_reference']}")
    print("=" * 62 + "\n")


def _ask_approve_edit_reject() -> dict:
    choice = input("Review the plan. [approve / edit / reject] > ").strip().lower()
    if choice == "approve":
        return {"action": "approve"}
    if choice == "edit":
        edits = []
        print("Enter edits, one per line. Each edit: <qid> <field> <new value>")
        print("  field is 'text' or 'difficulty'. Enter 'done' when finished.")
        while True:
            line = input("  edit > ").strip()
            if line.lower() == "done":
                break
            parts = line.split(" ", 2)
            if len(parts) != 3:
                print("  expected: qid field new-value")
                continue
            qid, field, new_value = parts
            if field not in ("text", "difficulty"):
                print("  field must be 'text' or 'difficulty'")
                continue
            edits.append({"question_id": qid, "field": field, "new_value": new_value})
        if not edits:
            print("No edits recorded; going back to review.")
            return _ask_approve_edit_reject()
        return {"action": "edit", "edits": edits}
    if choice == "reject":
        return {"action": "reject"}
    print("Type 'approve', 'edit', or 'reject'.")
    return _ask_approve_edit_reject()


def _handle_interrupt(payload: dict) -> dict | None:
    """Return the resume payload for this interrupt, or None if finished."""
    itype = payload.get("type")
    if itype == "request_github":
        username = input(
            "\nNo GitHub URL found in the resume.\n"
            "Enter the candidate's GitHub username > "
        ).strip()
        return {"github_url": f"https://github.com/{username}"}

    if itype == "question_plan_review":
        plan = payload.get("plan", {})
        _print_plan(plan)
        return _ask_approve_edit_reject()

    print(f"Unknown interrupt type: {itype}")
    return None


def _pending_resume_values(app, config: dict) -> list[dict]:
    """Read the thread's pending interrupts from its checkpoint.

    Returns the interrupt payloads for any nodes suspended at an interrupt,
    or [] if the thread has never started / already finished. This is what a
    resume must answer before the graph can continue — WITHOUT re-supplying
    the initial state (which would re-run everything from scratch).
    """
    snapshot = app.get_state(config)
    values: list[dict] = []
    for task in snapshot.tasks:
        for inter in task.interrupts:
            values.append(inter.value)
    return values


def _drive_interrupts(app, config: dict, pending: list[dict], initial: dict) -> dict:
    """Run the graph to completion, answering interrupts along the way.

    ``pending`` is the list of already-pending interrupt payloads read from
    the thread's checkpoint (empty on a fresh run). Any resume that triggers a
    further interrupt (e.g. edit -> re-review) is handled in the same loop.
    """
    result: dict = {}
    if pending:
        # Resuming an existing thread: answer the pending interrupts directly.
        # We do NOT re-invoke with the initial state, which would re-run the
        # whole pipeline from scratch.
        to_answer = list(pending)
        while to_answer:
            payload = to_answer.pop(0)
            resume_payload = _handle_interrupt(payload)
            if resume_payload is None:
                continue
            result = app.invoke(Command(resume=resume_payload), config)
            for inter in result.get("__interrupt__", []):
                to_answer.append(inter.value)
        return result

    # Fresh run.
    result = app.invoke(initial, config)
    while result.get("__interrupt__"):
        for inter in result["__interrupt__"]:
            resume_payload = _handle_interrupt(inter.value)
            if resume_payload is None:
                continue
            result = app.invoke(Command(resume=resume_payload), config)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FirstRound LangGraph prep + HITL")
    parser.add_argument("--jd", default=str(ROOT / "inputs" / "jd.txt"))
    parser.add_argument("--resume", default=str(ROOT / "inputs" / "resume.pdf"))
    parser.add_argument("--thread-id", default="prep-main")
    args = parser.parse_args(argv)

    jd_path = Path(args.jd)
    resume_path = Path(args.resume)
    thread_id = args.thread_id

    if not jd_path.exists():
        print(f"error: JD file not found: {jd_path}")
        return 1
    if not resume_path.exists():
        print(f"error: resume file not found: {resume_path}")
        return 1

    initial = {
        "inputs": {"jd": str(jd_path), "resume": str(resume_path)},
        "thread_id": thread_id,
        "approval_status": "pending",
        "edits_made": [],
    }
    config = {"configurable": {"thread_id": thread_id}}

    with get_checkpointer() as checkpointer:
        app = build_graph(checkpointer)

        pending = _pending_resume_values(app, config)
        if pending:
            print(f"[run_graph] resuming thread {thread_id} "
                  f"({len(pending)} pending interrupt(s))")
        else:
            print(f"[run_graph] starting thread {thread_id}")

        result = _drive_interrupts(app, config, pending, initial)

    # Post-loop: surface the outcome.
    status = result.get("approval_status", "unknown")
    print()
    if status == "approved":
        print("[run_graph] APPROVED — wrote output/prep/question_plan.json")
    elif status == "edited":
        print("[run_graph] plan was edited but not yet approved; run again to review.")
    elif status == "rejected":
        print("[run_graph] REJECTED — not approved, nothing proceeds.")
    else:
        print(f"[run_graph] finished with approval_status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
