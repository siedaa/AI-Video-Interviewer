"""Phase 7 evals: run each synthetic persona transcript through the scorer.

Reads evals/personas/*.json, scores each against the real question plan via
src.agents.scorer.run (Groq LLM + evidence-check guardrail), writes the per-
persona scorecards to evals/scorecards/, and writes evals/results.md with the
ranking table, the expected-vs-actual ranking check, and an honest analysis of
any misranking or surprising scorer behavior.

Expected ranking (per PRD §13): Strong > Nervous ≈ Average > Bluffer > Weak.
The two hard cases: Bluffer must land below Average, Nervous must land near
Strong. We do NOT hand-tune the scorer prompt to force this — we report what
actually happens.

Usage:
    python evals/run_evals.py             # score every persona (Groq LLM calls)
    python evals/run_evals.py --reuse     # reuse existing scorecards in evals/scorecards/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.scorer import run as score_run  # noqa: E402

PERSONAS_DIR = ROOT / "evals" / "personas"
PLAN_PATH = ROOT / "output" / "prep" / "question_plan.json"
SCORECARDS_DIR = ROOT / "evals" / "scorecards"
RESULTS_PATH = ROOT / "evals" / "results.md"

EXPECTED_ORDER = ["strong", "nervous", "average", "bluffer", "weak"]

# Written after inspecting the first real run — reflects what the scorer
# actually produced, including any surprises (do not hand-tune to force a pass).
HONEST_ANALYSIS = (
    "The expected ranking was Strong > Nervous ≈ Average > Bluffer > Weak, with the "
    "two hard cases being Bluffer below Average and Nervous near Strong. The scorer "
    "put Strong (#1, 4.6), Nervous (#3, 3.4), Average (#4, 2.6), and Weak (last, 1.8) "
    "in sensible slots, but **Bluffer misranked at #2 (3.6) — above both Average and "
    "Nervous** in both eval runs. This is the exact 'confident and wrong' failure mode "
    "the evals were built to expose, and we are reporting it rather than tuning it away.\n"
    "\n"
    "Why Bluffer passed the scorer: the LLM rewarded fluent, well-structured but "
    "substantively empty answers. Despite collapsing on every specific follow-up "
    "('those numbers were tuned by the infra team', 'under NDA', 'I've debugged "
    "dozens of these' with no actual method), Bluffer still earned mid-to-high "
    "competency scores (Debugging=4, Communication=4). The evidence-check guardrail "
    "does not catch this: its quotes are real transcript substrings, so it validates "
    "provenance, not correctness — a confident wrong answer is still quotable. "
    "Nervous also sits two positions below Strong (rank #3 vs #1), so the 'near "
    "Strong' criterion flags; its 3.4 is defensibly 'Nervous ≈ Average, above "
    "Average', and the gap is partly an artifact of Bluffer's inflation crowding the "
    "top two slots.\n"
    "\n"
    "Run-to-run variance is real and should be read alongside these numbers: with "
    "identical transcripts, run 1 produced strong 4.4/hire, bluffer 4.0/hire, average "
    "3.0/borderline, weak 2.0/no_hire, while run 2 produced 4.6/borderline, 3.6/"
    "borderline, 2.6/no_hire, 1.8/no_hire. Competency scores and recommendation come "
    "from the LLM (overall_score is a deterministic recomputation of them), so "
    "single-run numbers are noisy — but the structural ordering (strong > bluffer > "
    "nervous > average > weak) was identical in both runs, so the Bluffer misranking "
    "is robust, not a fluke. Oddly, strong rose to 4.6 yet its recommendation dropped "
    "to borderline, a symptom of recommendation being LLM-decided rather than derived "
    "from overall_score.\n"
    "\n"
    "Suggested (Phase 8 follow-up, deliberately NOT applied here to avoid hand-tuning): "
    "reward answer depth/elaboration and specificity-to-follow-ups in the scoring "
    "rubric, derive recommendation from overall_score thresholds instead of leaving it "
    "to the LLM, and consider a heuristic guardrail that flags competencies whose "
    "evidence_quote is very short or non-technical relative to the transcript's other "
    "answers. Weak and Nervous behaved as intended."
)


def _load_scorecard(path: Path) -> object:
    from src.schemas import Scorecard

    return Scorecard.model_validate_json(path.read_text(encoding="utf-8"))


def run_all(reuse: bool = False) -> list[tuple[str, object]]:
    SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, object]] = []
    for persona_path in sorted(PERSONAS_DIR.glob("*.json")):
        name = persona_path.stem
        out_path = SCORECARDS_DIR / f"{name}_scorecard.json"
        if reuse and out_path.exists():
            print(f"[evals] reusing existing scorecard for '{name}'")
            results.append((name, _load_scorecard(out_path)))
            continue
        print(f"[evals] scoring persona '{name}'...")
        scorecard = score_run(persona_path, PLAN_PATH, out_path)
        results.append((name, scorecard))
    results.sort(key=lambda item: item[1].overall_score, reverse=True)
    return results


def _ranking_check(results: list[tuple[str, object]]) -> str:
    order = [name for name, _ in results]
    actual_index = {name: i for i, name in enumerate(order)}

    strong_ok = order[0] == "strong"
    bluffer_below_average = actual_index["bluffer"] > actual_index["average"]
    nervous_near_strong = abs(actual_index["nervous"] - actual_index["strong"]) <= 1
    weak_last = order[-1] == "weak"

    lines = [
        "## Ranking check",
        "",
        f"Actual order: {', '.join(order)}",
        f"Expected order: {' > '.join(EXPECTED_ORDER)}",
        "",
        "- Strong lands #1: **" + ("PASS" if strong_ok else "FAIL") + "**",
        "- Bluffer lands below Average (hard case): **"
        + ("PASS" if bluffer_below_average else "FAIL") + "**",
        "- Nervous lands near Strong (hard case, within one position): **"
        + ("PASS" if nervous_near_strong else "FAIL") + "**",
        "- Weak lands last: **" + ("PASS" if weak_last else "FAIL") + "**",
        "",
        "All hard cases PASS: " + str(
            strong_ok and bluffer_below_average and nervous_near_strong and weak_last
        ),
    ]
    return "\n".join(lines)


def write_results(results: list[tuple[str, object]]) -> None:
    rows = []
    for rank, (name, sc) in enumerate(results, 1):
        rows.append(
            f"| {rank} | {name} | {sc.overall_score} | {sc.recommendation} |"
        )
    table = "\n".join(
        [
            "## Ranking table",
            "",
            "| Rank | Persona | overall_score | recommendation |",
            "|---|---|---|---|",
            *rows,
        ]
    )

    details = []
    for name, sc in results:
        comps = ", ".join(
            f"{c.name}={c.score}" for c in sc.competencies
        ) or "(no competencies)"
        details.append(f"- **{name}** — overall {sc.overall_score}, "
                       f"{sc.recommendation}; competencies: {comps}; "
                       f"flags: {len(sc.guardrail_flags)}")
    detail_section = "\n".join(["## Per-persona detail", "", *details])

    markdown = "\n\n".join(
        [
            "# FirstRound Evals — results",
            "",
            "Synthetic persona transcripts scored by src/agents/scorer.py against "
            "output/prep/question_plan.json. Run: `python evals/run_evals.py`.",
            table,
            _ranking_check(results),
            "## Honest analysis",
            "",
            HONEST_ANALYSIS,
            detail_section,
        ]
    )
    RESULTS_PATH.write_text(markdown + "\n", encoding="utf-8")
    print(f"[evals] wrote {RESULTS_PATH}")


def main() -> int:
    reuse = "--reuse" in sys.argv
    results = run_all(reuse=reuse)
    for rank, (name, sc) in enumerate(results, 1):
        print(
            f"[evals]   #{rank} {name}: overall={sc.overall_score} "
            f"recommendation={sc.recommendation}"
        )
    write_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
