"""Phase 8: PDF report generator (reportlab).

Reads output/scorecard.json + output/transcript.json and writes output/report.pdf
with the candidate name, per-competency scores + evidence quotes, recommendation,
and interview metadata.

Usage:
    python -m src.agents.report_generator
        [scorecard_path] [transcript_path] [output_path]

Defaults (repo-relative):
    output/scorecard.json  output/transcript.json  output/report.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.schemas import Scorecard, Transcript

ROOT = Path(__file__).resolve().parents[2]


def _safe(text: str) -> str:
    """Strip characters a Type1 PDF font can't render (e.g. Devanagari STT text).

    reportlab's built-in fonts are WinAnsi/latin-1 only; passing unencoded
    codepoints raises UnicodeEncodeError. Replace them rather than crash.
    """
    if not text:
        return ""
    return "".join(ch if ord(ch) < 256 else "?" for ch in text).replace("\n", " ")


def _format_duration(seconds: int) -> str:
    if seconds <= 0:
        return "0s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def run(
    scorecard_path: str | Path,
    transcript_path: str | Path,
    output_path: str | Path,
) -> Path:
    scorecard = Scorecard.model_validate_json(
        Path(scorecard_path).read_text(encoding="utf-8")
    )
    transcript = Transcript.model_validate_json(
        Path(transcript_path).read_text(encoding="utf-8")
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], spaceAfter=6)
    body = styles["BodyText"]

    elements = []
    elements.append(Paragraph(_safe("FirstRound — Interview Report"), title_style))
    elements.append(
        Paragraph(
            _safe(
                f"<b>{scorecard.candidate_name}</b> — {scorecard.role}<br/>"
                f"Interview date: {scorecard.interview_date} · "
                f"Duration: {_format_duration(scorecard.duration_seconds)} · "
                f"Transcript turns: {len(transcript.turns)}"
            ),
            body,
        )
    )
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(
        Paragraph(
            _safe(
                f"<b>Overall: {scorecard.overall_score}/5</b> — "
                f"<b>{scorecard.recommendation}</b>"
            ),
            body,
        )
    )
    if scorecard.recommendation_reasoning:
        elements.append(Paragraph(_safe(scorecard.recommendation_reasoning), body))
    elements.append(Spacer(1, 0.4 * cm))

    header = ["Competency", "Score", "Confidence", "Evidence quote", "Reasoning"]
    rows = [header]
    for c in scorecard.competencies:
        rows.append(
            [
                _safe(c.name),
                str(c.score),
                f"{c.confidence:.2f}",
                _safe(c.evidence_quote),
                _safe(c.reasoning),
            ]
        )
    table = Table(rows, colWidths=[3.0 * cm, 1.2 * cm, 2.0 * cm, 5.0 * cm, 5.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.4 * cm))

    def _bullet(items: list[str], label: str) -> None:
        if not items:
            return
        elements.append(Paragraph(f"<b>{label}</b>", body))
        for item in items:
            elements.append(Paragraph(f"• {_safe(item)}", body))
        elements.append(Spacer(1, 0.2 * cm))

    _bullet(scorecard.strengths, "Strengths")
    _bullet(scorecard.concerns, "Concerns")
    _bullet(scorecard.guardrail_flags, "Guardrail flags")

    elements.append(
        Paragraph(
            f"GitHub-grounded questions asked: {scorecard.github_grounded_questions_asked}",
            body,
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="FirstRound Interview Report",
        author="FirstRound",
    )
    doc.build(elements)
    print(f"[report] wrote {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = ROOT
    scorecard_path = Path(argv[0]) if len(argv) >= 1 else root / "output" / "scorecard.json"
    transcript_path = Path(argv[1]) if len(argv) >= 2 else root / "output" / "transcript.json"
    output_path = Path(argv[2]) if len(argv) >= 3 else root / "output" / "report.pdf"
    run(scorecard_path, transcript_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
