"""Phase 1 agent: extract the resume into structured JSON.

Reads inputs/resume.pdf, extracts text with pdfplumber, then uses Groq
structured output to pull name, roles, claims, skills and any GitHub /
LinkedIn URLs. Writes output/prep/resume.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pdfplumber

from src.agents.groq_client import chat_structured
from src.schemas import ResumeInfo

SYSTEM_PROMPT = """You extract structured data from a resume's raw text.
Return only valid JSON matching the given schema. Be faithful to the text:
- name: full name.
- roles: work/project roles, with title, company, years and a short
  description. List the ones that matter for a technical interview.
- claims: 5-10 concrete, verifiable assertions the candidate makes about
  their experience or projects (things an interviewer could probe).
- skills: the technical skills listed.
- github_url / linkedin_url: only if actually present in the text, as full
  URLs. Use null when absent. Double-check that a URL looks like a real
  profile (containing github.com or linkedin.com)."""

GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/[\w.-]+", re.IGNORECASE)
LINKEDIN_RE = re.compile(r"https?://(?:www\.)?linkedin\.com/in/[\w.-]+", re.IGNORECASE)


def extract_pdf_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(f"No text extracted from {pdf_path}")
    return text


def _fallback_urls(text: str) -> tuple[str | None, str | None]:
    """If the model returns no URLs, regex the raw text as a safety net."""
    gh = GITHUB_RE.search(text)
    li = LINKEDIN_RE.search(text)
    return (gh.group(0) if gh else None, li.group(0) if li else None)


def parse_resume(resume_text: str) -> ResumeInfo:
    info = chat_structured(
        ResumeInfo,
        SYSTEM_PROMPT,
        f"Resume text:\n\n{resume_text}",
    )
    gh, li = _fallback_urls(resume_text)
    if not info.github_url and gh:
        info.github_url = gh
    if not info.linkedin_url and li:
        info.linkedin_url = li
    return info


def run(input_path: str | Path, output_path: str | Path) -> ResumeInfo:
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"[resume_parser] extracting text from {input_path}...")
    text = extract_pdf_text(input_path)

    print("[resume_parser] parsing resume with Groq...")
    result = parse_resume(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(), indent=2), encoding="utf-8"
    )
    print(f"[resume_parser] wrote {output_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(__file__).resolve().parents[2]
    if len(argv) >= 2:
        pdf_path = Path(argv[0])
        out_path = Path(argv[1])
    else:
        pdf_path = root / "inputs" / "resume.pdf"
        out_path = root / "output" / "prep" / "resume.json"
    run(pdf_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())