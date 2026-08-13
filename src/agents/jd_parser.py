"""Phase 1 agent: parse the job description into structured JSON.

Reads inputs/jd.txt, extracts role/seniority/competencies/must-haves via Groq
structured output, writes output/prep/jd.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.agents.groq_client import chat_structured
from src.schemas import JDInfo

SYSTEM_PROMPT = """You extract structured data from a software job description.
Return only valid JSON matching the given schema. Be faithful to the text:
- role: the job title exactly as written.
- seniority: level such as "junior", "mid-level", "senior", or as stated.
- company/location: only if present.
- must_have_skills: the hard, explicitly-required technical skills (languages,
  frameworks, tools, APIs). Do not list soft skills here.
- competencies: the abstract qualities the JD says will be assessed (e.g.
  technical depth, problem solving, debugging, communication, shipping ability).
  Prefer terms the JD itself uses. 4-6 items."""


def parse_jd(jd_text: str) -> JDInfo:
    return chat_structured(
        JDInfo,
        SYSTEM_PROMPT,
        f"Job description:\n\n{jd_text}",
    )


def run(input_path: str | Path, output_path: str | Path) -> JDInfo:
    input_path = Path(input_path)
    output_path = Path(output_path)
    text = input_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"JD file is empty: {input_path}")

    print("[jd_parser] parsing job description...")
    result = parse_jd(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(), indent=2), encoding="utf-8"
    )
    print(f"[jd_parser] wrote {output_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(__file__).resolve().parents[2]
    if len(argv) >= 2:
        jd_path = Path(argv[0])
        out_path = Path(argv[1])
    else:
        jd_path = root / "inputs" / "jd.txt"
        out_path = root / "output" / "prep" / "jd.json"
    run(jd_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
