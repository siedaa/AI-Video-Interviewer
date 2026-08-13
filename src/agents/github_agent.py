"""Phase 1 agent: pull real GitHub evidence for the candidate.

Reads output/prep/resume.json to get github_url, then uses PyGithub (with
GITHUB_PAT) to list the candidate's repos, languages, recent commit messages
and dates, and reads the README plus at least one source file for the top 3
most relevant repos (most recently active or matching JD skills).

Writes output/prep/github.json.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from github import Auth, Github
from github.ContentFile import ContentFile

from src.schemas import (
    CommitInfo,
    FileEvidence,
    GithubInfo,
    JDInfo,
    RepoInfo,
    ResumeInfo,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

RELEVANT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb",
    ".cpp", ".c", ".h", ".cs", ".php", ".swift", ".kt", ".sh", ".sql",
    ".ipynb", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
}
PREFERRED_ENTRYPOINTS = {
    "main.py", "app.py", "index.py", "run.py", "cli.py", "app.js",
    "index.js", "index.ts", "main.js", "main.ts", "server.py",
    "manage.py", "requirements.txt", "package.json", "pyproject.toml",
    "setup.py", "README.md", "README.rst",
}
MAX_REPOS = 15
MAX_FILES_PER_REPO = 3
EXCERPT_CHARS = 2000


def _is_repo_name_match(name: str, skills: List[str]) -> bool:
    """True if any skill keyword appears inside the repo name."""
    n = name.lower()
    return any(s.lower() in n for s in skills if s.strip())


def _language_matches(language: str, skills: List[str]) -> bool:
    lang = (language or "").lower()
    for s in skills:
        s = s.lower()
        if not s:
            continue
        # exact or substring match (e.g. "python" vs "python 3.10 / django")
        if lang in s or s in lang:
            return True
    return False


def _parse_iso(iso: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _relevance(repo: RepoInfo, jd: JDInfo | None) -> float:
    """Score 0..100: recent activity + language match vs JD + stars."""
    score = 0.0

    skills = list((jd.must_have_skills if jd else []) or [])
    supports = ["github", "git", "api", "javascript", "python"]

    pushed = _parse_iso(repo.pushed_at)
    if pushed:
        days = max(0, (_now() - pushed).days)
        score += max(0.0, 60.0 - days * 0.5)

    if _language_matches(repo.language, skills):
        score += 30.0
    if any(_is_repo_name_match(k, skills) for k in [repo.name] + skills):
        score += 15.0
    if _is_repo_name_match(repo.name, supports) or (
        repo.name and "project" in repo.name.lower()
    ):
        score += 5.0

    score += min(repo.stars, 20) * 0.25
    return score


def _first_text_file(contents) -> ContentFile | None:
    """Pick one readable source file from a root listing (prefer entrypoints)."""
    files = [c for c in contents if c.type == "file" and c.size > 0]
    if not files:
        return None
    files.sort(key=lambda f: f.size, reverse=True)
    for f in files:
        if Path(f.name).name in PREFERRED_ENTRYPOINTS:
            return f
    for f in files:
        if Path(f.name).suffix in RELEVANT_EXTENSIONS or Path(
            f.name
        ).name.lower().startswith(("readme", "requirements")):
            return f
    return files[0]


def _safe_name(repo) -> str:
    return (repo.name or repo.full_name or "unknown").replace(
        " ", "-"
    )


def gather_github(
    github_url: str,
    resume: ResumeInfo | None,
    jd: JDInfo | None,
) -> GithubInfo:
    pat = os.environ.get("GITHUB_PAT", "").strip()
    if not pat:
        raise RuntimeError(
            "GITHUB_PAT is not set. Copy .env.example to .env and fill it in."
        )

    gh = Github(auth=Auth.Token(pat))
    user = gh.get_user(_username_from_url(github_url))

    info = GithubInfo(username=user.login, profile_url=github_url)

    raw_repos = list(user.get_repos(sort="pushed", direction="desc"))[:MAX_REPOS]

    # First pass: gather per-repo facts (lightweight: README deferred).
    repos: List[RepoInfo] = []
    for repo in raw_repos:
        languages = list((repo.get_languages() or {}).keys())
        commits = []
        try:
            page = repo.get_commits()[:8]
            for c in page:
                msg = c.commit.message.strip().splitlines()[0] if c.commit.message else ""
                commits.append(
                    CommitInfo(
                        message=msg,
                        date=c.commit.author.date.isoformat()
                        if c.commit.author and c.commit.author.date
                        else "",
                    )
                )
        except Exception:
            commits = []

        repos.append(
            RepoInfo(
                name=_safe_name(repo),
                full_name=repo.full_name,
                language=repo.language,
                languages=languages[:8],
                description=(repo.description or "")[:300],
                updated_at=repo.updated_at.isoformat(),
                pushed_at=repo.pushed_at.isoformat(),
                stars=repo.stargazers_count or 0,
                recent_commits=commits,
            )
        )

    # Relevance ranking.
    ranked = sorted(
        repos, key=lambda r: _relevance(r, jd), reverse=True
    )
    top3 = ranked[:3]
    for r in top3:
        r.top_repository = True

    # Read evidence for top repos: README + at least one real source file.
    for repo in top3:
        gh_repo = gh.get_repo(repo.full_name)
        try:
            readme = gh_repo.get_readme()
            repo.readme_excerpt = readme.decoded_content.decode(
                "utf-8", errors="replace"
            )[:EXCERPT_CHARS]
        except Exception:
            repo.readme_excerpt = None

        try:
            contents = gh_repo.get_contents("")
        except Exception:
            contents = []
        if isinstance(contents, ContentFile):
            contents = [contents]

        chosen: List[ContentFile] = []
        entry = _first_text_file(contents)
        if entry:
            chosen.append(entry)
        for c in contents:
            if len(chosen) >= MAX_FILES_PER_REPO:
                break
            if c.type != "file" or c == entry or c.size == 0:
                continue
            if Path(c.name).suffix in RELEVANT_EXTENSIONS:
                chosen.append(c)

        for c in chosen:
            try:
                raw = c.decoded_content.decode("utf-8", errors="replace")
            except Exception:
                continue
            repo.files_read.append(
                FileEvidence(path=c.path, excerpt=raw[:EXCERPT_CHARS])
            )

    info.repos = repos
    return info


def _username_from_url(github_url: str) -> str:
    url = github_url.rstrip("/")
    parts = [p for p in url.split("/") if p]
    if not parts or "github.com" not in url:
        raise ValueError(f"Not a GitHub URL: {github_url}")

    # Take the segment right after github.com.
    try:
        idx = next(i for i, p in enumerate(parts) if "github.com" in p)
    except StopIteration:
        raise ValueError(f"Not a GitHub URL: {github_url}")
    if idx + 1 >= len(parts):
        return ""
    return parts[idx + 1]


def run(
    resume_path: str | Path,
    output_path: str | Path,
    *,
    jd_path: str | Path | None = None,
) -> GithubInfo:
    resume_path = Path(resume_path)
    output_path = Path(output_path)

    resume = ResumeInfo.model_validate_json(
        resume_path.read_text(encoding="utf-8")
    )
    if not resume.github_url:
        raise ValueError(
            "resume.json has no github_url — add the candidate's GitHub "
            "profile link to the resume and re-run resume_parser."
        )

    jd = None
    if jd_path and Path(jd_path).exists():
        try:
            jd = JDInfo.model_validate_json(
                Path(jd_path).read_text(encoding="utf-8")
            )
        except Exception as exc:
            print(f"[github_agent] warning: could not read JD: {exc}")

    username = _username_from_url(resume.github_url)
    if not username:
        raise ValueError(f"Cannot derive GitHub username from {resume.github_url}")

    print(f"[github_agent] pulling GitHub data for @{username}...")
    info = gather_github(resume.github_url, resume, jd)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(info.model_dump(), indent=2), encoding="utf-8"
    )
    top = [r.full_name for r in info.repos if r.top_repository]
    print(f"[github_agent] wrote {output_path} ({len(info.repos)} repos; top: {top})")
    return info


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(__file__).resolve().parents[2]
    if len(argv) >= 2:
        resume_path = Path(argv[0])
        out_path = Path(argv[1])
    else:
        resume_path = root / "output" / "prep" / "resume.json"
        out_path = root / "output" / "prep" / "github.json"
    run(resume_path, out_path,
        jd_path=root / "output" / "prep" / "jd.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())