"""Phase 0 key verification.

Loads .env and makes ONE lightweight authenticated call to each of:
  - Gemini API        (list models)
  - Groq API          (list models)
  - GitHub REST API   (GET /user using the PAT)
  - LiveKit server API (list rooms)

Prints "OK" or "FAIL: <reason>" per service, never crashes on a single
failure, and summarizes all four at the end.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> None:
    # Prefer .env next to this script; fall back to CWD.
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=False)


def check_gemini() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        return "FAIL: GOOGLE_API_KEY is not set"
    from google import genai

    client = genai.Client(api_key=key)
    # Force the network call by consuming the (lazy) model list.
    for model in client.models.list():
        break
    else:
        return "FAIL: model list is empty"
    return "OK"


def check_groq() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return "FAIL: GROQ_API_KEY is not set"
    from groq import Groq

    client = Groq(api_key=key)
    models = client.models.list()
    if not models or not models.data:
        return "FAIL: model list is empty"
    return "OK"


def check_github() -> str:
    pat = os.environ.get("GITHUB_PAT", "").strip()
    if not pat:
        return "FAIL: GITHUB_PAT is not set"
    import github
    from github import Auth

    user = github.Github(auth=Auth.Token(pat)).get_user()
    if not user or not user.login:
        return "FAIL: /user returned no login"
    return "OK"


def check_livekit() -> str:
    api_key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    url = os.environ.get("LIVEKIT_URL", "").strip()
    if not api_key or not api_secret or not url:
        return "FAIL: LIVEKIT_API_KEY, LIVEKIT_API_SECRET or LIVEKIT_URL not set"
    from livekit import api

    async def probe() -> None:
        service = api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret)
        try:
            result = await service.room.list_rooms(api.ListRoomsRequest())
            rooms = getattr(result, "rooms", None)
        finally:
            await service.aclose()
        if rooms is None:
            raise RuntimeError("unexpected response from list_rooms")

    asyncio.run(probe())
    return "OK"


def main() -> int:
    load_environment()

    checks = [
        ("Gemini", check_gemini),
        ("Groq", check_groq),
        ("GitHub", check_github),
        ("LiveKit", check_livekit),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 - we report any failure and continue
            results[name] = f"FAIL: {type(exc).__name__}: {exc}"

    print()
    for name, status in results.items():
        print(f"{name:8s} {status}")
    print()

    failed = [name for name, status in results.items() if status != "OK"]
    if failed:
        print(f"FAILED: {len(failed)} of {len(results)} checks failed ({', '.join(failed)})")
        return 1
    print(f"ALL OK: {len(results)} of {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
