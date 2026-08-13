"""Minimal Gemini Live connectivity check — no LiveKit, no audio.

Loads GOOGLE_API_KEY from .env, opens a Gemini Live websocket session via
google.genai with the pinned model string, sends one text turn, and prints
success or the error. Fail fast; exit code 0 on success, 1 on failure.

Usage:
    python scripts/test_gemini_live.py [model]
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(Path(ROOT, ".env"))

os.environ.pop("GEMINI_API_KEY", None)

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

PINNED_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"


async def check(model: str) -> None:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        print("FAIL: GOOGLE_API_KEY missing from .env")
        sys.exit(1)

    client = genai.Client(api_key=key)

    print(f"connecting to Gemini Live with model={model!r} ...")
    async with client.aio.live.connect(model=model) as session:
        await session.send_client_content(
            turns=types.Content(parts=[types.Part(text="ping")]), turn_complete=True
        )
        print("OK: Gemini Live websocket session open; ping sent")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else PINNED_MODEL
    try:
        asyncio.run(check(model))
    except Exception as exc:  # noqa: BLE001 - report any error verbatim
        print(f"FAIL: {type(exc).__name__}: {exc}")
        sys.exit(1)