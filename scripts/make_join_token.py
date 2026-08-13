"""Mint a LiveKit JWT for a candidate test user and print a join URL.

The token grants join access to a room AND configures that room to dispatch
jobs to the worker's named agent (``firstround-interviewer``), so opening the
printed URL triggers ``python -m src.realtime.agent dev`` to join and talk.

Usage:
    python scripts/make_join_token.py [room_name] [identity]

The token must be opened from a local HTTP server (browsers block mic access
from file:// URLs). From the repo root, run:

    python -m http.server 8000 --directory src/realtime

then open the printed URL. It looks like:

    http://localhost:8000/room.html?url=wss://<host>.livekit.cloud&token=<jwt>
"""

from __future__ import annotations

import sys
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(Path(ROOT, ".env"))

from livekit.api import (  # noqa: E402
    AccessToken,
    RoomAgentDispatch,
    RoomConfiguration,
    VideoGrants,
)

AGENT_NAME = "firstround-interviewer"


def main() -> None:
    room_name = sys.argv[1] if len(sys.argv) > 1 else "firstround-test"
    identity = sys.argv[2] if len(sys.argv) > 2 else "candidate-test"

    ws_url = os_livekit_url().rstrip("/")

    token = (
        AccessToken()
        .with_identity(identity)
        .with_name("Candidate")
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(
            RoomConfiguration(
                name=room_name,
                agents=[RoomAgentDispatch(agent_name=AGENT_NAME)],
            )
        )
    ).to_jwt()

    print(f"http://localhost:8000/room.html?url={ws_url}&token={token}")
    print("serve first:  python -m http.server 8000 --directory src/realtime")


def os_livekit_url() -> str:
    import os

    return os.environ.get("LIVEKIT_URL", "") or "wss://localhost:7880"


if __name__ == "__main__":
    main()