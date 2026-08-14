"""FirstRound avatar: publishes a synthetic video track to the LiveKit room.

The track is a ~15fps camera feed built from three aligned mouth-state images
(mouth_closed/mid/open.png, 512x512 RGBA). The active state is selected from
the RMS amplitude of the agent's *outgoing* audio: silence -> closed,
moderate -> mid, loud -> open.

The outgoing audio is tapped by an ``AudioOutput`` wrapper inserted between the
AgentSession and the RoomIO audio sink, so the transcript, barge-in and
interruption logic are untouched.

Usage (inside a LiveKit entrypoint)::

    avatar = Avatar(ASSETS_DIR)
    await avatar.publish(ctx.room)              # publish the video track
    avatar.start()                              # begin the render loop
    session.output.audio = AmplitudeTap(avatar, session.output.audio)
"""

from __future__ import annotations

import array
import asyncio
import logging
import math
import time
from pathlib import Path

from livekit import rtc

from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities  # noqa: E402

logger = logging.getLogger(__name__)

from PIL import Image  # noqa: E402

WIDTH = 512
HEIGHT = 512
FPS = 15.0
RENDER_INTERVAL_S = 1.0 / FPS

# RGBA compositing background; the mouth images are transparent at the edges.
BACKGROUND = (40, 44, 52, 255)

# Normalized (0..1) RMS thresholds for mouth state selection.
CLOSED_RMS = 0.06
MID_RMS = 0.25

# Peak detector release: how fast the tracked peak decays per second.
RELEASE_RATE = 6.0
# While audio keeps flowing, hold the peak for this long so sub-word gaps
# between syllables don't snap the mouth shut.
HOLD_S = 0.18


def _rms(frame: rtc.AudioFrame) -> float:
    """Normalized (0..1) root-mean-square amplitude of an int16 PCM frame."""
    samples = array.array("h")
    samples.frombytes(bytes(frame.data))
    n = len(samples)
    if n == 0:
        return 0.0
    acc = 0.0
    for v in samples:
        acc += v * v
    return math.sqrt(acc / n) / 32768.0


class Avatar:
    """Renders the mouth-state images onto a video source published to the room."""

    def __init__(
        self,
        assets_dir: Path,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
        fps: float = FPS,
        background: tuple[int, int, int, int] = BACKGROUND,
    ) -> None:
        self._width = width
        self._height = height
        self._interval = 1.0 / fps
        self._background = background

        self._frames: dict[str, bytes] = self._load_states(assets_dir)
        self._state = "closed"
        self._peak = 0.0
        self._latest_rms = 0.0
        self._last_frame_t = 0.0
        self._running = False
        self._render_task: asyncio.Task[None] | None = None
        self._publication: rtc.LocalTrackPublication | None = None

        self._source = rtc.VideoSource(width=self._width, height=self._height)

    def _load_states(self, assets_dir: Path) -> dict[str, bytes]:
        frames: dict[str, bytes] = {}
        for state in ("closed", "mid", "open"):
            path = Path(assets_dir) / f"mouth_{state}.png"
            img = Image.open(path).convert("RGBA").resize((self._width, self._height))
            bg = Image.new("RGBA", img.size, self._background)
            out = Image.alpha_composite(bg, img)
            bgra = bytearray(self._width * self._height * 4)
            i = 0
            for r, g, b, a in out.getdata():
                bgra[i] = b
                bgra[i + 1] = g
                bgra[i + 2] = r
                bgra[i + 3] = a
                i += 4
            frames[state] = bytes(bgra)
        return frames

    @property
    def publication(self) -> rtc.LocalTrackPublication | None:
        return self._publication

    async def publish(self, room: rtc.Room) -> None:
        """Publish the synthetic video track to the room's local participant."""
        track = rtc.LocalVideoTrack.create_video_track("avatar_video", self._source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        self._publication = await room.local_participant.publish_track(track, options)
        logger.info("avatar video track published as %s", self._publication.sid)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_frame_t = time.monotonic()
        self._render_task = asyncio.create_task(self._render_loop(), name="avatar_render")

    async def stop(self) -> None:
        self._running = False
        if self._render_task is not None:
            await asyncio.gather(self._render_task, return_exceptions=True)
            self._render_task = None
        await self._source.aclose()

    def feed_audio(self, frame: rtc.AudioFrame) -> None:
        """Record the amplitude of one outgoing audio frame."""
        self._latest_rms = _rms(frame)
        self._last_frame_t = time.monotonic()

    def _state_for(self, peak: float) -> str:
        if peak < CLOSED_RMS:
            return "closed"
        if peak < MID_RMS:
            return "mid"
        return "open"

    async def _render_loop(self) -> None:
        last_tick = time.monotonic()
        try:
            while self._running:
                now = time.monotonic()
                dt = now - last_tick
                last_tick = now

                # decay the tracked peak; only re-arm it while audio is flowing
                if now - self._last_frame_t < HOLD_S:
                    target = self._latest_rms
                else:
                    target = 0.0
                self._peak = max(target, self._peak * math.exp(-RELEASE_RATE * dt))

                state = self._state_for(self._peak)
                if state != self._state:
                    self._state = state

                frame = rtc.VideoFrame(
                    width=self._width,
                    height=self._height,
                    type=int(rtc.VideoBufferType.BGRA),
                    data=self._frames[state],
                )
                self._source.capture_frame(frame)

                await asyncio.sleep(max(0.0, self._interval - (time.monotonic() - now)))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("avatar render loop failed")
            raise


class AmplitudeTap(AudioOutput):
    """AudioOutput wrapper that feeds outgoing-audio amplitude to an Avatar.

    Sits between the AgentSession and the RoomIO sink: observes every frame
    destined for the speakers (``feed_audio``) and passes it through unchanged,
    preserving all playback/segment accounting for the sink below.
    """

    def __init__(self, avatar: Avatar, next_in_chain: AudioOutput) -> None:
        super().__init__(
            label="AvatarAmplitudeTap",
            capabilities=AudioOutputCapabilities(pause=True),
            next_in_chain=next_in_chain,
        )
        self._avatar = avatar
        self._capturing = False

    @property
    def sample_rate(self) -> int | None:
        if self._sample_rate is not None:
            return self._sample_rate
        return self.next_in_chain.sample_rate if self.next_in_chain else None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self._avatar.feed_audio(frame)
        await super().capture_frame(frame)
        if self.next_in_chain is not None:
            await self.next_in_chain.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        if self.next_in_chain is not None:
            self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        if self.next_in_chain is not None:
            self.next_in_chain.clear_buffer()
