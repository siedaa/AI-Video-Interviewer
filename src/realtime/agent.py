"""FirstRound realtime interviewer: a LiveKit AgentServer voice agent.

Runs as a LiveKit worker using Gemini Live (Google plugin, native audio)
to conduct the prepared interview. At session start it loads the candidate
name + approved question plan into the system prompt as static context —
no live tool calls.

Run:
    python src/realtime/agent.py dev          # connect to LiveKit + console
    python src/realtime/agent.py start        # serve dispatch jobs
"""

from __future__ import annotations

import json
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(ROOT, ".env"))

os.environ.pop("GEMINI_API_KEY", None)

from google.genai import types  # noqa: E402

from livekit import agents  # noqa: E402
from livekit.agents import Agent, AgentServer, AgentSession, AutoSubscribe  # noqa: E402
from livekit.agents.llm import ChatMessage  # noqa: E402
from livekit.plugins import google  # noqa: E402

from src.schemas import QuestionPlan, ResumeInfo, Transcript, Turn  # noqa: E402

PINNED_GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GEMINI_REALTIME_MODEL = os.environ.get(
    "FIRSTROUND_GEMINI_MODEL", PINNED_GEMINI_LIVE_MODEL
)
QUESTION_PLAN_PATH = Path(
    os.environ.get("FIRSTROUND_QUESTION_PLAN", ROOT / "output" / "prep" / "question_plan.json")
)
RESUME_PATH = Path(os.environ.get("FIRSTROUND_RESUME", ROOT / "output" / "prep" / "resume.json"))
TRANSCRIPT_PATH = Path(
    os.environ.get("FIRSTROUND_TRANSCRIPT", ROOT / "output" / "transcript.json")
)
INTERRUPT_LATENCY_PATH = Path(
    os.environ.get(
        "FIRSTROUND_INTERRUPT_LATENCY", ROOT / "output" / "interrupt_latency.json"
    )
)

STAGE_BY_SOURCE = {
    "jd": "jd_fit",
    "resume": "resume_probe",
    "github": "github_deepdive",
    "scenario": "scenario",
}
MATCH_THRESHOLD = 0.3


class TranscriptTracker:
    """Approximates the interview stage of each turn and persists the transcript.

    Stages are prompt-driven inside the model; there is no explicit signal from
    the model about which stage it is in. We approximate by matching each agent
    turn against the unasked question_plan questions (SequenceMatcher ratio,
    threshold ~0.3) and mapping the matched question's source to a stage.
    "intro" is used before the first match and "wrap_up" once every question
    has been asked at least once.
    """

    def __init__(self, plan: QuestionPlan, path: Path) -> None:
        self._plan = plan
        self._path = path
        self._asked: set[str] = set()
        self._node = "intro"
        self._turns: list[Turn] = []

    def on_conversation_item(self, message: ChatMessage) -> None:
        if message.role not in ("user", "assistant"):
            return
        text = message.raw_text_content or ""
        if not text.strip():
            return

        node = self._node_for(message)
        self._node = node

        self._turns.append(
            Turn(
                speaker="agent" if message.role == "assistant" else "candidate",
                text=text,
                timestamp_ms=int(message.created_at * 1000),
                node=node,
                interrupted=message.interrupted if message.role == "assistant" else False,
            )
        )
        self._flush()

    def _node_for(self, message: ChatMessage) -> str:
        if message.role == "assistant":
            for q in self._plan.questions:
                if q.id in self._asked:
                    continue
                ratio = SequenceMatcher(None, message.raw_text_content or "", q.text).ratio()
                if ratio >= MATCH_THRESHOLD:
                    self._asked.add(q.id)
                    return STAGE_BY_SOURCE.get(q.source, self._node)
        if len(self._asked) >= len(self._plan.questions):
            return "wrap_up"
        return self._node

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            Transcript(turns=self._turns).model_dump_json(indent=2), encoding="utf-8"
        )


class BargeInLatencyTracker:
    """Measures barge-in latency = time for the agent to stop speaking once interrupted.

    Latency (per spec) is from the moment the user starts speaking while the agent
    is speaking until the agent transitions out of the "speaking" state. Target
    < 1.2s, ideally ~1s. Each measurement is written to disk immediately (synchronously
    in _record) and printed to the console, so an interruption is never lost even if
    the browser tab closes or the Gemini websocket drops mid-session. Sub-floor
    readings (<FLOOR_MS) are startup/state-flicker noise and are discarded.
    FLOOR_MS=20 still filters the ~2.5ms startup flicker while keeping fast,
    valid sub-100ms barge-in responses.

    Reconnect safety: a mid-session Gemini reconnect surfaces as a session-level
    "error" event with a recoverable RealtimeModelError. We hook it to re-sync the
    mirrored agent_state from the session's authoritative state and drop any pending
    interrupt window that straddled the drop, so genuinely occurring interruptions
    after the reconnect are detected instead of being masked by stale state.
    """

    TARGET_MS = 1200.0
    IDEAL_MS = 1000.0
    FLOOR_MS = 20.0

    def __init__(self, path: Path, session: AgentSession | None = None) -> None:
        self._path = path
        self._session = session
        self._agent_state: str | None = None
        self._interrupt_started_ms: float | None = None
        self._measurements: list[float] = []
        self._load_existing()

    def bind(self, session: AgentSession) -> None:
        self._session = session

    def on_session_error(self, event) -> None:
        """Re-sync state tracking after a recoverable (reconnect) session error."""
        error = getattr(event, "error", None)
        if getattr(error, "recoverable", False):
            self._interrupt_started_ms = None
            self._agent_state = self._session.agent_state if self._session else None
            print("[barge-in] session error detected; re-synced state tracking")

    def _is_agent_speaking(self) -> bool:
        if self._session is not None:
            return self._session.agent_state == "speaking"
        return self._agent_state == "speaking"

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._measurements = [float(item["latency_ms"]) for item in data]
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            self._measurements = []

    def on_agent_state_changed(self, event) -> None:
        """Track agent speech; when it stops after an interrupt, record the latency."""
        was_speaking = self._agent_state == "speaking"
        self._agent_state = event.new_state
        if was_speaking and self._agent_state != "speaking":
            if self._interrupt_started_ms is not None:
                latency_ms = time.monotonic() * 1000 - self._interrupt_started_ms
                self._interrupt_started_ms = None
                self._record(latency_ms)

    def on_user_state_changed(self, event) -> None:
        """Mark interrupt start when the user starts speaking over a speaking agent."""
        if (
            event.new_state == "speaking"
            and self._is_agent_speaking()
            and self._interrupt_started_ms is None
        ):
            self._interrupt_started_ms = time.monotonic() * 1000

    def _record(self, latency_ms: float) -> None:
        if latency_ms < self.FLOOR_MS:
            return
        self._measurements.append(latency_ms)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                [{"latency_ms": round(m, 1)} for m in self._measurements],
                indent=2,
            ),
            encoding="utf-8",
        )
        status = "OK" if latency_ms < self.TARGET_MS else "OVER TARGET"
        print(
            f"[barge-in] latency_ms={latency_ms:.1f} "
            f"(target <{self.TARGET_MS:.0f}ms, ideal ~{self.IDEAL_MS:.0f}ms) [{status}]"
        )


class InterviewerAgent(Agent):
    """LiveKit agent whose only behavioral input is the static system prompt."""

    def __init__(self, system_prompt: str) -> None:
        super().__init__(instructions=system_prompt)


def _load_context() -> tuple[ResumeInfo, QuestionPlan]:
    resume = ResumeInfo.model_validate_json(RESUME_PATH.read_text(encoding="utf-8"))
    plan = QuestionPlan.model_validate_json(QUESTION_PLAN_PATH.read_text(encoding="utf-8"))
    return resume, plan


def _format_questions(plan: QuestionPlan) -> str:
    lines: list[str] = []
    for i, q in enumerate(plan.questions, 1):
        block = (
            f"{i}. [{q.id}] source={q.source} difficulty={q.difficulty} "
            f"competency={q.competency}\n"
            f"   {q.text}"
        )
        if q.source_reference:
            block += f"\n   reference: {q.source_reference}"
        if q.follow_up_triggers:
            block += f"\n   follow-up triggers: {', '.join(q.follow_up_triggers)}"
        lines.append(block)
    return "\n\n".join(lines)


def build_system_prompt(resume: ResumeInfo, plan: QuestionPlan) -> str:
    resume_bits = [f"Name: {resume.name}"]
    if resume.roles:
        roles = "; ".join(f"{r.title} at {r.company} ({r.years})" for r in resume.roles)
        resume_bits.append(f"Recent roles: {roles}")
    if resume.skills:
        resume_bits.append(f"Skills: {', '.join(resume.skills)}")
    if resume.github_url:
        resume_bits.append(f"GitHub: {resume.github_url}")

    candidate_block = "\n".join(f"- {bit}" for bit in resume_bits)

    return f"""You are FirstRound, an AI voice interviewer conducting a live interview for the Junior AI Engineer role at Northwind Labs (Karachi).

CANDIDATE (loaded before this call; you have no tools and must never try to fetch anything mid-call):
{candidate_block}

INTERVIEW STAGES — run them in order:
1. intro — greet the candidate by name, introduce yourself, set expectations for the call
2. resume_probe — validate resume claims, probe for real depth
3. jd_fit — probe fit against the role's must-have skills and competencies
4. github_deepdive — ground questions in the candidate's real GitHub evidence
5. scenario — hypotheticals and problem decomposition
6. candidate_Qs — invite the candidate's questions and answer them
7. wrap_up — summarize the next steps and close warmly

APPROVED QUESTION PLAN — ask these 12 questions across the stages (adapt the wording to sound natural, but do not replace them):

{_format_questions(plan)}

ADAPTIVE FOLLOW-UP BEHAVIOR:
- If the candidate's answer is shallow, probe deeper with one targeted follow-up on the specific gap (use that question's follow-up triggers).
- Maximum 2 follow-up probes per topic, then move to the next question.
- If the answer is strong and specific, raise the difficulty: ask a harder variant or a sharper deeper follow-up.
- Target about 8-10 minutes total. Speak naturally, ask one question at a time, and actually listen between turns.
"""


server = AgentServer()


@server.rtc_session(agent_name="firstround-interviewer")
async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    resume, plan = _load_context()

    model = google.realtime.RealtimeModel(
        model=GEMINI_REALTIME_MODEL,
        voice="Puck",
        temperature=0.7,
        modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(),
    )

    session = AgentSession(
        llm=model,
        turn_handling={
            "interruption": {
                "enabled": True,
                "mode": "vad",
                "min_duration": 0.15,
                "backchannel_boundary": (0.0, 0.0),
            }
        },
    )
    tracker = TranscriptTracker(plan, TRANSCRIPT_PATH)
    latency_tracker = BargeInLatencyTracker(INTERRUPT_LATENCY_PATH, session)

    def _on_conversation_item(event) -> None:
        if isinstance(event.item, ChatMessage):
            tracker.on_conversation_item(event.item)

    session.on("conversation_item_added", _on_conversation_item)
    session.on("agent_state_changed", latency_tracker.on_agent_state_changed)
    session.on("user_state_changed", latency_tracker.on_user_state_changed)
    session.on("error", latency_tracker.on_session_error)

    await session.start(room=ctx.room, agent=InterviewerAgent(build_system_prompt(resume, plan)))

    await session.generate_reply(
        instructions=(
            "Begin the interview now: introduce yourself as the FirstRound AI interviewer, "
            f"greet the candidate by name ({resume.name}), and ask the first question from your plan."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)