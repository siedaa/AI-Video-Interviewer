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

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(ROOT, ".env"))

from google.genai import types  # noqa: E402

from livekit import agents  # noqa: E402
from livekit.agents import Agent, AgentServer, AgentSession, AutoSubscribe  # noqa: E402
from livekit.plugins import google  # noqa: E402

from src.schemas import QuestionPlan, ResumeInfo  # noqa: E402

GEMINI_REALTIME_MODEL = os.environ.get(
    "FIRSTROUND_GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview"
)
QUESTION_PLAN_PATH = Path(
    os.environ.get("FIRSTROUND_QUESTION_PLAN", ROOT / "output" / "prep" / "question_plan.json")
)
RESUME_PATH = Path(os.environ.get("FIRSTROUND_RESUME", ROOT / "output" / "prep" / "resume.json"))


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

    session = AgentSession(llm=model)

    await session.start(room=ctx.room, agent=InterviewerAgent(build_system_prompt(resume, plan)))

    await session.generate_reply(
        instructions=(
            "Begin the interview now: introduce yourself as the FirstRound AI interviewer, "
            f"greet the candidate by name ({resume.name}), and ask the first question from your plan."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)