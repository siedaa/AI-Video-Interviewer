# FirstRound Architecture

## System overview

FirstRound is an AI interviewer for a Junior AI Engineer role that works in three stages (matching PRD §1): offline, it parses a JD + resume + the candidate's real GitHub repos into a 12-question plan that a human recruiter explicitly approves; live, it joins a LiveKit room as a voice agent built on Gemini Live, interviews the candidate for ~8-10 minutes, adapts follow-up depth to answer quality, and stops instantly when interrupted; and after the call it will produce a quote-backed scorecard and expose everything through an MCP server usable from Claude Desktop. All interview context (JD, resume, GitHub findings, question plan) is known before the call starts and is injected as static context — there is no live tool-calling mid-call (deliberate reliability tradeoff, see Known limitations).

## Graph diagram

The offline prep pipeline runs as a LangGraph state machine (`src/graph.py`) with a genuine human-in-the-loop pause. Plain-text structure as of Phase 2 (4 data nodes + 1 real conditional edge + 1 HITL interrupt):

```
START
  |
  v
[parse_jd]          node : inputs/jd.txt -> state["jd"]
  |
  v
[parse_resume]      node : inputs/resume.pdf -> state["resume"]
  |
  v
CONDITIONAL EDGE #1 ("route_github")          <-- real, data-dependent
  | resume.github_url present?
  |   yes -> [github_agent]
  |   no  -> [request_github_manually] (interrupt: CLI asks for username)
  |                    |
  |                    v (override merged into resume.github_url, edge to github_agent)
  v
[github_agent]      node : resume.github_url + GITHUB_PAT
                      -> state["github_findings"] (repos, languages, commits,
                         README + real file excerpts for top 3)
  v
[question_planner]  node : jd + resume + github_findings
                      -> state["question_plan"] (12 questions)
                      -> state["approval_status"] = "pending"
  v
[hitl_gate]         node : interrupt() -- graph SUSPENDS, returns plan to CLI.
  |   CLI returns Command(resume={"action": ...}) via the same thread_id.
  |   approve -> writes output/prep/question_plan.json, approval_status="approved"
  |   edit    -> applies edits, edits_made += [...], approval_status="edited"
  |   reject  -> approval_status="rejected" (no file written)
  v
CONDITIONAL EDGE #2 ("route_approval")
  | approved -> END
  | edited   -> [hitl_gate] again (re-review loop)
  | rejected -> END
  v
END
```

Checkpointer: SqliteSaver on `checkpoint.db`, keyed by `thread_id`; a dropped call or restart resumes the exact thread state by re-invoking with the same `thread_id`. **This will grow in later phases per PRD §4** — the target is 6+ nodes and 2+ conditional edges once guardrails, scoring, and any approval-feedback loops are wired in.

## State object

`GraphState` (`src/schemas.py:153`) is a `TypedDict` (not a Pydantic model) so LangGraph's update/reducer semantics apply out of the box; all fields are optional so partial inputs are valid.

| Field | Type | Why it's there |
|---|---|---|
| `inputs` | `dict[str, str]` | Raw file paths (jd, resume) handed to the graph on entry |
| `jd` | `JDInfo` | Parsed JD (role, skills, competencies) driving jd_fit questions |
| `resume` | `ResumeInfo` | Parsed resume (name, roles, skills, GitHub/LinkedIn links) driving resume_probe questions and the GitHub branch |
| `github_findings` | `GithubInfo` | Repos, languages, recent commits, README/file excerpts — real evidence for github_deepdive questions |
| `question_plan` | `QuestionPlan` | The 12-question plan produced by `question_planner` and consumed by the live agent |
| `thread_id` | `str` | Stable id keying the checkpointer / resumability |
| `approval_status` | `Literal["pending","approved","edited","rejected"]` | HITL gate state: gates whether the plan proceeds or gets re-reviewed |
| `edits_made` | `List[ApprovalEdit]` | Recruiter edits applied during the HITL gate, logged for audit |

## Realtime voice architecture

The live interview (`src/realtime/agent.py`) uses LiveKit Agents with the Google plugin's Gemini Live realtime model, hardcoded default `gemini-2.5-flash-native-audio-preview-12-2025` (a dated, pinned model name — the bare alias times out on the WebSocket handshake; see ITERATION_NOTES.md). Key facts:

- **Explicit dispatch**: the worker is registered with `agent_name="firstround-interviewer"`; the join JWT carries a `RoomConfiguration` naming that agent, so joining the room starts the worker.
- **Static context, no live tool-calling**: candidate name + approved question plan are loaded into the system prompt at session start. This is a deliberate tradeoff per PRD §5, not an oversight — everything the agent needs is known before the call.
- **Voice only**: modalities=[AUDIO], no video track yet (avatar is Phase 5).
- **Stage tracking**: stages are prompt-driven inside the model, so the transcript labels them via a heuristic — `SequenceMatcher` token-overlap (ratio >= 0.3) against the unasked question_plan questions, mapped source→stage (`jd→jd_fit`, `resume→resume_probe`, `github→github_deepdive`, `scenario→scenario`), with `intro` before the first match and `wrap_up` once all 12 are asked. Written to `output/transcript.json`.
- **Barge-in**: `InterruptionOptions(enabled=True, mode="vad", min_duration=0.15, backchannel_boundary=(0.0, 0.0))` — the LiveKit defaults (`min_duration=0.5`, `backchannel_boundary=(1.0, 1.0)`) were too sluggish (~1s+ reaction); tuned down for near-instant interrupt.

## Measured latency

Barge-in interrupt latency is the number ARCHITECTURE.md/SUBMISSION.md must report against the <1200ms target in the spec (§4). Measured live across multiple sessions: **sub-100ms to 624ms**, well under the 1200ms target and close to the ~1s ideal.

How it's measured (`BargeInLatencyTracker` in `src/realtime/agent.py`): it hooks the session's `agent_state_changed` and `user_state_changed` events. When the user transitions to `"speaking"` while the agent is `"speaking"`, a monotonic start timestamp is recorded; when the agent then leaves `"speaking"`, latency is computed as the difference and written to `output/interrupt_latency.json`. Writes are immediate per-event (synchronous `write_text`, not at session close) so no detected interrupt is lost to a tab close or connection drop. A 20ms floor filters startup/state-flicker noise (the original 150ms floor was eating valid sub-100ms readings). On a mid-session Gemini reconnect the tracker re-syncs from the session's authoritative state so post-reconnect interrupts are still measured.

## Known limitations

- **No live tool-calling mid-call** — the agent can't re-fetch anything during the interview; if it needs a fact not in question_plan.json, it can't get it. Deliberate reliability tradeoff, not an oversight.
- **Gemini Live model pinning** — newer preview aliases have handshake/WS bugs with LiveKit; the dated pinned model works. Diagnosed via `scripts/test_gemini_live.py`.
- **Transient WebSocket disconnects** — free-tier Gemini Live dropped once mid-session (1006 abnormal closure, keepalive timeout); the plugin's built-in retry logic auto-recovered, but a candidate could notice a brief audio gap.
- **Stage tracking is heuristic** — transcript `node` labels come from token-overlap matching, not an explicit model signal; edge cases can mislabel a turn.
- **Avatar is amplitude-driven, not phoneme-driven** (once Phase 5 is built) — mouth shape approximates speech, doesn't match it exactly. Acceptable per rubric wording but stated plainly.

## Placeholders

- **TODO — Phase 3**: guardrails — banned_questions (8 categories) + evidence_check (no-quote-no-score), with pytest coverage.
- **TODO — Phase 5**: avatar/video pipeline — synthetic video track from 2-3 mouth-state images driven by live audio amplitude, composited over the static portrait at ~15fps.
- **TODO — Phase 6**: scoring pipeline (`src/agents/scorer.py`) producing `output/scorecard.json` per schema + MCP server (`mcp_server/`, fastmcp, stdio) exposing get_candidate / get_question_plan / save_score / get_scorecard / list_interviews.
- **TODO — Phase 7**: evals — 5 personas (Strong, Average, Weak, Bluffer, Nervous) run through the scorer, results in `evals/results.md`.
