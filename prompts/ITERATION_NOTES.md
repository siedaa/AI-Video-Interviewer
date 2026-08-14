# Iteration Notes (v1 → v2)

Log of every real bug we diagnosed and fixed across Phases 2–7, in the prompt-engineering spirit of spec requirement #12: each entry says what the v1 did, what broke, what changed, and why the change was the right one.

---

## [Phase 2] Checkpoint-resume bug: resume re-ran the whole pipeline from scratch

**v1 did:** when resuming an interrupted thread (e.g. approving the plan after the HITL pause), the CLI re-invoked the graph with the *initial* state again.

**What broke:** the graph re-executed `parse_jd → parse_resume → github_agent → question_planner` from the top instead of resuming from the pending interrupt — wasting API calls and producing a fresh un-edited plan rather than the one the recruiter had just reviewed. The HITL edit/approve loop was therefore unusable after the first pause.

**What changed:** `run_graph.py` now reads the thread's *pending interrupts* from its checkpoint (`app.get_state` → `snapshot.tasks[].interrupts`) and answers them with `Command(resume={...})` on the same `thread_id` — it never re-supplies the initial state on resume. (Commit `eac96a7`.) The checkpoint serde was also switched to `JsonPlusSerializer` with an explicit allowlist of the `src.schemas` models so the Pydantic state round-trips through msgpack.

**Why it's right:** the checkpointer *is* the resume mechanism; re-invoking with initial state defeats its purpose. Resume = "answer the pending interrupt", not "start over".

---

## [Phase 4] Browser autoplay AudioContext block: no audio until a user gesture

**v1 did:** `room.html` attached the agent's audio track and called `audioEl.play()` on `TrackSubscribed`, with no regard for the browser's autoplay policy.

**What broke:** browsers suspend the `AudioContext` until a user gesture grants activation, so on first join the agent's voice was silent (play was rejected / context stayed `suspended`). The call "worked" server-side but was mute client-side.

**What changed:** the **Join Interview** click handler now synchronously creates and `resume()`s a `window.AudioContext` at the very top of the handler, *before any `await`*, and re-arms it on each audio-track subscribe. Because the click is a direct user gesture, this grants the page user-activation that allows audio playback for the rest of the session.

**Why it's right:** autoplay is blocked by design; the fix authenticates the page's audio through the one channel the browser trusts (a synchronous gesture handler), instead of fighting the policy with muted/`unmute()` hacks.

---

## [Phase 4] Gemini Live model-alias handshake failure

**v1 did:** used the bare model alias `gemini-2.5-flash-native-audio-preview` from the PRD.

**What broke:** *every* Gemini Live WebSocket attempt timed out during the opening handshake (`TimeoutError: timed out during opening handshake`), despite a valid API key.

**What changed:** built an isolated test script `scripts/test_gemini_live.py` that opens a Gemini Live session directly via `google.genai`, bypassing LiveKit — it succeeded instantly with the pinned dated name `gemini-2.5-flash-native-audio-preview-12-2025`, proving the alias (not the key, not LiveKit) was broken. Agent default hardcoded to the dated pinned model (still overridable via `FIRSTROUND_GEMINI_MODEL`).

**Why it's right:** the PRD's stack notes warned the newest preview aliases have LiveKit-integration bugs; testing the failure in isolation before debugging inside the full stack is the pattern that got a fast diagnosis.

---

## [Phase 4] Model-string-corruption bug (1008 policy violation)

**v1 did:** the model name passed to the realtime model was correct at first — then, during a midnight Phase 4 session, connections started failing.

**What broke:** LiveKit/Gemini rejected the connection with a **1008 policy violation**. Reading the exact error string character-by-character showed the model name had been corrupted: stray characters were appended, turning `...12-2025` into `...12-2025QARTSS`.

**What changed:** traced the corruption to its source (stray characters being appended to the model string) and removed them, restoring the clean pinned name. A source-of-truth fix, not a masking retry.

**Why it's right:** a validation failure that looks like a permissions/policy problem was a data bug in the request string. Check the exact value being sent before assuming the service is rejecting the request.

---

## [Phase 4 — race diagnosed, fixed in Phase 5] AgentSession close race (generate_reply / avatar setup crash)

**v1 did:** after `await session.start(...)`, the entrypoint proceeded straight to `session.generate_reply()` and avatar video-track publish without checking whether the session was still alive.

**What broke:** if the participant disconnected during startup (a candidate's connection blipping mid-call is a real case, not just a test artifact), the code hit `RuntimeError: AgentSession isn't running` instead of exiting gracefully.

**What changed:** added `_session_live(session, room)` in `src/realtime/agent.py` — checks the room connection *and* the session's internal closing/activity state (`session._is_closing()` / `_activity`, the same state `generate_reply()` guards on). The avatar's close handler is wired immediately after `start()`, and both avatar setup and `generate_reply()` are gated behind liveness checks, logging `[entrypoint] session closed during startup; skipping ...` and returning cleanly. (Committed with the avatar in `0887095`.)

**Why it's right:** async teardown means the room's connection state alone is not a reliable "is the session live" signal — the agent's room can stay up after the candidate leaves. The check must mirror exactly what the downstream call itself validates.

---

## [Phase 4d] Latency floor filter was eating valid sub-100ms readings

**v1 did:** `BargeInLatencyTracker` ignored any measurement below a 150ms floor, treating it as startup/state-flicker noise.

**What broke:** real interrupts were coming in under 100ms (the tuned barge-in config is near-instant), and the tracker silently discarded them — the measured distribution looked worse than reality.

**What changed:** lowered the floor to 20ms (commit `29fa083`) and made latency writes immediate per-event (`write_text`, not deferred to session close) so no detected interrupt is lost to a tab close or connection drop; the tracker also re-syncs state after a recoverable Gemini reconnect.

**Why it's right:** a noise filter must sit below the genuine signal, not swallow it; and writing on the event path beats writing on the (often unreachable) close path.

---

## [Phase 5] room.html was audio-only: avatar video track never rendered

**v1 did:** `room.html`'s `TrackSubscribed` handler only attached audio tracks to the `<audio>` element.

**What broke:** once Phase 5 published the avatar's `SOURCE_CAMERA` video track, the face never appeared — the handler ignored the `Video` kind entirely, so the `<video>` element stayed empty despite the agent publishing video.

**What changed:** added a `Video` branch that does `track.attach(videoEl)` against `<video id="agent-video" autoplay playsinline>` (commit `ec5f242`).

**Why it's right:** the join page is the grader's window into the avatar; a "video call" that renders no video is not a video call. The autoplay/`playsinline` attributes keep it consistent with the audio unlock fix above.

---

## [Phase 7] Evals: Bluffer ranks above Average (the "confident and wrong" failure)

**v1 did:** scorer (`src/agents/scorer.py`) sent each persona transcript through Groq `chat_structured` with the rubric, then `evidence_check.py` re-validated that every `evidence_quote` is a verbatim transcript substring before writing the scorecard.

**What broke:** `evals/results.md` ranks strong 4.6 > **bluffer 3.6** > nervous 3.4 > average 2.6 > weak 1.8 — Bluffer lands **#2**, above both Average and Nervous, in two separate runs of identical transcripts. The scorer's LLM rewards fluent, well-structured but *empty* answers; the guardrail checks quote provenance, not correctness, so a confident wrong answer is still quotable.

**What changed:** nothing in the scorer — per the PRD §7 exit criteria we documented the failure honestly (`evals/results.md`) rather than hand-tune the prompt to force the expected table. The honest analysis suggests Phase-8 follow-ups: reward answer depth / specificity-to-follow-ups in the rubric, derive `recommendation` from `overall_score` thresholds instead of the LLM, and treat very short/non-technical quotes as a heuristic flag.

**Why it's right:** the evals exist to find this exact failure mode; a scorer that only separates Strong from Weak is trivially correct, and an honest "my scorer failed here and here's why" is the deliverable. Reporting it beats tuning it away.