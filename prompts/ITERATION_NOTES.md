# Iteration Notes

_TODO: log prompt iterations and decisions here as we go (Phase 8)._

---

## [Phase 4] Gemini Live model alias handshake failure

**What broke:** Using the bare model alias `gemini-2.5-flash-native-audio-preview` caused every Gemini Live WebSocket connection attempt to time out during the opening handshake (`TimeoutError: timed out during opening handshake`), even though the API key itself was valid.

**How we diagnosed it:** Built an isolated test script (`scripts/test_gemini_live.py`) that opened a Gemini Live session directly via `google.genai`, bypassing LiveKit entirely, to rule out whether the problem was the key, the model, or the LiveKit integration layer. That script succeeded instantly using the pinned dated model name `gemini-2.5-flash-native-audio-preview-12-2025`, proving the alias itself was the problem, not the key or LiveKit.

**The fix:** Hardcoded the pinned dated model name as the default in `agent.py` (still overridable via `FIRSTROUND_GEMINI_MODEL` env var).

**Why this matters:** This is exactly the kind of failure the PRD's stack notes warned about — Gemini Live's newest preview aliases have had integration bugs, which is why we pinned a specific model from the start. The isolated-test-script approach (proving the failure outside the full stack before debugging inside it) is a pattern worth reusing for Phase 6 (Groq scorer) if similar issues come up there.

---

## [Phase 4] Model-string-corruption bug (1008 policy violation)

**What broke:** Midnight during Phase 4, the Gemini Live connection started failing to connect with a 1008 policy-violation error instead of handshaking. The model name passed to the realtime model appeared corrupted.

**How we diagnosed it:** Read the exact error string character-by-character in the failure output and found stray characters had been appended to the model name — the alias ended up as `...12-2025QARTSS` (a "QARTSS" suffix that was not part of the real alias).

**The fix:** Traced the corruption back to its source (stray characters being appended to the model string) and removed it, restoring the clean pinned model name `gemini-2.5-flash-native-audio-preview-12-2025`. A source-of-truth fix rather than masking the symptom.

**Why this matters:** A validation failure that reads like a permissions/policy problem ("1008 policy violation") was actually a data bug in the request string. Worth remembering: check the exact value being sent before assuming the service is rejecting the request.

---

## [Phase 5] Session-liveness race condition (generate_reply/avatar setup)

**What broke:** If the participant disconnected (or the session otherwise closed) before startup setup fully completed, the code still proceeded to `session.generate_reply()` and the avatar's video-track publish, which crashed with `RuntimeError: AgentSession isn't running` instead of detecting the closed session and exiting gracefully. This is a real edge case — a candidate's connection can blip mid-interview, not only in testing.

**The fix:** Added a `_session_live()` helper (`src/realtime/agent.py`) that checks the room connection plus the session's internal closing/activity state (`session._is_closing()` / `_activity` — the same state `generate_reply()` guards on). The avatar's close handler is now wired immediately after `start()`, and both the avatar setup and `generate_reply()` are gated behind liveness checks before proceeding. On a closed session it logs `[entrypoint] session closed during startup/avatar setup; skipping ...` and returns cleanly.

**Why this matters:** Async teardown means the room's connection state alone isn't a reliable "is the session live" signal — the agent's room can stay up after the candidate leaves. The check must mirror what the downstream call itself validates.