# Iteration Notes

_TODO: log prompt iterations and decisions here as we go (Phase 8)._

---

## [Phase 4] Gemini Live model alias handshake failure

**What broke:** Using the bare model alias `gemini-2.5-flash-native-audio-preview` caused every Gemini Live WebSocket connection attempt to time out during the opening handshake (`TimeoutError: timed out during opening handshake`), even though the API key itself was valid.

**How we diagnosed it:** Built an isolated test script (`scripts/test_gemini_live.py`) that opened a Gemini Live session directly via `google.genai`, bypassing LiveKit entirely, to rule out whether the problem was the key, the model, or the LiveKit integration layer. That script succeeded instantly using the pinned dated model name `gemini-2.5-flash-native-audio-preview-12-2025`, proving the alias itself was the problem, not the key or LiveKit.

**The fix:** Hardcoded the pinned dated model name as the default in `agent.py` (still overridable via `FIRSTROUND_GEMINI_MODEL` env var).

**Why this matters:** This is exactly the kind of failure the PRD's stack notes warned about — Gemini Live's newest preview aliases have had integration bugs, which is why we pinned a specific model from the start. The isolated-test-script approach (proving the failure outside the full stack before debugging inside it) is a pattern worth reusing for Phase 6 (Groq scorer) if similar issues come up there.