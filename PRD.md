# PRD — FirstRound: AI Video Interviewer
**Status doc — any AI/session should read this file top to bottom before writing code, then update the Phase Tracker at the bottom before stopping.**

---

## 1. What we're building (one paragraph)

An AI agent that: (1) offline, parses a JD + resume + the candidate's real GitHub repos into a question plan a human recruiter approves; (2) live, joins a video call as a face+voice, interviews the candidate for 8+ minutes, adapts follow-up depth to answer quality, and stops instantly when interrupted; (3) after the call, produces a scorecard where every score is backed by a real transcript quote; (4) exposes all of this through an MCP server usable from Claude Desktop.

Full spec: see the uploaded `FirstRound-Final-Test.pdf` — this PRD translates that spec into a build order and locks in the stack so we don't re-litigate decisions every session.

**Chosen JD:** #1 — Junior AI Engineer, Northwind Labs, Karachi (0–2 yrs). Probes: Python fundamentals, LangChain/LangGraph, RAG, Git/REST, one shipped project.

---

## 2. Locked-in stack (do not change without updating this section)

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** (Python) + SQLite checkpointer | Required by spec; checkpointer gives us call-drop resume for free |
| Realtime voice | **Gemini Live API**, model `gemini-2.5-flash-native-audio-preview` | Native audio in/out + built-in barge-in. Pin this exact model — the newer 3.1 preview has open compatibility bugs with LiveKit as of mid-2026 |
| No live tool-calling | Question plan is injected as **static context** in the system prompt at session start, not fetched via live function calls | Gemini Live's mid-response tool-calling has known bugs (stalls/truncation). We don't need live lookups — everything the agent needs (JD, resume, GitHub findings, question plan) is known before the call starts |
| Transport | **LiveKit Cloud (free tier) + LiveKit Agents (Python)** | Handles WebRTC, voice activity detection, and interruption plumbing so we don't hand-roll it |
| AI face | **Simple 2D lip-sync**: one static portrait + 2–3 mouth-position overlay images swapped based on live audio amplitude, rendered as a canvas/video track | Full marks per rubric footnote; far less fragile than Simli/Tavus trial credits under time pressure |
| STT/parsing LLM (offline, non-realtime) | **Groq** (fast, you already have a working pipeline from your lead-gen agent) for JD/resume parsing and GitHub analysis | Reuse your existing Groq integration pattern |
| GitHub grounding | **GitHub REST API + PAT** (5,000 req/hr) | Fallback (unauthenticated 60/hr) only if PAT setup fails |
| Resume parsing | `pdfplumber` + Groq structured output | |
| MCP server | **fastmcp (Python)**, stdio transport | Simplest path to "works in Claude Desktop" |
| Report | `reportlab` (PDF) | Markdown fallback costs −1 mark, avoid it |
| Deploy | Localhost is fine (bonus mark only for live URL) — don't burn time on Render/Railway until everything works locally |

**Environment variables needed (put names only in `.env.example`, never commit the real `.env`):**
`GOOGLE_API_KEY` (Gemini), `GROQ_API_KEY`, `GITHUB_PAT`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`

---

## 3. Repo structure (fixed — grading reads these exact paths)

```
firstround/
├─ README.md
├─ ARCHITECTURE.md
├─ SUBMISSION.md
├─ .env.example
├─ src/
│  ├─ graph.py
│  ├─ nodes/
│  ├─ agents/         # github_agent, resume_parser, jd_parser, question_planner, scorer
│  ├─ realtime/        # transport, avatar, barge-in
│  └─ guardrails/       # banned_questions, evidence_check
├─ mcp_server/
├─ prompts/ + ITERATION_NOTES.md
├─ evals/ (personas/, run_evals.py, results.md)
├─ inputs/ (jd.txt, resume.pdf)
└─ output/
   ├─ prep/ (jd.json, resume.json, github.json, question_plan.json)
   ├─ transcript.json
   ├─ scorecard.json
   └─ report.pdf
```

Schemas for `scorecard.json`, `transcript.json`, `question_plan.json` are in the source PDF §6 — copy them verbatim into `src/schemas.py` as Pydantic models in Phase 1, don't re-derive them later.

---

## 4. Build order — 8 phases

Work through these **in order**. Each phase has a goal, what "done" looks like, and a ready-to-paste prompt for opencode. Don't start a phase until the previous one's exit criteria are met.

### Phase 0 — Scaffolding + keys (do this first, always)
**Goal:** every API key verified working before writing real logic.
**Exit criteria:** a script that pings Gemini, Groq, GitHub, and LiveKit and prints "OK" for all four.
**Opencode prompt:**
> Create the repo skeleton exactly matching the structure in PRD.md §3. Add a `scripts/verify_keys.py` that loads `.env` and makes one trivial authenticated call to each of: Gemini API, Groq API, GitHub REST API, LiveKit server API — printing OK/FAIL per service. Add `.env.example` with the key names from PRD.md §2 and nothing else.

### Phase 1 — Prep pipeline (offline, no LangGraph yet)
**Goal:** JD + resume + GitHub → `question_plan.json`, no manual copy-paste.
**Exit criteria:** running one script on `inputs/jd.txt` + `inputs/resume.pdf` produces valid `output/prep/*.json` files matching the schemas, with ≥3 questions in the plan tagged `"source": "github"` and a real repo/file reference.
**Opencode prompt:**
> Build `src/agents/jd_parser.py`, `resume_parser.py`, `github_agent.py`, `question_planner.py`. jd_parser and resume_parser use pdfplumber + Groq structured JSON output to fill the schemas in src/schemas.py. github_agent finds the candidate's GitHub link from the resume, pulls repos/languages/README/recent commits via GitHub REST API, and summarizes real evidence per repo. question_planner combines all three into 12 questions per the question_plan.json schema, at least 3 with source="github" citing a specific repo/file. Add a CLI entry point `run_prep.py`.

### Phase 2 — LangGraph skeleton with HITL gate
**Goal:** the prep pipeline becomes a real graph with a genuine pause for recruiter approval.
**Exit criteria:** running the graph stops after `question_planner`, waits for approve/edit/reject input, and only proceeds on approve.
**Opencode prompt:**
> Wrap the Phase 1 agents as LangGraph nodes in src/graph.py with typed state (src/schemas.py). Add a human-in-the-loop interrupt after question_planner using LangGraph's interrupt/Command pattern. Add a SQLite checkpointer. Write a small CLI harness that shows the plan, accepts approve/edit/reject, and only continues on approve — log edits to `edits_made` in question_plan.json.

### Phase 3 — Guardrails
**Goal:** banned-topic filter + no-quote-no-score rule, both with tests.
**Exit criteria:** `pytest` proves a banned question (age/gender/marital/religion/nationality/health/salary/politics) never reaches the candidate, and a score without an `evidence_quote` is rejected.
**Opencode prompt:**
> Build src/guardrails/banned_questions.py (keyword + LLM-based classifier for the 8 banned categories from PRD.md, blocking before the question is asked) and evidence_check.py (rejects any scorecard competency entry with an empty or non-matching evidence_quote). Write tests in tests/ that prove both fire, including at least one adversarial banned-question phrasing.

### Phase 4 — Realtime call: voice only, no face yet
**Goal:** highest-risk piece solved first — a joinable LiveKit room where Gemini Live speaks, listens, and stops when interrupted.
**Exit criteria:** you can join the room from a browser, the agent greets you by name using data from question_plan.json, and talking over it makes it stop within ~1 second (measure and record the number).
**Opencode prompt:**
> Build src/realtime/agent.py using LiveKit Agents with the Google plugin, model gemini-2.5-flash-native-audio-preview, modalities=[AUDIO]. Load candidate name + question_plan.json into the system prompt at session start — no live tool calls. Implement the conversation flow as prompt-driven stages: intro → resume_probe → jd_fit → github_deepdive → scenario → candidate_Qs → wrap_up, with instructions in the prompt for adaptive follow-up (shallow answer → probe, max 2, then move on; strong answer → raise difficulty). Log every turn to output/transcript.json per the schema, including an `interrupted` boolean. Add a LiveKit room join page (minimal HTML) for testing.

### Phase 5 — Avatar (2D lip-sync)
**Goal:** visible face synced to the agent's speech.
**Exit criteria:** the AI's video tile shows a portrait whose mouth changes shape while it talks and goes still while it listens.
**Opencode prompt:**
> Add src/realtime/avatar.py: publish a synthetic video track to the LiveKit room built from 3 mouth-state images (closed/mid/open) selected by real-time audio amplitude of the outgoing Gemini Live audio stream, composited over the static portrait at ~15fps. Wire it into the Phase 4 agent so it publishes alongside the audio track.

### Phase 6 — Scoring + MCP server
**Goal:** post-call scorecard generation + MCP tools live in Claude Desktop.
**Exit criteria:** running the scorer on a transcript produces a valid scorecard.json (guardrail-checked), and Claude Desktop can call all 5 MCP tools successfully.
**Opencode prompt:**
> Build src/agents/scorer.py: takes output/transcript.json + question_plan.json, produces output/scorecard.json per schema via Groq, running evidence_check.py on every entry before writing. Build mcp_server/ with fastmcp exposing get_candidate, get_question_plan, save_score, get_scorecard, list_interviews over stdio. Add setup instructions for registering it in Claude Desktop's config.

### Phase 7 — Evals
**Goal:** prove the scorer separates confident-and-wrong from anxious-and-right.
**Exit criteria:** `evals/results.md` shows all 5 personas ranked correctly, especially Bluffer landing below Average and Nervous landing near Strong — with honest notes on any persona that misranks.
**Opencode prompt:**
> Write 5 synthetic transcripts in evals/personas/ (Strong, Average, Weak, Bluffer, Nervous per PRD §13 behaviors). Build evals/run_evals.py to run each through scorer.py and output a ranking table plus pass/fail against expected ranking to evals/results.md. Do not hand-tune the scorer prompt to force a perfect result — report failures honestly.

### Phase 8 — Docs, real interview, videos, submission
**Goal:** everything the rubric grades outside of code.
**Exit criteria:** ARCHITECTURE.md, SUBMISSION.md, prompts/ITERATION_NOTES.md complete; one real 8+ min recorded interview with consent; two edited videos within time limits; clean-clone test passes.
**This phase is manual/you-driven — I'll help you write these docs once Phases 0–7 are real and working, not before.**

---

## 5. Known risks (write these into ARCHITECTURE.md's "limitations" section later)

- Gemini Live's newest preview models have open LiveKit-integration bugs — we're pinned to the stable model on purpose.
- No live tool-calling means the agent can't re-fetch anything mid-call; if it needs a fact not in question_plan.json, it can't get it. This is a deliberate tradeoff for reliability, not an oversight — say so in the docs.
- The avatar is amplitude-driven, not phoneme-driven — mouth shape approximates speech, doesn't match it exactly. Acceptable per rubric wording but worth stating plainly.

## 6. Phase Tracker — update this before ending any session

| Phase | Status | Notes |
|---|---|---|
| 0 — Scaffolding | In progress | Repo skeleton, `.venv`, verify_keys.py in place; all 4 key checks pass (Gemini/Groq/GitHub/LiveKit OK). Awaiting user's own run of verify_keys.py. |
| 1 — Prep pipeline | In progress | src/schemas.py, jd_parser, resume_parser, github_agent, question_planner, run_prep.py written. Ran end-to-end on test inputs: 12 questions / 4 github-grounded with verified repo/file/commit refs. Stopping for user to test run_prep.py. |
| 2 — LangGraph + HITL | In progress | src/graph.py (5 nodes + 2 conditional edges + SqliteSaver on checkpoint.db), run_graph.py CLI with approve/edit/reject. Tested approve, edit->reapprove, reject, github-override interrupt, and crash-resume. Awaiting user test. |
| 1 — Prep pipeline | Not started | |
| 2 — LangGraph + HITL | Not started | |
| 3 — Guardrails | Not started | |
| 4 — Realtime voice | Not started | |
| 5 — Avatar | Not started | |
| 6 — Scoring + MCP | Not started | |
| 7 — Evals | Not started | |
| 8 — Docs + submission | Not started | |
