# FirstRound — AI Video Interviewer

An AI that parses a real JD + resume + GitHub repos into a recruiter-approved question plan, joins a LiveKit video call as a face + voice (Gemini Live + a 2D lip-synced avatar), interviews the candidate live with adaptive follow-ups and instant barge-in, produces an evidence-quote-backed scorecard, and exposes everything through an MCP server for Claude Desktop.

JD chosen: **#1 — Junior AI Engineer, Northwind Labs, Karachi (0–2 yrs)**. Stack is locked in `PRD.md` §2 (LangGraph + SQLite checkpointer, Gemini Live pinned to `gemini-2.5-flash-native-audio-preview-12-2025`, Groq for offline parsing/scoring, fastmcp, reportlab).

## Setup (from a clean clone)

1. **Clone + enter**
   ```bash
   git clone https://github.com/siedaa/AI-Video-Interviewer.git
   cd AI-Video-Interviewer
   ```
   (Repo is currently private — the grader needs access before cloning.)

2. **Create the venv**
   ```bash
   python -m venv .venv
   # Windows:  .\.venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   ```

3. **Install + configure keys**
   ```bash
   pip install -r requirements.txt
   # Windows:  copy .env.example .env
   # macOS/Linux: cp .env.example .env
   ```
   Fill the 6 keys in `.env` (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `GITHUB_PAT`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`). `.env` is gitignored — only names are committed.

4. **Prep + approval**
   ```bash
   python run_prep.py            # JD + resume + GitHub -> output/prep/*.json (direct pipeline)
   python run_graph.py           # same pipeline as a LangGraph with a HITL pause; approve/edit/reject
   ```
   Both write `output/prep/{jd,resume,github,question_plan}.json`. `run_graph.py` is the graded path (real pause, checkpointer on `checkpoint.db`).

5. **Run everything else**
   ```bash
   python scripts/make_join_token.py     # prints a join URL (room + agent dispatch baked in)
   python -m http.server 8000 --directory src/realtime   # serve room.html (mic needs http, not file://)
   python -m src.realtime.agent dev      # worker joins the room, greets the candidate, interviews
   python mcp_server/server.py           # MCP server (stdio) for Claude Desktop
   python evals/run_evals.py             # run the 5 eval personas through the scorer (--reuse skips re-scoring)
   ```

## Project structure

```
src/            # graph.py (LangGraph), agents/, realtime/ (agent + avatar + room.html), guardrails/, schemas.py
mcp_server/     # fastmcp stdio server (5 tools) + Claude Desktop registration README
prompts/        # ITERATION_NOTES.md (v1->v2 bug log)
evals/          # personas/, run_evals.py, results.md
inputs/         # jd.txt, resume.pdf (untracked)
output/         # prep/, transcript.json, scorecard.json, interrupt_latency.json
docs/           # screenshots (MCP-in-Claude-Desktop proof)
tests/          # pytest: guardrails, scorer, MCP server
scripts/        # verify_keys.py, test_gemini_live.py, make_join_token.py
```

Key docs: `ARCHITECTURE.md` (graph diagram, state object, measured latency, limitations), `SUBMISSION.md` (one-page submission), `PRD.md` (status + phase tracker).
