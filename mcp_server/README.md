# FirstRound MCP server

A [fastmcp](https://fastmcp.readthedocs.io/) stdio server that exposes the FirstRound
interview pipeline's data to Claude Desktop. Reads the single candidate's artifacts
from the repo's `output/` directory. No API keys needed — every tool is offline.

## Tools

| Tool | Signature | What it does |
|---|---|---|
| `get_candidate` | `get_candidate(candidate_id_or_name)` | Parsed resume + JD match info (matched/missing must-have skills). Matches against the current candidate's name; unknown IDs return a `not_found` response. |
| `get_question_plan` | `get_question_plan()` | The approved question plan. |
| `save_score` | `save_score(competency_name, score, evidence_quote, reasoning)` | Append/update one competency. The quote is validated against the transcript via `src/guardrails/evidence_check.py` first — a quote that is empty or not in the transcript is **rejected and returned as an error**, never saved. |
| `get_scorecard` | `get_scorecard()` | The full scorecard. |
| `list_interviews` | `list_interviews()` | Available interview outputs (currently the one candidate's `transcript.json`/`scorecard.json`). |

Every tool returns a structured dict with a `status` field: `ok`, `not_found`
(missing file / unknown ID — never a crash), or `error` (guardrail rejection).

**Candidate identity assumption:** the project tracks a single candidate as flat
files in `output/prep/`; there is no multi-candidate ID system yet. `get_candidate`
treats `output/prep/` as "the" candidate. `list_interviews` is structured to scale
if `output/` ever holds multiple interviews' data.

## Registering in Claude Desktop

The config file lives at (per OS):

| OS | Config path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` (usually `C:\Users\<you>\AppData\Roaming\Claude\claude_desktop_config.json`) |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

Add an `mcpServers` entry. Use the **absolute path to the script** (not `-m`) so it
works regardless of Claude Desktop's working directory — `server.py` resolves the
repo root from its own `__file__`:

```json
{
  "mcpServers": {
    "firstround": {
      "command": "C:\\Users\\sieda\\OneDrive\\Desktop\\AI video interviewer\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\sieda\\OneDrive\\Desktop\\AI video interviewer\\mcp_server\\server.py"
      ]
    }
  }
}
```

(Backslashes must be escaped as `\\` in JSON, or use forward slashes:
`"command": "C:/Users/sieda/OneDrive/Desktop/AI video interviewer/.venv/Scripts/python.exe"`.)

If you want to launch via `python -m mcp_server.server` instead, the client must
run with the repo root as the working directory (Claude Code supports a `cwd`
field for this; Claude Desktop does not — hence the script-path form above).

After editing, **restart Claude Desktop** (fully quit, then reopen), then check the
tools are live by opening Settings → Developer and looking for the `firstround`
MCP server (or just ask Claude to call `list_interviews`).

## Running / testing standalone

```powershell
# Live stdio server (pairs with an MCP client like Claude Desktop):
.\.venv\Scripts\python.exe mcp_server\server.py

# Smoke tests (calls every tool directly, incl. not-found + guardrail-rejection paths):
.\.venv\Scripts\python.exe -m pytest tests\test_mcp_server.py -q
```
