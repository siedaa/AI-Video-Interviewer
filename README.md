# FirstRound — AI Video Interviewer

An AI agent that parses a JD + resume + GitHub repos into an approved question plan, interviews a candidate live as a face+voice on a video call, produces an evidence-backed scorecard, and exposes everything through an MCP server.

## Setup

TODO: fill in as we go.

## Run locally

TODO: fill in as we go.

## Project structure

```
src/            # core code: graph, nodes, agents, realtime, guardrails
mcp_server/     # MCP server for Claude Desktop
prompts/        # prompt templates + iteration notes
evals/          # personas + scoring evals
inputs/         # jd.txt, resume.pdf
output/         # prep/, transcript.json, scorecard.json, report.pdf
scripts/        # key verification and run scripts
```
