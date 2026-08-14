"""Smoke tests for the FirstRound MCP server (mcp_server/server.py).

Every tool is called directly (fastmcp keeps the decorated functions callable),
using tmp_path fixtures in place of the real output/ directory. Covers each of
the 5 tools plus the required not-found and guardrail-rejection paths.
"""

from __future__ import annotations

import json

import pytest

import mcp_server.server as server

CANDIDATE_TEXT = "I built a retrieval-augmented generation system using Chroma as the vector store."


@pytest.fixture()
def mcp_fs(tmp_path, monkeypatch):
    """Point the server at a throwaway output/ tree and seed it."""
    prep = tmp_path / "prep"
    prep.mkdir()
    monkeypatch.setattr(server, "RESUME_PATH", prep / "resume.json")
    monkeypatch.setattr(server, "JD_PATH", prep / "jd.json")
    monkeypatch.setattr(server, "PLAN_PATH", prep / "question_plan.json")
    monkeypatch.setattr(server, "TRANSCRIPT_PATH", tmp_path / "transcript.json")
    monkeypatch.setattr(server, "SCORECARD_PATH", tmp_path / "scorecard.json")

    (prep / "resume.json").write_text(json.dumps({
        "name": "Sieda Khan",
        "roles": [{"title": "Backend AI Intern", "company": "FlyRank",
                   "years": "2025-2026", "description": "Built a RAG service."}],
        "claims": ["Built a RAG chatbot."],
        "skills": ["Python", "FastAPI", "LangChain", "RAG", "Git"],
        "github_url": "https://github.com/siedaa",
        "linkedin_url": "https://www.linkedin.com/in/sieda-khan",
    }), encoding="utf-8")
    (prep / "jd.json").write_text(json.dumps({
        "role": "Junior AI Engineer", "seniority": "Junior",
        "company": "Northwind Labs", "location": "Karachi",
        "must_have_skills": ["Python", "LangChain or LangGraph", "RAG", "Git", "REST APIs"],
        "competencies": ["Technical depth", "Debugging"],
    }), encoding="utf-8")
    (prep / "question_plan.json").write_text(json.dumps({
        "approved_by_human": True,
        "questions": [
            {"id": "q1", "text": "Describe a RAG system you built.",
             "competency": "Technical depth", "source": "jd", "source_reference": "",
             "difficulty": "medium", "follow_up_triggers": []},
            {"id": "q2", "text": "How do you debug a 500 error?",
             "competency": "Debugging", "source": "scenario", "source_reference": "",
             "difficulty": "easy", "follow_up_triggers": []},
        ],
    }), encoding="utf-8")
    (tmp_path / "transcript.json").write_text(json.dumps({
        "turns": [
            {"speaker": "agent", "text": "Hello Sieda Khan, describe a RAG system you built.",
             "timestamp_ms": 1786742401000, "node": "intro", "interrupted": False},
            {"speaker": "candidate", "text": CANDIDATE_TEXT,
             "timestamp_ms": 1786742403000, "node": "resume_probe", "interrupted": False},
        ],
    }), encoding="utf-8")
    (tmp_path / "scorecard.json").write_text(json.dumps({
        "candidate_name": "Sieda Khan", "role": "Junior AI Engineer",
        "interview_date": "2026-08-14", "duration_seconds": 2,
        "competencies": [], "overall_score": 0.0, "recommendation": "borderline",
        "recommendation_reasoning": "", "strengths": [], "concerns": [],
        "guardrail_flags": [], "github_grounded_questions_asked": 0,
    }), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_all_five_tools_are_registered():
    for name in ["get_candidate", "get_question_plan", "save_score",
                 "get_scorecard", "list_interviews"]:
        assert await server.mcp.get_tool(name) is not None, name


# ---------------------------------------------------------------------------
# get_candidate
# ---------------------------------------------------------------------------


def test_get_candidate_by_name_returns_structured_match(mcp_fs):
    result = server.get_candidate("Sieda Khan")
    assert result["status"] == "ok"
    c = result["candidate"]
    assert c["candidate_name"] == "Sieda Khan"
    assert c["github_url"] == "https://github.com/siedaa"
    # soft match: "LangChain or LangGraph" is covered by resume skill "LangChain"
    match = c["jd_match"]
    assert "Python" in match["matched_skills"]
    assert "LangChain or LangGraph" in match["matched_skills"]
    assert "REST APIs" in match["missing_skills"]
    assert match["must_have_coverage"] == "4/5 must-have skills present"


def test_get_candidate_unknown_id_returns_not_found(mcp_fs):
    result = server.get_candidate("Someone Else")
    assert result["status"] == "not_found"
    assert "not found" in result["message"].lower()


def test_get_candidate_missing_resume_returns_not_found(mcp_fs, monkeypatch):
    monkeypatch.setattr(server, "RESUME_PATH", mcp_fs / "does-not-exist.json")
    result = server.get_candidate("Sieda Khan")
    assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# get_question_plan
# ---------------------------------------------------------------------------


def test_get_question_plan_returns_structured_plan(mcp_fs):
    result = server.get_question_plan()
    assert result["status"] == "ok"
    assert result["question_count"] == 2
    assert result["approved_by_human"] is True
    assert result["questions"][0]["id"] == "q1"


def test_get_question_plan_missing_returns_not_found(mcp_fs, monkeypatch):
    monkeypatch.setattr(server, "PLAN_PATH", mcp_fs / "nope.json")
    result = server.get_question_plan()
    assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# save_score
# ---------------------------------------------------------------------------


def test_save_score_appends_grounded_entry(mcp_fs):
    result = server.save_score(
        "Technical depth", 4,
        "I built a retrieval-augmented generation system using Chroma as the vector store",
        "Strong concrete answer",
    )
    assert result["status"] == "ok"
    assert result["saved"]["name"] == "Technical depth"
    assert result["competencies_count"] == 1
    assert result["overall_score"] == 4.0
    # persisted
    sc = json.loads(server.SCORECARD_PATH.read_text(encoding="utf-8"))
    assert len(sc["competencies"]) == 1
    assert sc["competencies"][0]["evidence_quote"] == (
        "I built a retrieval-augmented generation system using Chroma as the vector store"
    )


def test_save_score_updates_existing_entry_instead_of_duplicating(mcp_fs):
    server.save_score("Technical depth", 3, "I built a retrieval-augmented generation system",
                      "first pass")
    result = server.save_score("Technical depth", 5, "I built a retrieval-augmented generation system",
                               "revised")
    assert result["status"] == "ok"
    assert result["competencies_count"] == 1
    assert result["overall_score"] == 5.0


def test_save_score_fabricated_quote_is_rejected(mcp_fs):
    result = server.save_score(
        "Debugging", 2,
        "I single-handedly built a distributed database from scratch",
        "invented",
    )
    assert result["status"] == "error"
    assert "rejected" in result["message"]
    assert "does not appear" in result.get("reason", "")
    # nothing was written
    sc = json.loads(server.SCORECARD_PATH.read_text(encoding="utf-8"))
    assert sc["competencies"] == []


def test_save_score_empty_quote_is_rejected(mcp_fs):
    result = server.save_score("Debugging", 2, "   ", "no quote")
    assert result["status"] == "error"
    assert "evidence_quote" in result["message"]


def test_save_score_without_scorecard_returns_not_found(mcp_fs, monkeypatch):
    monkeypatch.setattr(server, "SCORECARD_PATH", mcp_fs / "no-scorecard.json")
    result = server.save_score("Debugging", 2, "some quote", "x")
    assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# get_scorecard
# ---------------------------------------------------------------------------


def test_get_scorecard_returns_full_structured_scorecard(mcp_fs):
    result = server.get_scorecard()
    assert result["status"] == "ok"
    assert result["scorecard"]["candidate_name"] == "Sieda Khan"


def test_get_scorecard_missing_returns_not_found(mcp_fs, monkeypatch):
    monkeypatch.setattr(server, "SCORECARD_PATH", mcp_fs / "no-scorecard.json")
    result = server.get_scorecard()
    assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# list_interviews
# ---------------------------------------------------------------------------


def test_list_interviews_returns_current_interview(mcp_fs):
    result = server.list_interviews()
    assert result["status"] == "ok"
    assert len(result["interviews"]) == 1
    interview = result["interviews"][0]
    assert interview["candidate_name"] == "Sieda Khan"
    assert interview["turns"] == 2
    assert interview["scorecard_exists"] is True


def test_list_interviews_empty_when_no_outputs(mcp_fs, monkeypatch):
    monkeypatch.setattr(server, "TRANSCRIPT_PATH", mcp_fs / "no-transcript.json")
    monkeypatch.setattr(server, "SCORECARD_PATH", mcp_fs / "no-scorecard.json")
    result = server.list_interviews()
    assert result["status"] == "ok"
    assert result["interviews"] == []


# ---------------------------------------------------------------------------
# MCP protocol wrapper
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_tool_roundtrip_through_mcp_call_tool(mcp_fs):
    out = await server.mcp.call_tool("get_question_plan", {})
    assert out.is_error is False
    assert out.structured_content["status"] == "ok"
    assert out.structured_content["question_count"] == 2
