"""Tests for the banned-question guardrail (Phase 3a).

Covers: every banned category blocked by the keyword layer, an adversarial
indirect phrasing that only the LLM layer catches, and normal technical
questions passing through unblocked.
"""

from __future__ import annotations

from src.guardrails.banned_questions import (
    _LLMDecision,
    check_question,
    keyword_screen,
)

# One direct question per banned category -> keyword layer must block it.
BANNED_DIRECT_CASES = [
    ("How old are you?", "age"),
    ("What is your gender?", "gender"),
    ("Are you married?", "marital_status"),
    ("What religion do you follow?", "religion"),
    ("What is your nationality?", "nationality"),
    ("Are you pregnant or expecting a child?", "health_pregnancy"),
    ("What was your previous salary?", "salary_history"),
    ("Who did you vote for in the last election?", "politics"),
]


def test_every_banned_category_is_blocked_by_keyword_layer():
    for question, category in BANNED_DIRECT_CASES:
        check = check_question(question, use_llm=False)
        assert check.blocked, f"expected blocked: {question!r}"
        assert check.category == category, (
            f"expected category {category!r}, got {check.category!r} for {question!r}"
        )
        assert check.layer == "keyword", f"expected keyword layer for {question!r}"


def test_keyword_screen_returns_none_for_clean_question():
    assert keyword_screen("Describe your experience with LangGraph.") is None


def test_normal_technical_questions_pass_unblocked():
    clean_questions = [
        "Walk me through how you would design a RAG pipeline.",
        "What is the difference between a list and a generator in Python?",
        "Describe a bug you found via GitHub issues and how you fixed it.",
        "How do you approach writing tests for a new module?",
    ]
    for question in clean_questions:
        check = check_question(question, use_llm=False)
        assert not check.blocked, f"should not block clean question: {question!r}"


def test_adversarial_phrasing_blocked_by_llm_layer():
    """Indirect wording implying marital/parental status without the keyword.

    The keyword layer must miss it (so the short-circuit path is exercised
    correctly) and the LLM layer (mocked here) must catch it.
    """
    adversarial = "How do you balance work with family responsibilities?"

    keyword_result = check_question(adversarial, use_llm=False)
    assert not keyword_result.blocked, "keyword layer should miss this phrasing"

    def mock_llm(question, model):
        assert question == adversarial
        return _LLMDecision(
            blocked=True,
            category="marital_status",
            reason="asks about family responsibilities, implying marital/parental status",
        )

    check = check_question(adversarial, llm_caller=mock_llm)
    assert check.blocked
    assert check.category == "marital_status"
    assert check.layer == "llm"


def test_adversarial_nationality_phrasing_blocked_by_llm_layer():
    adversarial = "Where did you grow up?"

    keyword_result = check_question(adversarial, use_llm=False)
    assert not keyword_result.blocked, "keyword layer should miss this phrasing"

    def mock_llm(question, model):
        return _LLMDecision(blocked=True, category="nationality", reason="asks origin/upbringing")

    check = check_question(adversarial, llm_caller=mock_llm)
    assert check.blocked
    assert check.category == "nationality"
    assert check.layer == "llm"


def test_clean_question_passes_llm_layer_too():
    def mock_llm(question, model):
        return _LLMDecision(blocked=False)

    check = check_question("Describe a RAG project you shipped.", llm_caller=mock_llm)
    assert not check.blocked
    assert check.layer == "" or check.layer == "-"
