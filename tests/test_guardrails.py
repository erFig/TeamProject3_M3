"""Guardrail tests - no LLM or network needed. Run with:  python -m pytest"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails import (
    check_payload,
    check_user_input,
    find_ungrounded_amounts,
)  # noqa: E402

TOOL_RESULT = '{"safe_to_spend_total": 430.25, "safe_to_spend_per_day": 39.11, "bills_total": 1245.0}'


# ---- input guardrail ----
def test_normal_budget_questions_pass():
    for q in [
        "How much can I safely spend per day until payday?",
        "How am I doing this month, and what should I do this week?",
        "Where am I spending the most money lately?",
        "How close am I to my emergency fund goal?",
    ]:
        assert check_user_input(q).blocked is False, q


def test_out_of_scope_topics_are_blocked():
    for q, topic in [
        ("Should I buy bitcoin with my paycheck?", "crypto"),
        ("Which index funds should I invest in?", "investing"),
        ("How do I file my taxes this year?", "taxes"),
        ("Should I open a Roth IRA for retirement?", "retirement"),
    ]:
        check = check_user_input(q)
        assert check.blocked and check.category == f"out_of_scope:{topic}", q


def test_sensitive_data_is_blocked():
    for q in [
        "My card number is 4111 1111 1111 1111, check my balance",
        "my account is 123456789012 can you look",
        "SSN 123-45-6789",
        "my password is hunter2",
    ]:
        check = check_user_input(q)
        assert check.blocked and check.category == "sensitive_data", q


def test_dates_and_amounts_are_not_mistaken_for_card_numbers():
    assert not check_user_input("Can I spend $1,875.25 before 2026-10-09?").blocked


# ---- output (grounding) guardrail ----
def test_exact_tool_numbers_are_grounded():
    answer = "You can spend $39.11 per day ($430.25 total) after $1,245.00 in bills."
    assert find_ungrounded_amounts(answer, [TOOL_RESULT]) == []


def test_rounded_or_invented_numbers_are_flagged():
    answer = "You can spend about $39 per day, roughly $430 total, or $500 if you skip a bill."
    assert find_ungrounded_amounts(answer, [TOOL_RESULT]) == ["$39", "$430", "$500"]


def test_amounts_from_the_users_question_are_allowed():
    question = "Can I save $300 by December?"
    assert (
        find_ungrounded_amounts("Saving $300 is possible.", [TOOL_RESULT, question])
        == []
    )


def test_answer_without_dollar_amounts_is_grounded():
    assert (
        find_ungrounded_amounts("Try cooking at home twice this week.", [TOOL_RESULT])
        == []
    )


# ---- handoff payload check ----
def test_payload_numbers_must_match_tools():
    good = {
        "safe_to_spend_total": 430.25,
        "safe_to_spend_per_day": 39.11,
        "goal_name": "Emergency fund",
    }
    bad = {"safe_to_spend_total": 430.0, "safe_to_spend_per_day": 39.11}
    assert check_payload(good, [TOOL_RESULT]) == []
    assert check_payload(bad, [TOOL_RESULT]) == ["safe_to_spend_total=430.0"]
