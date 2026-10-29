"""BudgetWise tool tests - no LLM and no network needed (except a one-time embedding-model
download for the knowledge-base tests). Run with:  python -m pytest

Part 1 calls the tool functions directly; part 2 starts the real MCP server and calls them through MCP.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcp_server as tools  # noqa: E402


@pytest.fixture(autouse=True)
def temp_memory(tmp_path, monkeypatch):
    """Every test gets its own plan memory file, starting from the sample plan."""
    monkeypatch.setattr(tools, "MEMORY_FILE", tmp_path / "plan_history.json")


# ---- Part 1a: M2 tools (unchanged) ----
def test_spending_by_category_top_discretionary_is_dining():
    r = tools.get_spending_by_category(30)
    assert r["top_category"] == "rent"
    assert r["top_discretionary_category"] == "dining"
    dining = next(c for c in r["by_category"] if c["category"] == "dining")
    assert dining["total"] == 359.10
    assert r["total_spent"] == 2485.70


def test_spending_by_category_seven_day_window():
    r = tools.get_spending_by_category(7)
    assert r["days"] == 7
    assert r["total_spent"] == 146.61
    assert r["top_category"] == "groceries"
    assert r["top_discretionary_category"] == "groceries"


def test_spending_rejects_bad_input():
    assert "error" in tools.get_spending_by_category(0)
    assert "error" in tools.get_spending_by_category(9999)


def test_safe_to_spend():
    r = tools.get_safe_to_spend()
    assert r["days_until_paycheck"] == 11
    assert r["bills_total"] == 1245.00  # internet (due after payday) is excluded
    assert r["safe_to_spend_total"] == 430.25
    assert r["safe_to_spend_per_day"] == 39.11
    assert r["warning"] is None


def test_income_variability():
    r = tools.get_income_variability(3)
    assert [m["income"] for m in r["per_month"]] == [2850.00, 3420.00, 2610.00]
    assert r["average_income"] == 2960.00
    assert r["lowest_month_income"] == 2610.00
    assert r["highest_month_income"] == 3420.00
    assert r["conservative_baseline"] == 2610.00


def test_income_rejects_bad_input():
    assert "error" in tools.get_income_variability(0)


# ---- Part 1b: M3 tools ----
def test_goal_progress():
    r = tools.get_goal_progress()
    assert r["goal_name"] == "Emergency fund"
    assert r["remaining"] == 380.00
    assert r["days_left"] == 64
    assert r["needed_per_week"] == 41.56  # 380 x 7 / 64
    assert r["weekly_safe_to_spend"] == 273.80  # 430.25 x 7 / 11
    assert r["fits_this_week"] is True


def test_build_weekly_plan_math():
    r = tools.build_weekly_plan(41.56, "dining", 25)
    assert r["weekly_safe_to_spend"] == 273.80
    assert r["weekly_spending_allowance"] == 232.24
    assert r["daily_limit"] == 33.18
    assert r["focus_category_weekly_average"] == 83.79  # 359.10 x 7 / 30
    assert r["focus_category_cap"] == 62.84
    assert r["freed_up_by_cut"] == 20.95
    assert r["on_track_for_goal"] is True


def test_build_weekly_plan_flags_goal_not_on_track():
    assert tools.build_weekly_plan(20, "dining", 10)["on_track_for_goal"] is False


def test_build_weekly_plan_rejects_bad_input():
    assert "error" in tools.build_weekly_plan(
        999, "dining", 25
    )  # more than is safe to spend
    assert "error" in tools.build_weekly_plan(-5, "dining", 25)
    assert "error" in tools.build_weekly_plan(
        40, "rent", 25
    )  # fixed cost, not discretionary
    assert "error" in tools.build_weekly_plan(40, "dining", 80)  # cut too large


def test_check_last_plan_uses_sample_plan():
    r = tools.check_last_plan()
    assert r["has_previous_plan"] is True
    assert r["plan_week_start"] == "2026-09-21"
    assert r["actual_spending_per_day"] == 20.94  # 146.61 / 7
    assert r["daily_limit_result"] == "under by 14.06 per day"
    assert r["actual_focus_spent"] == 65.38  # dining 9/23 + 9/26
    assert r["focus_result"] == "over the cap by 5.38"
    assert r["stuck_to_plan"] is False


def test_save_plan_is_remembered_but_not_compared_to_itself():
    saved = tools.save_plan(33.18, "dining", 62.84, 41.56, "test plan")
    assert saved["saved"] is True and saved["plans_in_memory"] == 2
    # A plan saved "today" is next week's baseline; today's check still compares the earlier plan.
    assert tools.check_last_plan()["plan_week_start"] == "2026-09-21"


def test_reset_memory_restores_sample_plan():
    tools.save_plan(33.18, "dining", 62.84, 41.56, "test plan")
    tools.reset_memory()
    assert len(tools._load_memory()["plans"]) == 1


def test_knowledge_chunks_cover_all_documents():
    chunks = tools._knowledge_chunks()
    assert len({c["source"] for c in chunks}) == 5
    assert all(c["sources"] for c in chunks)  # every passage keeps its citation


def test_knowledge_search_finds_relevant_guidance():
    r = tools.search_budget_knowledge(
        "how do I budget when my income changes every month", 3
    )
    files = [x["source_file"] for x in r["results"]]
    assert "irregular_income_baseline.md" in files or "weekly_spending_plan.md" in files
    r = tools.search_budget_knowledge("emergency fund size", 2)
    assert r["results"][0]["source_file"] == "emergency_fund_and_buffer.md"


def test_knowledge_search_rejects_empty_query():
    assert "error" in tools.search_budget_knowledge("   ")


# ---- Part 2: through the real MCP protocol ----
SERVER = Path(__file__).resolve().parent.parent / "mcp_server.py"
ALL_TOOLS = sorted(
    [
        "get_spending_by_category",
        "get_safe_to_spend",
        "get_income_variability",
        "get_goal_progress",
        "build_weekly_plan",
        "save_plan",
        "check_last_plan",
        "search_budget_knowledge",
    ]
)


async def _call(tool: str, args: dict | None = None):
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = [t.name for t in (await session.list_tools()).tools]
            res = await session.call_tool(tool, args or {})
            return listed, json.loads(res.content[0].text)


def test_server_lists_all_tools():
    listed, _ = asyncio.run(_call("get_safe_to_spend"))
    assert sorted(listed) == ALL_TOOLS


def test_mcp_safe_to_spend_executes():
    _, data = asyncio.run(_call("get_safe_to_spend"))
    assert data["safe_to_spend_per_day"] == 39.11


def test_mcp_spending_executes():
    _, data = asyncio.run(_call("get_spending_by_category", {"days": 30}))
    assert data["top_discretionary_category"] == "dining"


def test_mcp_income_executes():
    _, data = asyncio.run(_call("get_income_variability", {"months": 3}))
    assert data["conservative_baseline"] == 2610.00


def test_mcp_build_weekly_plan_executes():
    _, data = asyncio.run(
        _call(
            "build_weekly_plan",
            {"save_this_week": 41.56, "focus_category": "dining", "cut_percent": 25},
        )
    )
    assert data["daily_limit"] == 33.18


def test_mcp_knowledge_search_executes():
    tools.build_knowledge_index()  # make sure the index exists before the server process uses it
    _, data = asyncio.run(
        _call("search_budget_knowledge", {"query": "cut eating out", "top_k": 2})
    )
    assert len(data["results"]) == 2
