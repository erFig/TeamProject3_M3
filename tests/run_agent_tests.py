"""Live end-to-end evaluation: sends the 10 documented test cases to the real two-agent system
(needs NRP access) and writes a results table to tests/results/latest_results.md.

Usage:
    python tests/run_agent_tests.py            # readable pass/fail report + results file
    python tests/run_agent_tests.py --verbose  # also show each agent step (Thought / Action / Observation)
    python tests/run_agent_tests.py --trace    # also show SDK trace spans
    python tests/run_agent_tests.py TC6 TC9    # run only some cases

A case PASSES only if ALL of these hold (see tests/test_cases.md):
  1. Route   - the right agent finished (Analyst, Planner after a handoff, or the input guardrail).
  2. Tools   - every expected MCP tool was called.
  3. Content - every expected figure/keyword appears in the answer (regex, case-insensitive).
  4. Grounded - the grounding output guardrail passed (every $ amount came from a tool).
The plan memory is reset before the run so results are repeatable.
"""

import asyncio
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcp_server  # noqa: E402
from agent import MODEL, BudgetWise  # noqa: E402

CASES = [
    {
        "id": "TC1",
        "input": "Where am I spending the most money lately?",
        "route": "Analyst",
        "tools": ["get_spending_by_category"],
        "expected": "Dining is the top discretionary category at $359.10",
        "patterns": [r"dining", r"359\.10"],
    },
    {
        "id": "TC2",
        "input": "How much can I safely spend per day until my next paycheck?",
        "route": "Analyst",
        "tools": ["get_safe_to_spend"],
        "expected": "$39.11 per day ($430.25 total over 11 days)",
        "patterns": [r"39\.11", r"430\.25"],
    },
    {
        "id": "TC3",
        "input": "How unpredictable is my income, and what should I plan around?",
        "route": "Analyst",
        "tools": ["get_income_variability"],
        "expected": "Plan around the conservative baseline of $2,610 (lowest month)",
        "patterns": [r"2,?610"],
    },
    {
        "id": "TC4",
        "input": "How close am I to my emergency fund goal?",
        "route": "Analyst",
        "tools": ["get_goal_progress"],
        "expected": "$380.00 left; $41.56 per week needed by Dec 1",
        "patterns": [r"380", r"41\.56"],
    },
    {
        "id": "TC5",
        "input": "Did I stick to last week's plan?",
        "route": "Analyst",
        "tools": ["check_last_plan"],
        "expected": "Under the daily limit, but dining went $5.38 over its $60 cap",
        "patterns": [r"dining", r"5\.38"],
    },
    {
        "id": "TC6",
        "input": "How am I doing this month, and what should I do this week?",
        "route": "Planner",
        "tools": [
            "get_safe_to_spend",
            "check_last_plan",
            "search_budget_knowledge",
            "build_weekly_plan",
            "save_plan",
        ],
        "expected": "Handoff to Planner; plan with a daily limit, savings amount and a dining cap",
        "patterns": [r"daily limit", r"dining", r"save"],
    },
    {
        "id": "TC7",
        "input": "I want to reach my emergency fund goal by December. What should I do this week?",
        "route": "Planner",
        "tools": ["get_goal_progress", "build_weekly_plan", "save_plan"],
        "expected": "Handoff to Planner; plan saves $41.56 this week toward the emergency fund",
        "patterns": [r"41\.56", r"emergency fund"],
    },
    {
        "id": "TC8",
        "input": "What's a good way to budget when my paychecks are different every week?",
        "route": "Analyst",
        "tools": ["search_budget_knowledge"],
        "expected": "Guidance from the knowledge base: plan around your lowest (baseline) income",
        "patterns": [r"lowest|baseline|conservative"],
    },
    {
        "id": "TC9",
        "input": "Should I put my paycheck into index funds or Bitcoin?",
        "route": "Input guardrail",
        "tools": [],
        "expected": "Blocked as out of scope (investing); no model call",
        "patterns": [r"does not give"],
    },
    {
        "id": "TC10",
        "input": "My card number is 4111 1111 1111 1111. Can you check my balance?",
        "route": "Input guardrail",
        "tools": [],
        "expected": "Blocked as sensitive data; no model call",
        "patterns": [r"don't share"],
    },
]

LINE = "=" * 72
RESULTS_FILE = Path(__file__).resolve().parent / "results" / "latest_results.md"


def _norm(text: str) -> str:
    return text.replace("$", "").lower()


def _cell(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[:limit] + " ..."


async def main() -> int:
    args = sys.argv[1:]
    verbose, trace = "--verbose" in args, "--trace" in args
    wanted = {a.upper() for a in args if not a.startswith("--")}
    cases = [c for c in CASES if not wanted or c["id"] in wanted]
    mcp_server.reset_memory()

    print(
        f"{LINE}\nBudgetWise AI - live multi-agent evaluation   (model: {MODEL})\n{LINE}"
    )
    rows = []
    async with BudgetWise(verbose=verbose, trace=trace) as bw:
        for case in cases:
            print(f"\n--- {case['id']} " + "-" * 60)
            print(f"Input:     {case['input']}")
            print(f"Expected:  {case['expected']}")
            try:
                ans = await bw.ask(case["input"])
            except Exception as e:  # a crash is a failed case, not a crashed run
                ans = None
                error = f"{type(e).__name__}: {e}"
            if ans is None:
                checks = {
                    "route": False,
                    "tools": False,
                    "content": False,
                    "grounded": False,
                }
                route, called, answer_text, guard = "error", [], error, "n/a"
            else:
                called = [t for _, t in ans.tools_called]
                route = ans.answered_by
                answer_text, guard = ans.text, ans.guardrail
                checks = {
                    "route": route == case["route"],
                    "tools": all(t in called for t in case["tools"]),
                    "content": all(
                        re.search(p, _norm(ans.text)) for p in case["patterns"]
                    ),
                    "grounded": route != "Output guardrail",
                }
            passed = all(checks.values())
            failed = [k for k, ok in checks.items() if not ok]
            print(f"Route:     {route}   (expected {case['route']})")
            print(f"Tools:     {', '.join(called) or 'none'}")
            print(f"Guardrail: {guard}")
            print("Answer:")
            print(textwrap.indent(answer_text, "    "))
            print(f"RESULT: {'PASS' if passed else 'FAIL - ' + ', '.join(failed)}")
            rows.append(
                {
                    **case,
                    "route_actual": route,
                    "called": called,
                    "answer": answer_text,
                    "guard": guard,
                    "passed": passed,
                    "failed": failed,
                }
            )

    n_pass = sum(r["passed"] for r in rows)
    rate = 100 * n_pass / len(rows)
    print(f"\n{LINE}\nSUMMARY\n{LINE}")
    for r in rows:
        print(
            f"  {r['id']:<5} {'PASS' if r['passed'] else 'FAIL'}   route={r['route_actual']:<16} "
            f"{'' if r['passed'] else 'failed: ' + ', '.join(r['failed'])}"
        )
    print(f"\n{n_pass}/{len(rows)} test cases passed ({rate:.0f}%).")

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    lines = [
        "# BudgetWise AI - live evaluation results",
        "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M} | Model: {MODEL} | Passed: {n_pass}/{len(rows)} ({rate:.0f}%)",
        "",
        "| ID | Input | Expected | Expected route | Actual route | Tools called | Guardrails | Answer (excerpt) | Result |",
        "|----|-------|----------|----------------|--------------|--------------|------------|------------------|--------|",
    ]
    for r in rows:
        result = "PASS" if r["passed"] else "FAIL (" + ", ".join(r["failed"]) + ")"
        lines.append(
            f"| {r['id']} | {_cell(r['input'])} | {_cell(r['expected'])} | {r['route']} | "
            f"{r['route_actual']} | {_cell(', '.join(r['called']) or 'none')} | {_cell(r['guard'])} | "
            f"{_cell(r['answer'])} | {result} |"
        )
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Results table written to {RESULTS_FILE.relative_to(Path.cwd()) if RESULTS_FILE.is_relative_to(Path.cwd()) else RESULTS_FILE}"
    )
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
