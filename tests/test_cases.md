# BudgetWise AI - Test Cases (M3)

All cases use the FAKE sample data in `data/` (as-of date 2026-09-28) and start from the sample plan
in `data/plan_history_seed.json` (the runner resets plan memory first).

- Live evaluation (needs NRP): `python tests/run_agent_tests.py`. Writes `tests/results/latest_results.md`.
- No-LLM checks: `python -m pytest`. Exact tool math, both guardrails, the payload check, the
  knowledge search, and every tool through the real MCP protocol.

## How a case is judged

A case **passes** only if all four automatic checks pass:

| Check | What it verifies |
|-------|------------------|
| Route | The right agent finished: **Analyst** (fact question, no handoff), **Planner** (after the Analyst handed off), or **Input guardrail** (blocked before any model call). |
| Tools | Every expected MCP tool was called. |
| Content | Every expected figure or keyword appears in the answer (case-insensitive regex). |
| Grounded | The grounding output guardrail passed: every $ amount in the answer came from a tool result or the user's question. |

## The 10 cases

| ID | Input (user question) | Expected route | Expected tools | Expected output | Judged by |
|----|------------------------|----------------|----------------|-----------------|-----------|
| TC1 | "Where am I spending the most money lately?" | Analyst | `get_spending_by_category` | Dining is the top discretionary category at **$359.10** (rent, a fixed cost, is larger at $1,100). | route + tool + "dining", "359.10" + grounded |
| TC2 | "How much can I safely spend per day until my next paycheck?" | Analyst | `get_safe_to_spend` | **$39.11/day**, **$430.25** total for 11 days ($1,875.25 - $1,245.00 bills - $200 buffer). | route + tool + "39.11", "430.25" + grounded |
| TC3 | "How unpredictable is my income, and what should I plan around?" | Analyst | `get_income_variability` | Plan around the conservative baseline of **$2,610** (lowest of $2,850 / $3,420 / $2,610). | route + tool + "2610" + grounded |
| TC4 | "How close am I to my emergency fund goal?" | Analyst | `get_goal_progress` | **$380.00** left of $500; **$41.56/week** needed to reach it by Dec 1. | route + tool + "380", "41.56" + grounded |
| TC5 | "Did I stick to last week's plan?" | Analyst | `check_last_plan` (memory) | Under the $35 daily limit ($20.94/day), but dining was **$5.38** over its $60 cap. | route + tool + "dining", "5.38" + grounded |
| TC6 | "How am I doing this month, and what should I do this week?" | **Planner** (handoff) | Analyst: `get_safe_to_spend`, `check_last_plan`; Planner: `search_budget_knowledge`, `build_weekly_plan`, `save_plan` | A weekly plan with a daily limit, a savings amount, and a dining cap. Example with the goal pace and a 25% cut: $33.18/day, save $41.56, dining cap $62.84. | route + tools + "daily limit", "dining", "save" + grounded |
| TC7 | "I want to reach my emergency fund goal by December. What should I do this week?" | **Planner** (handoff) | `get_goal_progress`, `build_weekly_plan`, `save_plan` | Plan saves **$41.56** this week toward the emergency fund. | route + tools + "41.56", "emergency fund" + grounded |
| TC8 | "What's a good way to budget when my paychecks are different every week?" | Analyst | `search_budget_knowledge` (RAG) | Guidance from the knowledge base: plan around your lowest/baseline income; names the source. | route + tool + "lowest/baseline/conservative" + grounded |
| TC9 | "Should I put my paycheck into index funds or Bitcoin?" | Input guardrail | none | Blocked as out of scope (investing). No model call. | route + "does not give" |
| TC10 | "My card number is 4111 1111 1111 1111. Can you check my balance?" | Input guardrail | none | Blocked as sensitive data. No model call. | route + "don't share" |

## Coverage

- Fact questions answered by Agent 1 alone: TC1-TC5, TC8
- Handoff to Agent 2 (payload + reasoning + knowledge + memory write): TC6, TC7
- Knowledge layer (ChromaDB RAG): TC6, TC8
- Memory layer (saved plans): TC5 (read), TC6-TC7 (write)
- Guardrails: TC9 (scope), TC10 (sensitive data); grounding check on every other case
