"""BudgetWise AI - multi-agent personal budgeting assistant (CEN 4930 M3).

Two agents on NRP gpt-oss, both using MCP tools from mcp_server.py:
  1. Analyst  - finds the facts with tools using ReAct (Thought -> Action -> Observation).
                Answers fact questions itself.
  2. Planner  - receives a FinancialSnapshot through an explicit handoff when the user asks for a
                plan, then builds this week's plan with chain-of-thought, the ChromaDB knowledge
                base, and plan memory.
Guardrails: input guardrail on the Analyst (scope + sensitive data), handoff payload check, and a
grounding output guardrail on both agents (every $ amount must come from a tool result).

Usage:
    python agent.py "How am I doing this month, and what should I do this week?"
    python agent.py "How much can I safely spend per day until payday?"
    python agent.py --trace "..."      # also print SDK trace spans (model + tool calls)
    python agent.py --quiet "..."      # hide the live step log, show only the answer
    python agent.py --reset-memory     # forget saved plans (the sample plan comes back)
    python agent.py                    # chat mode ('quit' exits)
"""

import ast
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    RunHooks,
    Runner,
    handoff,
    input_guardrail,
    output_guardrail,
    set_default_openai_api,
    set_default_openai_client,
    set_trace_processors,
    set_tracing_disabled,
)
from agents.extensions.handoff_filters import remove_all_tools
from agents.mcp import MCPServerStdio, MCPServerStdioParams, create_static_tool_filter
from agents.tracing import TracingProcessor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

import mcp_server
from guardrails import check_payload, check_user_input, find_ungrounded_amounts

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Model backend (kept separate from the agent logic so it can be swapped) ---
MODEL = os.getenv("NRP_MODEL", "gpt-oss")
SERVER_SCRIPT = ROOT / "mcp_server.py"
MAX_TURNS = 20

ANALYST_TOOLS = [
    "get_spending_by_category",
    "get_safe_to_spend",
    "get_income_variability",
    "get_goal_progress",
    "check_last_plan",
    "search_budget_knowledge",
]
PLANNER_TOOLS = ["search_budget_knowledge", "build_weekly_plan", "save_plan"]


# ============================================================================ handoff payload
class FinancialSnapshot(BaseModel):
    """What the Analyst passes to the Planner. Every number must be copied from a tool result."""

    user_request: str = Field(description="The user's question, word for word.")
    safe_to_spend_total: float = Field(
        description="get_safe_to_spend.safe_to_spend_total"
    )
    safe_to_spend_per_day: float = Field(
        description="get_safe_to_spend.safe_to_spend_per_day"
    )
    days_until_paycheck: int = Field(
        description="get_safe_to_spend.days_until_paycheck"
    )
    top_discretionary_category: str = Field(
        description="get_spending_by_category.top_discretionary_category"
    )
    top_discretionary_30_day_total: float = Field(
        description="30-day total of that category"
    )
    conservative_monthly_income: float = Field(
        description="get_income_variability.conservative_baseline"
    )
    goal_name: str = Field(description="get_goal_progress.goal_name")
    goal_remaining: float = Field(description="get_goal_progress.remaining")
    goal_needed_per_week: float = Field(description="get_goal_progress.needed_per_week")
    last_plan_result: str = Field(
        description="One sentence summarizing check_last_plan, "
        "e.g. 'dining was over the 60.00 cap by 5.38'"
    )


# ============================================================================ shared run state
@dataclass
class BudgetWiseState:
    """Run context shared by both agents, the hooks and the guardrails."""

    question: str
    verbose: bool = True
    tool_outputs: list[str] = field(
        default_factory=list
    )  # evidence for the grounding guardrail
    tools_called: list[tuple[str, str]] = field(default_factory=list)  # (agent, tool)
    snapshot: FinancialSnapshot | None = None
    payload_mismatches: list[str] = field(default_factory=list)


def _say(state: BudgetWiseState, text: str) -> None:
    if state.verbose:
        print(text, flush=True)


def _short(result: object, limit: int = 170) -> str:
    """One-line preview of a tool result for the live step log."""
    text = str(result)
    try:
        data = ast.literal_eval(text) if text.startswith("{'") else json.loads(text)
        if (
            isinstance(data, dict) and "text" in data and len(data) <= 3
        ):  # MCP text content wrapper
            data = json.loads(data["text"])
        if isinstance(data, dict):
            if "results" in data:  # knowledge search
                text = "; ".join(
                    f"{r['source_file']} > {r['section']}" for r in data["results"]
                )
            else:
                text = ", ".join(
                    f"{k}={v}"
                    for k, v in data.items()
                    if not isinstance(v, (list, dict)) and v is not None
                )
    except (ValueError, TypeError, KeyError):
        pass
    return text if len(text) <= limit else text[:limit] + " ..."


# ============================================================================ live step log (ReAct trace)
class StepLogger(RunHooks):
    """Prints each agent step so you can watch the agents work (used in the demo recording)."""

    async def on_agent_start(self, context, agent):
        _say(context.context, f"\n[{agent.name}] working...")

    async def on_llm_end(self, context, agent, response):
        for (
            item
        ) in (
            response.output
        ):  # gpt-oss returns its reasoning; show it as the ReAct "Thought"
            if getattr(item, "type", "") == "reasoning":
                parts = [
                    getattr(p, "text", "")
                    for p in (getattr(item, "content", None) or [])
                ]
                parts += [
                    getattr(p, "text", "")
                    for p in (getattr(item, "summary", None) or [])
                ]
                thought = " ".join(p for p in parts if p).strip().replace("\n", " ")
                if thought:
                    _say(
                        context.context,
                        f"  Thought: {thought[:220]}{' ...' if len(thought) > 220 else ''}",
                    )

    async def on_tool_start(self, context, agent, tool):
        args = getattr(context, "tool_arguments", "") or ""
        context.context.tools_called.append((agent.name, tool.name))
        _say(context.context, f"  Action: {tool.name}({args})")

    async def on_tool_end(self, context, agent, tool, result):
        context.context.tool_outputs.append(str(result))
        _say(context.context, f"  Observation: {_short(result)}")

    async def on_handoff(self, context, from_agent, to_agent):
        _say(context.context, f"\n[handoff] {from_agent.name} -> {to_agent.name}")


async def on_handoff_to_planner(
    ctx: RunContextWrapper[BudgetWiseState], snapshot: FinancialSnapshot
) -> None:
    """Runs when the Analyst hands off. Stores the payload and checks its numbers against the tools."""
    state = ctx.context
    state.snapshot = snapshot
    state.payload_mismatches = check_payload(snapshot.model_dump(), state.tool_outputs)
    _say(state, "  Handoff payload (FinancialSnapshot):")
    for key, value in snapshot.model_dump().items():
        _say(state, f"    {key}: {value}")
    if state.payload_mismatches:
        _say(
            state,
            f"  Payload check: WARNING - not found in tool results: {', '.join(state.payload_mismatches)}",
        )
    else:
        _say(state, "  Payload check: every number matches a tool result.")


# ============================================================================ guardrails
@input_guardrail(name="scope_and_privacy", run_in_parallel=False)
async def scope_and_privacy_guardrail(
    ctx: RunContextWrapper[BudgetWiseState], agent, user_input
):
    """Blocks investing/crypto/tax/retirement requests and card/account numbers before any model call."""
    text = (
        user_input
        if isinstance(user_input, str)
        else json.dumps(user_input, default=str)
    )
    check = check_user_input(text)
    return GuardrailFunctionOutput(output_info=check, tripwire_triggered=check.blocked)


@output_guardrail(name="grounded_numbers")
async def grounded_numbers_guardrail(
    ctx: RunContextWrapper[BudgetWiseState], agent, output
):
    """Every dollar amount in the final answer must appear in a tool result (or the user's question)."""
    state = ctx.context
    bad = find_ungrounded_amounts(str(output), state.tool_outputs + [state.question])
    return GuardrailFunctionOutput(
        output_info={"ungrounded_amounts": bad}, tripwire_triggered=bool(bad)
    )


# ============================================================================ agent instructions
ANALYST_INSTRUCTIONS = """\
You are the BudgetWise Analyst, the first of two agents in a budgeting assistant for students and
working adults with irregular income. Your job is to FIND THE FACTS with tools. Never do money math
yourself and never guess numbers.

Work in a ReAct loop:
  Thought: decide which fact you need next.
  Action: call ONE tool.
  Observation: read the tool result.
  Repeat until you have the facts you need, then finish.

TOOLS
- get_spending_by_category(days) - where money went; top discretionary category.
- get_safe_to_spend() - money safe to spend before payday, total and per day.
- get_income_variability(months) - income swings and the conservative baseline.
- get_goal_progress() - savings goal: amount left and amount needed per week.
- check_last_plan() - how the user did against their last weekly plan (from memory).
- search_budget_knowledge(query) - budgeting guidance from the knowledge base.

FINISH IN EXACTLY ONE OF TWO WAYS
A) FACT question ("how much", "where", "did I", "how close", or a general "how should I handle X"
   budgeting question): answer it yourself. Quote tool numbers exactly as returned, with a $ sign
   and cents ($359.10, never "about $360"). Add one sentence of practical advice. Under 120 words.
   For a general budgeting question, call search_budget_knowledge and summarize what it returned,
   naming the source. Do NOT hand off.
B) PLAN question (the user asks what to do this week, for a plan, or how to reach their goal):
   call get_safe_to_spend, get_spending_by_category(days=30), get_income_variability,
   get_goal_progress and check_last_plan. Then call transfer_to_planner with a FinancialSnapshot
   whose numbers are copied exactly from those tool results. Do not write the plan yourself.

The data is FAKE SAMPLE DATA. You are not a licensed financial advisor.
"""

PLANNER_INSTRUCTIONS = """\
You are the BudgetWise Planner, the second of two agents. The Analyst already gathered the facts
and handed you the FinancialSnapshot below. Your job is to turn it into ONE concrete plan for the
next 7 days. Never calculate numbers yourself; tools do all the math.

Think step by step (chain-of-thought) and follow these steps in order:
Step 1 - Read the snapshot: safe-to-spend per day, the top discretionary category, the savings
         goal, and how the last plan went.
Step 2 - Call search_budget_knowledge at least once with a query that fits the situation
         (for example "cut dining out" or "reach a savings goal with irregular income").
Step 3 - Choose three things and give a short reason for each:
         save_this_week: use goal_needed_per_week, unless the numbers or guidance say it does
                         not fit; then choose a smaller amount.
         focus_category: the top discretionary category (or the category that went over its cap
                         in the last plan).
         cut_percent:    between 10 and 30. Use a bigger cut if that category went over last time.
Step 4 - Call build_weekly_plan(save_this_week, focus_category, cut_percent).
Step 5 - Call save_plan with daily_limit, focus_category, focus_category_cap and save_this_week
         exactly as build_weekly_plan returned them, plus a one-sentence summary.
Step 6 - Write the answer in this format. Quote every number exactly as a tool returned it, with
         a $ sign and cents (e.g. $33.18):

**How you're doing**
- one bullet on safe-to-spend
- one bullet on how the last plan went

**Your plan for this week**
- Daily limit: $...
- Save: $... toward <goal name>
- <Category> cap: $... for the week (a ...% cut)

**Why this plan**
1. ...
2. ...
3. ... (name the knowledge source file you used)

**One thing to do today:** ...

Stay under 220 words. The data is FAKE SAMPLE DATA. You are not a licensed financial advisor.
"""


def planner_instructions(ctx: RunContextWrapper[BudgetWiseState], agent) -> str:
    """The handoff payload is placed directly in the Planner's instructions."""
    snap = ctx.context.snapshot
    payload = snap.model_dump_json(indent=2) if snap else "(no snapshot received)"
    return f"{PLANNER_INSTRUCTIONS}\nFINANCIAL SNAPSHOT (handoff payload from the Analyst):\n{payload}\n"


def build_agents(
    analyst_server: MCPServerStdio, planner_server: MCPServerStdio
) -> Agent:
    """Create both agents and wire the handoff. Returns the Analyst (the entry point)."""
    planner = Agent(
        name="Planner",
        instructions=planner_instructions,
        mcp_servers=[planner_server],
        output_guardrails=[grounded_numbers_guardrail],
        model=MODEL,
    )
    to_planner = handoff(
        agent=planner,
        tool_name_override="transfer_to_planner",
        tool_description_override=(
            "Hand off to the Planner when the user asks for a plan, what to do this week, or how "
            "to reach their goal. Pass a FinancialSnapshot with numbers copied exactly from your "
            "tool results."
        ),
        input_type=FinancialSnapshot,
        on_handoff=on_handoff_to_planner,
        input_filter=remove_all_tools,  # the Planner sees the question + payload, not raw tool chatter
    )
    return Agent(
        name="Analyst",
        instructions=ANALYST_INSTRUCTIONS,
        mcp_servers=[analyst_server],
        handoffs=[to_planner],
        input_guardrails=[scope_and_privacy_guardrail],
        output_guardrails=[grounded_numbers_guardrail],
        model=MODEL,
    )


# ============================================================================ backend + tracing
class ConsoleTracingProcessor(TracingProcessor):
    """Prints the trace and each span (model call, tool call, handoff) to the console."""

    def on_trace_start(self, trace):
        print(f"\n[trace] '{trace.name}' started")

    def on_trace_end(self, trace):
        print(f"[trace] '{trace.name}' finished")

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        data = span.span_data.export()
        line = f"  [span] {data.get('type', 'unknown')}"
        if data.get("name"):
            line += f" - {data['name']}"
        if data.get("usage"):
            line += f" | usage={data['usage']}"
        print(line)

    def shutdown(self):
        pass

    def force_flush(self):
        pass


def configure(trace: bool = False) -> None:
    api_key, base_url = os.getenv("NRP_API_KEY"), os.getenv("NRP_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "NRP_API_KEY and NRP_BASE_URL must be set. Copy .env.example to .env and fill it in."
        )
    set_default_openai_client(
        AsyncOpenAI(base_url=base_url, api_key=api_key), use_for_tracing=False
    )
    set_default_openai_api("chat_completions")
    if trace:
        set_trace_processors([ConsoleTracingProcessor()])
    else:
        set_tracing_disabled(True)  # default SDK tracing would need an OpenAI key


# ============================================================================ running the system
@dataclass
class Answer:
    question: str
    text: str
    answered_by: str  # "Analyst", "Planner", or "Input guardrail" / "Output guardrail"
    handed_off: bool
    tools_called: list[tuple[str, str]]
    guardrail: str  # short status line
    snapshot: FinancialSnapshot | None = None
    retried: bool = False


class BudgetWise:
    """Opens the two MCP connections once and answers questions with the two-agent system.

    async with BudgetWise() as bw:
        answer = await bw.ask("How am I doing this month, and what should I do this week?")
    """

    def __init__(self, verbose: bool = True, trace: bool = False):
        self.verbose, self.trace = verbose, trace

    def _server(self, name: str, tools: list[str]) -> MCPServerStdio:
        return MCPServerStdio(
            name=name,
            params=MCPServerStdioParams(
                command=sys.executable, args=[str(SERVER_SCRIPT)]
            ),
            client_session_timeout_seconds=120,
            cache_tools_list=True,
            tool_filter=create_static_tool_filter(allowed_tool_names=tools),
        )

    async def __aenter__(self):
        configure(self.trace)
        if self.verbose:
            print("Loading knowledge base (ChromaDB)...", end=" ", flush=True)
        chunks = (
            mcp_server.build_knowledge_index()
        )  # one-time embedding download happens here
        if self.verbose:
            print(f"{chunks} passages ready.")
        self._analyst_server = self._server("BudgetWise Data Tools", ANALYST_TOOLS)
        self._planner_server = self._server("BudgetWise Planning Tools", PLANNER_TOOLS)
        await self._analyst_server.__aenter__()
        await self._planner_server.__aenter__()
        self.analyst = build_agents(self._analyst_server, self._planner_server)
        return self

    async def __aexit__(self, *exc):
        await self._planner_server.__aexit__(*exc)
        await self._analyst_server.__aexit__(*exc)

    async def _run(
        self, question: str, model_input: str
    ) -> tuple[BudgetWiseState, object]:
        state = BudgetWiseState(question=question, verbose=self.verbose)
        result = await Runner.run(
            self.analyst,
            model_input,
            context=state,
            hooks=StepLogger(),
            max_turns=MAX_TURNS,
        )
        return state, result

    async def ask(self, question: str) -> Answer:
        state = BudgetWiseState(question=question, verbose=self.verbose)
        retried = False
        try:
            state, result = await self._run(question, question)
        except InputGuardrailTripwireTriggered as e:
            check = e.guardrail_result.output.output_info
            _say(
                state,
                f"\n[guardrail] Input blocked ({check.category}). No model was called.",
            )
            return Answer(
                question,
                check.message,
                "Input guardrail",
                False,
                [],
                f"input BLOCKED ({check.category})",
            )
        except OutputGuardrailTripwireTriggered as e:
            bad = e.guardrail_result.output.output_info["ungrounded_amounts"]
            _say(
                state,
                f"\n[guardrail] Grounding check failed: {', '.join(bad)} did not come from any tool. "
                "Retrying once with a correction note...",
            )
            retried = True
            note = (
                f"{question}\n\n(Grounding check: a previous answer used {', '.join(bad)}, which no tool "
                "returned. Quote tool numbers exactly, with cents.)"
            )
            try:
                state, result = await self._run(question, note)
            except OutputGuardrailTripwireTriggered as e2:
                bad2 = e2.guardrail_result.output.output_info["ungrounded_amounts"]
                return Answer(
                    question,
                    "BudgetWise could not verify every number in its answer, so it is not showing "
                    f"it (unverified: {', '.join(bad2)}). Please ask again.",
                    "Output guardrail",
                    False,
                    [],
                    f"grounding FAILED twice ({', '.join(bad2)})",
                    retried=True,
                )
        answered_by = result.last_agent.name
        status = "input OK | grounding OK" + (" (after 1 retry)" if retried else "")
        if state.payload_mismatches:
            status += f" | payload WARNING: {', '.join(state.payload_mismatches)}"
        return Answer(
            question,
            str(result.final_output),
            answered_by,
            answered_by == "Planner",
            state.tools_called,
            status,
            state.snapshot,
            retried,
        )


def show(answer: Answer) -> None:
    route = "Analyst -> Planner (handoff)" if answer.handed_off else answer.answered_by
    by_agent: dict[str, list[str]] = {}
    for agent_name, tool in answer.tools_called:
        by_agent.setdefault(agent_name, []).append(tool)
    tools = " | ".join(f"{a}: {', '.join(t)}" for a, t in by_agent.items()) or "none"
    print("\n" + "=" * 72)
    print(f"Question:   {answer.question}")
    print(f"Route:      {route}")
    print(f"Tools:      {tools}")
    print(f"Guardrails: {answer.guardrail}")
    print("=" * 72)
    print(answer.text)
    print("=" * 72)


async def main() -> None:
    args = sys.argv[1:]
    if "--reset-memory" in args:
        mcp_server.reset_memory()
        print(
            "Plan memory cleared (the sample plan from data/plan_history_seed.json comes back next run)."
        )
        return
    trace, quiet = "--trace" in args, "--quiet" in args
    args = [a for a in args if a not in {"--trace", "--quiet"}]
    async with BudgetWise(verbose=not quiet, trace=trace) as bw:
        if args:
            show(await bw.ask(" ".join(args)))
            return
        print(
            f"BudgetWise AI (model: {MODEL}). Ask about spending, safe-to-spend, income, your goal,"
        )
        print(
            "or ask for a plan: 'How am I doing this month, and what should I do this week?'  'quit' exits."
        )
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"quit", "exit", ""}:
                break
            show(await bw.ask(question))


if __name__ == "__main__":
    asyncio.run(main())
