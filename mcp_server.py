"""BudgetWise MCP server (stdio): every tool the two agents can call.

All money math lives here, not in the LLM. Data comes from the FAKE sample files in data/.

Tools
  Facts (Analyst):    get_spending_by_category, get_safe_to_spend, get_income_variability,
                      get_goal_progress, check_last_plan
  Planning (Planner): build_weekly_plan, save_plan
  Knowledge (both):   search_budget_knowledge  -> ChromaDB vector search over knowledge/*.md

agent.py launches this file automatically (one connection per agent, each with its own tool list).
To run it alone: python mcp_server.py
NOTE: never print() in this file - stdout is the MCP protocol channel.
"""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("BudgetWise Tools", log_level="WARNING")  # keep the console quiet

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = ROOT / "knowledge"
CHROMA_DIR = ROOT / "chroma_db"  # local vector store (git-ignored, rebuilt automatically)
MEMORY_FILE = ROOT / "memory" / "plan_history.json"  # saved weekly plans (git-ignored)
SEED_FILE = DATA_DIR / "plan_history_seed.json"
KNOWLEDGE_COLLECTION = "budget_knowledge"

# Fixed costs the user cannot easily cut; used to find the top *discretionary* category.
FIXED_CATEGORIES = {"rent", "utilities", "debt_payment"}


# --------------------------------------------------------------------------- data helpers
def _load_profile() -> dict:
    with open(DATA_DIR / "profile.json", encoding="utf-8") as f:
        return json.load(f)


def _load_transactions() -> list[dict]:
    with open(DATA_DIR / "transactions.csv", newline="", encoding="utf-8") as f:
        return [
            {
                "date": date.fromisoformat(r["date"]),
                "description": r["description"],
                "category": r["category"],
                "amount": float(r["amount"]),
            }
            for r in csv.DictReader(f)
        ]


def _as_of() -> date:
    return date.fromisoformat(_load_profile()["as_of"])


def _spending_between(start: date, end: date) -> dict[str, float]:
    """Total spending per category for transactions in the window (start, end]."""
    totals: dict[str, float] = defaultdict(float)
    for t in _load_transactions():
        if start < t["date"] <= end and t["amount"] < 0:
            totals[t["category"]] += -t["amount"]
    return totals


# --------------------------------------------------------------------------- Tool 1
@mcp.tool()
def get_spending_by_category(days: int = 30) -> dict:
    """Get total spending per category over the last N days (default 30).
    Returns category totals, share of spending, the top category overall, and the
    top discretionary category (excludes rent, utilities, debt payments)."""
    if not isinstance(days, int) or days < 1 or days > 365:
        return {"error": "days must be a whole number between 1 and 365"}
    end = _as_of()
    start = end - timedelta(days=days)  # window is (start, end]
    totals = _spending_between(start, end)
    total_spent = sum(totals.values())
    by_category = [
        {
            "category": c,
            "total": round(v, 2),
            "share_pct": round(100 * v / total_spent, 1) if total_spent else 0.0,
        }
        for c, v in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    ]
    discretionary = [c for c in by_category if c["category"] not in FIXED_CATEGORIES]
    return {
        "window_start_exclusive": start.isoformat(),
        "window_end": end.isoformat(),
        "days": days,
        "total_spent": round(total_spent, 2),
        "by_category": by_category,
        "top_category": by_category[0]["category"] if by_category else None,
        "top_discretionary_category": discretionary[0]["category"] if discretionary else None,
    }


# --------------------------------------------------------------------------- Tool 2
@mcp.tool()
def get_safe_to_spend() -> dict:
    """Calculate how much money is safe to spend before the next paycheck:
    balance minus bills due before payday minus the savings buffer, in total and per day."""
    p = _load_profile()
    as_of = date.fromisoformat(p["as_of"])
    payday = date.fromisoformat(p["next_paycheck_date"])
    days_left = max((payday - as_of).days, 1)
    bills = [
        b for b in p["upcoming_bills"]
        if as_of <= date.fromisoformat(b["due_date"]) <= payday
    ]
    bills_total = round(sum(b["amount"] for b in bills), 2)
    safe = round(p["current_balance"] - bills_total - p["savings_buffer"], 2)
    return {
        "as_of": as_of.isoformat(),
        "next_paycheck_date": payday.isoformat(),
        "days_until_paycheck": days_left,
        "current_balance": p["current_balance"],
        "bills_due_before_paycheck": bills,
        "bills_total": bills_total,
        "savings_buffer": p["savings_buffer"],
        "safe_to_spend_total": safe,
        "safe_to_spend_per_day": round(safe / days_left, 2),
        "warning": "Bills and buffer already exceed your balance." if safe < 0 else None,
    }


# --------------------------------------------------------------------------- Tool 3
@mcp.tool()
def get_income_variability(months: int = 3) -> dict:
    """Summarize how much monthly income varies over the last N complete months
    (default 3): per-month income, average, lowest, highest, and a conservative baseline."""
    if not isinstance(months, int) or months < 1 or months > 12:
        return {"error": "months must be a whole number between 1 and 12"}
    as_of = _as_of()
    y, m = as_of.year, as_of.month
    wanted = []  # the complete calendar months before the as_of month
    for _ in range(months):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        wanted.append((y, m))
    income: dict[tuple[int, int], float] = defaultdict(float)
    for t in _load_transactions():
        key = (t["date"].year, t["date"].month)
        if key in wanted and t["amount"] > 0:
            income[key] += t["amount"]
    per_month = [
        {"month": f"{y}-{m:02d}", "income": round(income.get((y, m), 0.0), 2)}
        for (y, m) in sorted(wanted)
    ]
    values = [x["income"] for x in per_month]
    lowest, highest = min(values), max(values)
    return {
        "months_analyzed": months,
        "per_month": per_month,
        "average_income": round(sum(values) / len(values), 2),
        "lowest_month_income": lowest,
        "highest_month_income": highest,
        "swing": round(highest - lowest, 2),
        "conservative_baseline": lowest,
        "note": "Plan fixed costs against conservative_baseline (the lowest month).",
    }


# --------------------------------------------------------------------------- Tool 4 (new in M3)
def _weekly_safe_to_spend() -> float:
    """Safe-to-spend before payday, scaled to one 7-day week."""
    s = get_safe_to_spend()
    return round(s["safe_to_spend_total"] * 7 / s["days_until_paycheck"], 2)


@mcp.tool()
def get_goal_progress() -> dict:
    """Show progress toward the user's savings goal: amount saved, amount left, weeks left
    until the deadline, how much must be saved per week to make it, and whether that
    weekly amount fits inside this week's safe-to-spend money."""
    g = _load_profile()["savings_goal"]
    as_of = _as_of()
    deadline = date.fromisoformat(g["deadline"])
    days_left = max((deadline - as_of).days, 1)
    remaining = round(max(g["target"] - g["saved_so_far"], 0.0), 2)
    needed_per_week = round(remaining * 7 / days_left, 2)
    weekly_safe = _weekly_safe_to_spend()
    return {
        "goal_name": g["name"],
        "target": g["target"],
        "saved_so_far": g["saved_so_far"],
        "remaining": remaining,
        "percent_complete": round(100 * g["saved_so_far"] / g["target"], 1) if g["target"] else 0.0,
        "deadline": g["deadline"],
        "days_left": days_left,
        "weeks_left": round(days_left / 7, 1),
        "needed_per_week": needed_per_week,
        "weekly_safe_to_spend": weekly_safe,
        "fits_this_week": needed_per_week <= weekly_safe,
    }


# --------------------------------------------------------------------------- Tool 5 (new in M3)
@mcp.tool()
def build_weekly_plan(save_this_week: float, focus_category: str, cut_percent: int = 20) -> dict:
    """Do the math for a 7-day spending plan. The Planner chooses the strategy; this tool
    calculates every dollar figure.
    save_this_week: dollars to move to the savings goal this week (0 or more).
    focus_category: one discretionary category to cap this week (e.g. "dining").
    cut_percent: how much to cut that category below its recent weekly average (0-50).
    Returns the weekly spending allowance, daily limit, focus-category cap, money freed by
    the cut, and whether the savings amount keeps the goal on track."""
    as_of = _as_of()
    weekly_safe = _weekly_safe_to_spend()
    if weekly_safe <= 0:
        return {"error": "Nothing is safe to spend before payday, so no weekly plan can be built. "
                         "Focus on covering bills first."}
    try:
        save_this_week = round(float(save_this_week), 2)
        cut_percent = int(cut_percent)
    except (TypeError, ValueError):
        return {"error": "save_this_week must be a number and cut_percent a whole number"}
    if save_this_week < 0 or save_this_week > weekly_safe:
        return {"error": f"save_this_week must be between 0 and {weekly_safe} "
                         f"(this week's safe-to-spend amount)"}
    if cut_percent < 0 or cut_percent > 50:
        return {"error": "cut_percent must be between 0 and 50"}

    last_30 = _spending_between(as_of - timedelta(days=30), as_of)
    focus_category = str(focus_category).strip().lower()
    discretionary = sorted(c for c in last_30 if c not in FIXED_CATEGORIES)
    if focus_category not in discretionary:
        return {"error": f"focus_category must be one of the discretionary categories: {', '.join(discretionary)}"}

    spending_allowance = round(weekly_safe - save_this_week, 2)
    daily_limit = round(spending_allowance / 7, 2)
    focus_weekly_average = round(last_30[focus_category] * 7 / 30, 2)
    focus_cap = round(focus_weekly_average * (100 - cut_percent) / 100, 2)
    goal = get_goal_progress()
    return {
        "week_start": as_of.isoformat(),
        "week_end": (as_of + timedelta(days=6)).isoformat(),
        "weekly_safe_to_spend": weekly_safe,
        "save_this_week": save_this_week,
        "weekly_spending_allowance": spending_allowance,
        "daily_limit": daily_limit,
        "focus_category": focus_category,
        "focus_category_weekly_average": focus_weekly_average,
        "cut_percent": cut_percent,
        "focus_category_cap": focus_cap,
        "freed_up_by_cut": round(focus_weekly_average - focus_cap, 2),
        "goal_name": goal["goal_name"],
        "goal_needed_per_week": goal["needed_per_week"],
        "on_track_for_goal": save_this_week + 0.005 >= goal["needed_per_week"],
        "how_calculated": [
            "weekly_safe_to_spend = safe_to_spend_total x 7 / days_until_paycheck",
            "weekly_spending_allowance = weekly_safe_to_spend - save_this_week",
            "daily_limit = weekly_spending_allowance / 7",
            "focus_category_weekly_average = last 30 days in that category x 7 / 30",
            "focus_category_cap = weekly average x (100 - cut_percent) / 100",
        ],
    }


# --------------------------------------------------------------------------- memory (new in M3)
def _load_memory() -> dict:
    """Saved weekly plans. Starts from the sample plan in data/plan_history_seed.json."""
    if not MEMORY_FILE.exists():
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        seed = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        MEMORY_FILE.write_text(json.dumps({"plans": seed["plans"]}, indent=2), encoding="utf-8")
    return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))


def reset_memory() -> None:
    """Forget all saved plans (the sample plan comes back on the next run)."""
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()


# --------------------------------------------------------------------------- Tool 6 (new in M3)
@mcp.tool()
def save_plan(daily_limit: float, focus_category: str, focus_category_cap: float,
              save_this_week: float, summary: str) -> dict:
    """Remember this week's plan so next week's check-in can compare against it.
    Pass the numbers exactly as build_weekly_plan returned them, plus a one-sentence summary."""
    try:
        plan = {
            "week_start": _as_of().isoformat(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "daily_limit": round(float(daily_limit), 2),
            "focus_category": str(focus_category).strip().lower(),
            "focus_category_cap": round(float(focus_category_cap), 2),
            "save_this_week": round(float(save_this_week), 2),
            "summary": str(summary)[:300],
        }
    except (TypeError, ValueError):
        return {"error": "daily_limit, focus_category_cap and save_this_week must be numbers"}
    memory = _load_memory()
    memory["plans"].append(plan)
    MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")
    return {"saved": True, "week_start": plan["week_start"], "plans_in_memory": len(memory["plans"])}


# --------------------------------------------------------------------------- Tool 7 (new in M3)
@mcp.tool()
def check_last_plan() -> dict:
    """Compare the most recent earlier weekly plan (from memory) with what was actually spent
    since it started: flexible spending per day vs the daily limit, and focus-category
    spending vs its cap."""
    as_of = _as_of()
    earlier = [p for p in _load_memory()["plans"] if date.fromisoformat(p["week_start"]) < as_of]
    if not earlier:
        return {"has_previous_plan": False, "note": "No earlier plan in memory yet."}
    plan = max(earlier, key=lambda p: (p["week_start"], p["saved_at"]))
    start = date.fromisoformat(plan["week_start"])
    days = (as_of - start).days
    spent = _spending_between(start, as_of)
    flexible_total = round(sum(v for c, v in spent.items() if c not in FIXED_CATEGORIES), 2)
    per_day = round(flexible_total / days, 2)
    focus_spent = round(spent.get(plan["focus_category"], 0.0), 2)
    day_diff = round(per_day - plan["daily_limit"], 2)
    focus_diff = round(focus_spent - plan["focus_category_cap"], 2)
    return {
        "has_previous_plan": True,
        "plan_week_start": plan["week_start"],
        "days_checked": days,
        "plan_summary": plan["summary"],
        "daily_limit": plan["daily_limit"],
        "actual_flexible_spending": flexible_total,
        "actual_spending_per_day": per_day,
        "daily_limit_result": (f"over by {abs(day_diff):.2f} per day" if day_diff > 0
                               else f"under by {abs(day_diff):.2f} per day"),
        "focus_category": plan["focus_category"],
        "focus_category_cap": plan["focus_category_cap"],
        "actual_focus_spent": focus_spent,
        "focus_result": (f"over the cap by {abs(focus_diff):.2f}" if focus_diff > 0
                         else f"under the cap by {abs(focus_diff):.2f}"),
        "stuck_to_plan": day_diff <= 0 and focus_diff <= 0,
        "planned_savings": plan["save_this_week"],
    }


# --------------------------------------------------------------------------- knowledge layer (new in M3)
def _knowledge_chunks() -> list[dict]:
    """Split each knowledge/*.md file into one chunk per '## ' section."""
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        title, section, body, sources = path.stem, None, [], ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                if section:
                    chunks.append({"source": path.name, "title": title, "section": section, "body": body})
                section, body = line[3:].strip(), []
            elif line.startswith("Sources:"):
                sources = line[len("Sources:"):].strip()
            elif line.strip():
                body.append(line.strip())
        if section:
            chunks.append({"source": path.name, "title": title, "section": section, "body": body})
        for c in chunks:
            if c["source"] == path.name:
                c["sources"] = sources
    for c in chunks:
        c["text"] = f"{c['title']} - {c['section']}: " + " ".join(c["body"])
    return chunks


def _knowledge_collection():
    """Open the ChromaDB collection, (re)building it if the knowledge files changed."""
    import chromadb  # imported lazily so the other tools start fast
    from chromadb.config import Settings

    chunks = _knowledge_chunks()
    fingerprint = hashlib.sha256("".join(c["text"] + c["sources"] for c in chunks).encode()).hexdigest()[:16]
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    try:
        col = client.get_collection(KNOWLEDGE_COLLECTION)
        if (col.metadata or {}).get("fingerprint") == fingerprint and col.count() == len(chunks):
            return col
        client.delete_collection(KNOWLEDGE_COLLECTION)
    except Exception:  # collection does not exist yet
        pass
    col = client.create_collection(KNOWLEDGE_COLLECTION, metadata={"fingerprint": fingerprint})
    col.add(
        ids=[f"{c['source']}#{i}" for i, c in enumerate(chunks)],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "section": c["section"], "sources": c["sources"]} for c in chunks],
    )
    return col


def build_knowledge_index() -> int:
    """Build the vector index ahead of time (agent.py calls this before starting the server,
    so the one-time embedding-model download never happens inside an MCP call)."""
    return _knowledge_collection().count()


@mcp.tool()
def search_budget_knowledge(query: str, top_k: int = 3) -> dict:
    """Search the BudgetWise knowledge base (budgeting guidance from Penn State Extension and
    the CFPB, stored in a ChromaDB vector database) for passages that match the query.
    Use it for strategy and advice, e.g. "irregular income baseline" or "cut dining out"."""
    if not str(query).strip():
        return {"error": "query must not be empty"}
    top_k = max(1, min(int(top_k), 5))
    col = _knowledge_collection()
    res = col.query(query_texts=[str(query)], n_results=min(top_k, col.count()))
    results = [
        {
            "source_file": meta["source"],
            "section": meta["section"],
            "text": doc,
            "original_sources": meta["sources"],
            "distance": round(dist, 3),
        }
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    return {"query": query, "results": results}


if __name__ == "__main__":
    mcp.run()  # stdio transport
