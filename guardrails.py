"""BudgetWise guardrail checks (plain Python, no LLM).

agent.py wraps these in Agents SDK guardrails:
  - check_user_input        -> input guardrail on the Analyst (runs before any model call)
  - find_ungrounded_amounts -> output guardrail on both agents (every $ amount must come from a tool)
  - check_payload           -> verifies the handoff payload numbers against the tool results

Keeping the checks deterministic makes them free, fast, and easy to unit test (tests/test_guardrails.py).
"""
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# ---------------------------------------------------------------- input guardrail
OUT_OF_SCOPE = {
    "investing": [r"\binvest(ing|ment|ments|or|ors)?\b", r"\bstocks?\b", r"\bindex funds?\b",
                  r"\betfs?\b", r"\bmutual funds?\b", r"\bshares? of\b", r"\boptions trading\b"],
    "crypto": [r"\bcrypto(currency|currencies)?\b", r"\bbitcoin\b", r"\bethereum\b", r"\bdogecoin\b"],
    "taxes": [r"\btax(es)?\b", r"\btax return\b", r"\birs\b", r"\bw-?2\b", r"\b1099\b"],
    "retirement": [r"\bretire(ment)?\b", r"\b401\s?\(?k\)?", r"\broth\b", r"\bira\b", r"\bpension\b"],
}

SENSITIVE = [
    ("card or account number", r"\b(?:\d[ -]?){12,19}\b"),
    ("Social Security number", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("password or PIN", r"\b(password|passcode|pin)\b\s*(is|:|=)"),
]

OUT_OF_SCOPE_MESSAGE = (
    "BudgetWise only helps with day-to-day budgeting: spending, safe-to-spend, income swings, "
    "savings goals and weekly plans. It does not give {topic} advice. For that, please talk to a "
    "licensed professional."
)
SENSITIVE_MESSAGE = (
    "Please don't share your {what}. BudgetWise never needs it, and this prototype only uses the "
    "sample data in data/. Your question was not sent to the AI model."
)


@dataclass
class InputCheck:
    blocked: bool
    category: str = ""
    message: str = ""


def check_user_input(text: str) -> InputCheck:
    """Block sensitive personal data and out-of-scope (investing/crypto/tax/retirement) requests."""
    lowered = text.lower()
    for what, pattern in SENSITIVE:
        if re.search(pattern, lowered):
            return InputCheck(True, "sensitive_data", SENSITIVE_MESSAGE.format(what=what))
    for topic, patterns in OUT_OF_SCOPE.items():
        if any(re.search(p, lowered) for p in patterns):
            return InputCheck(True, f"out_of_scope:{topic}", OUT_OF_SCOPE_MESSAGE.format(topic=topic))
    return InputCheck(False)


# ---------------------------------------------------------------- output (grounding) guardrail
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d{1,2})?")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _to_cents(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def evidence_numbers(evidence: list[str]) -> set[Decimal]:
    """Every number that appears in the evidence (tool results, the user's question)."""
    found = set()
    for text in evidence:
        for raw in _NUMBER.findall(str(text)):
            value = _to_cents(raw)
            if value is not None:
                found.add(value)
    return found


def find_ungrounded_amounts(answer: str, evidence: list[str]) -> list[str]:
    """Return the dollar amounts in the answer that do not appear in any tool result
    (or in the user's own question). An empty list means the answer is grounded.
    Matching is exact to the cent, so "$360" for a tool value of 359.10 is flagged."""
    allowed = evidence_numbers(evidence)
    bad = []
    for whole, cents in _MONEY.findall(answer):
        shown = f"${whole}{cents}"
        value = _to_cents(whole + cents)
        if value is not None and value not in allowed and shown not in bad:
            bad.append(shown)
    return bad


def check_payload(payload: dict, evidence: list[str]) -> list[str]:
    """Return the payload fields whose number does not match any tool result."""
    allowed = evidence_numbers(evidence)
    mismatched = []
    for field, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if _to_cents(str(value)) not in allowed:
                mismatched.append(f"{field}={value}")
    return mismatched
