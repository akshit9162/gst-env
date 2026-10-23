"""Outcome-based grading: compares a submitted return to ground truth,
independent of how the agent got there (GUI, tools or plain text)."""

from .generator import Task
from .solver import HEADS, solve

TOLERANCE = 1  # rupees

# Field -> weight. The headline number carries the most weight; per-head
# fields catch agents that get the total right for the wrong reasons.
WEIGHTS = {
    "outward_taxable_value": 1,
    "nil_exempt_value": 1,
    **{f"outward_tax.{h}": 1 for h in HEADS},
    **{f"rcm_tax.{h}": 1 for h in HEADS},
    **{f"itc_available.{h}": 1 for h in HEADS},
    "blocked_itc": 1,
    "total_cash_payable": 5,
}


def _get(d: dict, path: str):
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def grade(task: Task, submission: dict) -> dict:
    truth = solve(task)
    fields = {}
    for path in WEIGHTS:
        got = _get(submission, path)
        try:
            ok = got is not None and abs(float(got) - _get(truth, path)) <= TOLERANCE
        except (TypeError, ValueError):
            ok = False
        fields[path] = ok
    score = sum(WEIGHTS[p] for p, ok in fields.items() if ok) / sum(WEIGHTS.values())
    return {
        "reward": round(score, 4),
        "passed": all(fields.values()),
        "fields": fields,
    }
