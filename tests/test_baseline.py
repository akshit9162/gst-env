import importlib.util
from pathlib import Path

from gst_env import generate_task, solve

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).parents[1] / "scripts" / "baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class ScriptedAgent:
    """Lists invoices, reads each one, then submits the true answer."""

    def __init__(self, task, answer, chatter: bool = False):
        self.task, self.answer, self.chatter, self.step = task, answer, chatter, 0

    def start(self, prompt):
        return [prompt]

    def turn(self, messages):
        self.step += 1
        if self.chatter and self.step == 1:
            return baseline.Turn([], "end_turn", 10, 5)  # replies without a tool call
        if self.step <= 2:
            calls = [baseline.ToolCall("l", "list_invoices", {"kind": "sale", "page": 1}),
                     baseline.ToolCall("g", "get_invoice", {"invoice_id": "S-001"}),
                     baseline.ToolCall("bad", "get_invoice", None)]
        else:
            calls = [baseline.ToolCall("s", "submit_return", {"gstr3b": self.answer})]
        return baseline.Turn(calls, "tool_use", 100, 20)

    def add_results(self, messages, results):
        messages.extend(out for _, out in results)

    def nudge(self, messages):
        messages.append("nudge")


def test_episode_submits_and_scores():
    task = generate_task(5, "hard")
    row = baseline.run_episode(ScriptedAgent(task, solve(task)), "claude-opus-5-5", "high", task, max_turns=10)
    assert row["passed"] and row["reward"] == 1.0
    assert row["tool_calls"] == 7 and row["turns"] == 3
    assert row["cost_usd"] > 0


def test_nudge_then_wrong_answer_and_turn_cap():
    task = generate_task(6, "medium")
    wrong = {**solve(task), "total_cash_payable": -1}
    row = baseline.run_episode(ScriptedAgent(task, wrong, chatter=True), "sarvam-105b", "high", task, max_turns=10)
    assert row["submitted"] and not row["passed"] and "total_cash_payable" in row["failed_fields"]
    assert row["effort"] is None

    capped = baseline.run_episode(ScriptedAgent(task, wrong), "claude-opus-5-5", "high", task, max_turns=2)
    assert not capped["submitted"] and capped["reward"] == 0.0


def test_bad_arguments_return_errors_not_crashes():
    session = baseline.FilingSession(generate_task(1, "easy"))
    assert "error" in baseline.call_tool(session, baseline.ToolCall("x", "get_invoice", {}))
    assert "error" in baseline.call_tool(session, baseline.ToolCall("x", "nope", {}))
    assert "error" in baseline.call_tool(session, baseline.ToolCall("x", "submit_return", None))
