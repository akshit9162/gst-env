"""Run a model as the filing agent and report rewards.

    python scripts/baseline.py --per-difficulty 3 --out results/opus-5-5.jsonl
    python scripts/baseline.py --model sarvam-105b --per-difficulty 3 --out results/sarvam-105b.jsonl
    python scripts/baseline.py --model gpt-6-astra --per-difficulty 3 --out results/gpt-6-astra.jsonl

Claude models read ANTHROPIC_API_KEY, Sarvam models SARVAM_API_KEY and OpenAI
models (gpt-*) OPENAI_API_KEY. All can be set in a local .env file.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anthropic
import openai
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError as SarvamApiError

from gst_env import DIFFICULTIES, FilingSession, generate_task

@dataclass(frozen=True)
class Price:
    """USD per million tokens."""
    input: float
    output: float
    cache_read: float
    cache_write: float


# Sarvam bills in rupees; converted at this rate. Update when it drifts.
INR_PER_USD = 88.0


def _inr(input: float, output: float, cache_read: float) -> Price:
    return Price(input / INR_PER_USD, output / INR_PER_USD, cache_read / INR_PER_USD, input / INR_PER_USD)


# List prices as of Oct 2026; check each provider's pricing page before quoting costs.
# Claude cache writes (5-minute TTL) bill at 1.25x input. Sarvam does not bill writes separately.
PRICES = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
    "gpt-6-astra": Price(10.00, 50.00, 1.00, 12.50),
    "sarvam-105b": _inr(15.0, 60.0, 5.0),  # ₹ per million tokens after the 15 Oct 2026 price cut
}
UNPRICED = Price(0, 0, 0, 0)

HEADS = {
    "type": "object",
    "properties": {h: {"type": "integer"} for h in ("igst", "cgst", "sgst")},
    "required": ["igst", "cgst", "sgst"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "list_invoices",
        "description": "List invoice IDs of one kind for the filing period, 10 per page.",
        "input_schema": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["sale", "purchase"]}, "page": {"type": "integer"}},
            "required": ["kind", "page"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_invoice",
        "description": "Get full details of one invoice: counterparty state code, taxable value, rate, purchase category and reverse-charge flag.",
        "input_schema": {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_credit_ledger",
        "description": "Get the electronic credit ledger balance (IGST, CGST, SGST) brought forward into this period.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "describe_category",
        "description": "Describe what a purchase category code means.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_return",
        "description": "Submit the completed GSTR-3B. All amounts in whole rupees. Can only be called once and ends the task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gstr3b": {
                    "type": "object",
                    "properties": {
                        "outward_taxable_value": {"type": "integer"},
                        "nil_exempt_value": {"type": "integer"},
                        "outward_tax": HEADS,
                        "rcm_tax": HEADS,
                        "itc_available": HEADS,
                        "blocked_itc": {"type": "integer"},
                        "total_cash_payable": {"type": "integer"},
                    },
                    "required": [
                        "outward_taxable_value", "nil_exempt_value", "outward_tax", "rcm_tax",
                        "itc_available", "blocked_itc", "total_cash_payable",
                    ],
                    "additionalProperties": False,
                }
            },
            "required": ["gstr3b"],
            "additionalProperties": False,
        },
    },
]

NUDGE = "You have not submitted yet. Call submit_return with your answer."


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict | None  # None when the model sent arguments that are not valid JSON


@dataclass
class Turn:
    calls: list[ToolCall]
    stop: str
    input_tokens: int  # uncached input only
    output_tokens: int
    fatal: bool = False  # stop the episode (API error, refusal, truncation)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def sarvam_usage(u) -> dict:
    """prompt_tokens includes cached tokens; split them out."""
    if not u:
        return dict(input_tokens=0, output_tokens=0)
    cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
    return dict(input_tokens=u.prompt_tokens - cached, output_tokens=u.completion_tokens, cache_read_tokens=cached)


def openai_usage(u) -> dict:
    """input_tokens includes cache reads and writes; split them out."""
    if not u:
        return dict(input_tokens=0, output_tokens=0)
    d = u.input_tokens_details
    read = (d.cached_tokens or 0) if d else 0
    write = (getattr(d, "cache_write_tokens", 0) or 0) if d else 0
    return dict(input_tokens=u.input_tokens - read - write, output_tokens=u.output_tokens,
                cache_read_tokens=read, cache_write_tokens=write)


class ClaudeAgent:
    def __init__(self, model: str, effort: str):
        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort
        self.tools = [{**t, "strict": True} for t in TOOLS]

    def start(self, prompt: str) -> list:
        return [{"role": "user", "content": prompt}]

    def turn(self, messages: list) -> Turn:
        try:
            r = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                tools=self.tools,
                output_config={"effort": self.effort},
                # The conversation only ever grows by appending, so each turn
                # re-reads the previous prefix from cache.
                cache_control={"type": "ephemeral"},
                messages=messages,
            )
        except anthropic.APIStatusError as e:
            return Turn([], f"api_error_{e.status_code}", 0, 0, fatal=True)
        usage = dict(
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            cache_read_tokens=r.usage.cache_read_input_tokens or 0,
            cache_write_tokens=r.usage.cache_creation_input_tokens or 0,
        )
        if r.stop_reason in ("refusal", "max_tokens"):
            return Turn([], r.stop_reason, fatal=True, **usage)
        messages.append({"role": "assistant", "content": r.content})
        calls = [ToolCall(b.id, b.name, b.input) for b in r.content if b.type == "tool_use"]
        return Turn(calls, r.stop_reason, **usage)

    def add_results(self, messages: list, results: list[tuple[ToolCall, dict]]) -> None:
        messages.append({"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(out),
                **({"is_error": True} if "error" in out else {}),
            }
            for call, out in results
        ]})

    def nudge(self, messages: list) -> None:
        messages.append({"role": "user", "content": NUDGE})


class SarvamAgent:
    def __init__(self, model: str):
        self.client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))
        self.model = model
        self.tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in TOOLS
        ]

    def start(self, prompt: str) -> list:
        return [{"role": "user", "content": prompt}]

    def turn(self, messages: list) -> Turn:
        try:
            r = self.client.chat.completions(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=8192,
            )
        except SarvamApiError as e:
            return Turn([], f"api_error_{e.status_code}", 0, 0, fatal=True)
        usage = sarvam_usage(r.usage)
        choice = r.choices[0]
        if choice.finish_reason == "length":
            return Turn([], "length", fatal=True, **usage)
        tool_calls = choice.message.tool_calls or []
        messages.append({
            "role": "assistant",
            "content": choice.message.content or "",
            **({"tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]} if tool_calls else {}),
        })
        calls = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(tc.id, tc.function.name, args))
        return Turn(calls, choice.finish_reason, **usage)

    def add_results(self, messages: list, results: list[tuple[ToolCall, dict]]) -> None:
        for call, out in results:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(out)})

    def nudge(self, messages: list) -> None:
        messages.append({"role": "user", "content": NUDGE})


class OpenAIAgent:
    """Responses API; every output item (including reasoning) is replayed each turn."""

    def __init__(self, model: str, effort: str):
        self.client = openai.OpenAI()
        self.model, self.effort = model, effort
        self.tools = [
            {"type": "function", "name": t["name"], "description": t["description"], "parameters": t["input_schema"], "strict": True}
            for t in TOOLS
        ]

    def start(self, prompt: str) -> list:
        return [{"role": "user", "content": prompt}]

    def turn(self, messages: list) -> Turn:
        try:
            r = self.client.responses.create(
                model=self.model,
                input=messages,
                tools=self.tools,
                reasoning={"effort": self.effort},
                max_output_tokens=16000,
            )
        except openai.APIStatusError as e:
            return Turn([], f"api_error_{e.status_code}", 0, 0, fatal=True)
        usage = openai_usage(r.usage)
        if r.status != "completed":
            return Turn([], r.status, fatal=True, **usage)
        messages.extend(item.model_dump(exclude_none=True, by_alias=True) for item in r.output)
        calls = []
        for item in r.output:
            if item.type != "function_call":
                continue
            try:
                args = json.loads(item.arguments)
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(item.call_id, item.name, args))
        return Turn(calls, "tool_use" if calls else "end_turn", **usage)

    def add_results(self, messages: list, results: list[tuple[ToolCall, dict]]) -> None:
        for call, out in results:
            messages.append({"type": "function_call_output", "call_id": call.id, "output": json.dumps(out)})

    def nudge(self, messages: list) -> None:
        messages.append({"role": "user", "content": NUDGE})


def make_agent(model: str, effort: str):
    if model.startswith("sarvam"):
        return SarvamAgent(model)
    if model.startswith("gpt"):
        return OpenAIAgent(model, effort)
    return ClaudeAgent(model, effort)


def call_tool(session: FilingSession, call: ToolCall) -> dict:
    args = call.args
    if not isinstance(args, dict):
        return {"error": "arguments were not valid JSON"}
    try:
        match call.name:
            case "list_invoices":
                return session.list_invoices(args["kind"], args.get("page", 1))
            case "get_invoice":
                return session.get_invoice(args["invoice_id"])
            case "describe_category":
                return session.describe_category(args["category"])
            case "get_credit_ledger":
                return session.get_credit_ledger()
            case "submit_return":
                return session.submit_return(args["gstr3b"])
    except (KeyError, TypeError) as e:
        return {"error": f"bad arguments: {e}"}
    return {"error": f"unknown tool {call.name}"}


def run_episode(agent, model: str, effort: str, task, max_turns: int) -> dict:
    session = FilingSession(task)
    messages = agent.start(session.instructions())
    input_tokens = output_tokens = cache_read = cache_write = turns = tool_calls = 0
    stop = None

    while not session.done and turns < max_turns:
        turns += 1
        t = agent.turn(messages)
        input_tokens += t.input_tokens
        output_tokens += t.output_tokens
        cache_read += t.cache_read_tokens
        cache_write += t.cache_write_tokens
        stop = t.stop
        if t.fatal:
            break
        if not t.calls:
            agent.nudge(messages)
            continue
        tool_calls += len(t.calls)
        agent.add_results(messages, [(c, call_tool(session, c)) for c in t.calls])

    p = PRICES.get(model, UNPRICED)
    cost = (input_tokens * p.input + cache_write * p.cache_write
            + cache_read * p.cache_read + output_tokens * p.output) / 1e6
    # What the same episode would have cost with every input token billed at full price.
    uncached_cost = ((input_tokens + cache_write + cache_read) * p.input + output_tokens * p.output) / 1e6
    result = session.result or {"reward": 0.0, "passed": False, "fields": {}}
    return {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "n_invoices": len(task.invoices),
        "model": model,
        "effort": None if model.startswith("sarvam") else effort,
        "reward": result["reward"],
        "passed": result["passed"],
        "submitted": session.done,
        "failed_fields": [f for f, ok in result["fields"].items() if not ok],
        "turns": turns,
        "tool_calls": tool_calls,
        "stop_reason": stop,
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 4),
        "uncached_cost_usd": round(uncached_cost, 4),
    }


def load_dotenv(path: Path = Path(".env")) -> None:
    """Fill unset or empty env vars from a local .env (KEY=value lines)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#") and not os.environ.get(key):
            os.environ[key] = value.strip().strip("'\"")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--difficulties", nargs="+", default=list(DIFFICULTIES))
    parser.add_argument("--per-difficulty", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=1000, help="keeps baseline seeds apart from exported tasks")
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/baseline.jsonl"))
    args = parser.parse_args()

    agent = make_agent(args.model, args.effort)
    tasks = [
        generate_task(args.seed_offset + i, d)
        for d in args.difficulties
        for i in range(args.per_difficulty)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "a") as f:
        futures = [pool.submit(run_episode, agent, args.model, args.effort, t, args.max_turns) for t in tasks]
        rows = []
        for fut in futures:
            row = fut.result()
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{row['task_id']:<22} reward={row['reward']:.2f} passed={row['passed']!s:<5} "
                  f"calls={row['tool_calls']:<3} ${row['cost_usd']:.3f}  missed={row['failed_fields']}")

    print("\nsummary")
    for d in args.difficulties:
        sub = [r for r in rows if r["difficulty"] == d]
        if sub:
            print(f"  {d:<7} pass={sum(r['passed'] for r in sub)}/{len(sub)}  "
                  f"mean_reward={sum(r['reward'] for r in sub) / len(sub):.3f}")
    cost = sum(r["cost_usd"] for r in rows)
    uncached = sum(r["uncached_cost_usd"] for r in rows)
    saved = f"  (without caching ${uncached:.2f}, saved {1 - cost / uncached:.0%})" if uncached else ""
    print(f"  total cost ${cost:.2f}{saved}")
    print(f"  tokens: input {sum(r['input_tokens'] for r in rows)}, cache read {sum(r['cache_read_tokens'] for r in rows)}, "
          f"cache write {sum(r['cache_write_tokens'] for r in rows)}, output {sum(r['output_tokens'] for r in rows)}")


if __name__ == "__main__":
    main()
