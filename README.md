# gst-env

**RL environments and benchmarks for Indian tax and compliance work, used to train and evaluate AI agents.**

Agent training today mostly targets Western SaaS such as Salesforce, Jira and Excel. Compliance work in
emerging markets has almost no coverage: rule-heavy filings, state-wise tax splits, blocked credits,
reverse charge and statutory set-off order. Errors there cost real money, and agents get them wrong.
India alone has over 1.4 crore GST registrations filing returns every month.

`gst-env` starts with **GSTR-3B**, India's monthly GST return. An agent reads a month of invoices
through tools and files the return. A deterministic solver computes the correct answer, and the grader
scores the outcome rather than the steps taken, so the same tasks work for tool-use, GUI and
plain-text agents and for training as well as evaluation.

## Baseline results

| Model | Easy | Medium | Hard |
|---|---|---|---|
| Claude Opus 5.5 (high effort) | pending | pending | pending |
| GPT (OpenAI) | pending | pending | pending |
| Sarvam-105B | pending | pending | pending |

Pass rate = share of tasks where every graded field is correct. Reproduce with `scripts/baseline.py`.

## What an episode looks like

1. The agent receives the filing instructions for a business registered in Telangana.
2. It calls `list_invoices`, `get_invoice`, `describe_category` and `get_credit_ledger` to read
   sales, purchases and the credit balance brought forward.
3. It calls `submit_return` once with the GSTR-3B figures.
4. The grader compares 13 fields to ground truth (₹1 tolerance) and returns a reward in [0, 1].
   `total_cash_payable` carries extra weight. Fields whose true value is zero can only cost reward,
   never earn it, so an empty or all-zero return scores 0.

| Level | Covers |
|---|---|
| easy | Intra-state sales and purchases (CGST + SGST) |
| medium | Adds inter-state supplies (IGST) and nil-rated sales |
| hard | Adds blocked ITC under section 17(5), reverse-charge purchases and an uneven credit-ledger balance brought forward |

The ground truth applies ITC set-off in the Rule 88A order and picks the split of IGST credit that
minimises cash outflow. The test suite checks this against brute-force search.

## Reward integrity

`scripts/reward_audit.py` scores cheating and shortcut strategies over 300 tasks per level.
Cells show mean reward / pass rate.

| Strategy | Easy | Medium | Hard |
|---|---|---|---|
| All zeros | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |
| Constant guess | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |
| Random guess | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |
| Reads sales only | 0.25 / 0% | 0.29 / 0% | 0.30 / 0% |
| Reads first page only | 1.00 / 100% | 1.00 / 100% | 0.41 / 4% |
| Treats all purchases as eligible | 1.00 / 100% | 1.00 / 100% | 0.40 / 2% |
| Treats all parties as in-state | 1.00 / 100% | 0.58 / 13% | 0.42 / 9% |
| Ignores the credit ledger | 1.00 / 100% | 1.00 / 100% | 0.69 / 9% |
| Pools credit across heads | 1.00 / 100% | 1.00 / 100% | 0.86 / 57% |
| Uses ITC to pay reverse-charge tax | 1.00 / 100% | 1.00 / 100% | 0.81 / 41% |

Strategies that do no work earn nothing. Each shortcut passes only at the levels that leave out the
rule it skips, and fails hard tasks often. Credit pooling still passes 57% of hard tasks because on
those tasks the head-wise rule genuinely does not change the answer; the planned expert tier targets
it directly. `tests/test_reward_audit.py` fails the build if any of this regresses.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,baseline]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/export_tasks.py --per-difficulty 50     # data/tasks.jsonl + answers.jsonl

# API keys go in .env: ANTHROPIC_API_KEY, OPENAI_API_KEY and/or SARVAM_API_KEY
.venv/bin/python scripts/baseline.py --per-difficulty 3 --out results/opus-5-5.jsonl
.venv/bin/python scripts/baseline.py --model gpt-6-astra --per-difficulty 3 --out results/gpt-6-astra.jsonl
.venv/bin/python scripts/baseline.py --model sarvam-105b --per-difficulty 3 --out results/sarvam-105b.jsonl
```

## Layout

| File | Purpose |
|---|---|
| `gst_env/rules.py` | Rates, states, blocked (17(5)) and reverse-charge categories |
| `gst_env/generator.py` | Seeded task generator |
| `gst_env/solver.py` | Ground truth, including Rule 88A set-off with minimum cash |
| `gst_env/grader.py` | Weighted field-level reward |
| `gst_env/tools.py` | Tool interface and episode state |
| `scripts/export_tasks.py` | Dumps tasks and answer keys as JSONL |
| `scripts/baseline.py` | Runs Claude, OpenAI (gpt-*) or Sarvam models as the agent and records rewards and cost |
| `scripts/reward_audit.py` | Scores degenerate and shortcut strategies to check the reward can't be gamed |

## Roadmap

- **Expert tier:** CA-designed tasks covering credit and debit notes, GSTR-2B mismatches, amendments,
  Rule 42/43 reversals and late ITC under section 16(4).
- **GUI tier:** a mock GST portal for computer-use agents, graded on the same ground truth.
- **More filings:** GSTR-1 reconciliation, TDS returns, Tally bookkeeping tasks.
- **Recorded demonstrations** by practising CAs.
- **Adapters** for Prime Intellect `verifiers` and HUD.

## Simplifications (CA review in progress)

1. Rates use the post-22 Sep 2025 slabs (0/5/18/40). No cess.
2. Place of supply is the counterparty's state. No exports, SEZ or special place-of-supply services.
3. Blocked credit is all-or-nothing per invoice. No Rule 42/43 or Rule 37 reversals.
4. ITC is assumed to be reflected in GSTR-2B, with no mismatches or time limits.
5. Reverse-charge categories are a subset of notification 13/2017, with fixed rates.
6. No credit or debit notes, amendments, advances, interest or late fee.
7. B2C place-of-supply reporting (table 3.2) is not graded.
8. Credit-ledger opening balances are synthetic, and cash-ledger balances and interest are not modelled.

Each item is a candidate for the expert tier once a CA has validated it.
