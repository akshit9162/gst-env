"""Score degenerate and shortcut strategies to check the reward can't be gamed.

    python scripts/reward_audit.py --tasks 200

Degenerate strategies (no real work) should score ~0. Shortcut strategies
(real but incomplete work) may earn partial credit but must never pass.
"""

import argparse
import random
from dataclasses import replace

from gst_env import DIFFICULTIES, generate_task, grade, solve
from gst_env.generator import Task
from gst_env.rules import HOME_STATE
from gst_env.solver import HEADS, minimum_cash


def zeros(task: Task) -> dict:
    z = {h: 0 for h in HEADS}
    return {"outward_taxable_value": 0, "nil_exempt_value": 0, "outward_tax": z, "rcm_tax": z,
            "itc_available": z, "blocked_itc": 0, "total_cash_payable": 0}


def constant(task: Task) -> dict:
    c = {h: 50_000 for h in HEADS}
    return {"outward_taxable_value": 500_000, "nil_exempt_value": 50_000, "outward_tax": c, "rcm_tax": c,
            "itc_available": c, "blocked_itc": 50_000, "total_cash_payable": 150_000}


def random_guess(task: Task) -> dict:
    rng = random.Random(task.task_id)
    r = lambda: rng.randint(0, 400_000)  # noqa: E731
    return {"outward_taxable_value": r() * 10, "nil_exempt_value": r(), "outward_tax": {h: r() for h in HEADS},
            "rcm_tax": {h: r() for h in HEADS}, "itc_available": {h: r() for h in HEADS},
            "blocked_itc": r(), "total_cash_payable": r()}


def first_page_only(task: Task) -> dict:
    """Reads only the first 10 sales and 10 purchases (skips pagination)."""
    sales = [i for i in task.invoices if i.kind == "sale"][:10]
    purchases = [i for i in task.invoices if i.kind == "purchase"][:10]
    return solve(replace(task, invoices=sales + purchases))


def sales_only(task: Task) -> dict:
    """Computes outward tax correctly, ignores all purchases."""
    return solve(replace(task, invoices=[i for i in task.invoices if i.kind == "sale"]))


def all_purchases_eligible(task: Task) -> dict:
    """Ignores blocked credit and reverse charge: every purchase is normal ITC."""
    clean = [i if i.kind == "sale" else replace(i, category="raw_material", reverse_charge=False) for i in task.invoices]
    return solve(replace(task, invoices=clean))


def all_intra_state(task: Task) -> dict:
    """Treats every counterparty as in-state (CGST+SGST only)."""
    return solve(replace(task, invoices=[replace(i, counterparty_state=HOME_STATE) for i in task.invoices]))


def ignore_ledger(task: Task) -> dict:
    """Never reads the credit ledger brought forward."""
    return solve(replace(task, opening_credit={h: 0 for h in HEADS}))


def pooled_setoff(task: Task) -> dict:
    """Correct figures, but cash = total liability - total ITC (ignores head rules)."""
    out = solve(task)
    liab = sum(out["outward_tax"].values())
    credit = sum(out["itc_available"].values()) + sum(task.opening_credit.values())
    return {**out, "total_cash_payable": max(liab - credit, 0) + sum(out["rcm_tax"].values())}


def rcm_offset_by_credit(task: Task) -> dict:
    """Correct figures, but lets ITC pay reverse-charge tax too."""
    out = solve(task)
    liab = {h: out["outward_tax"][h] + out["rcm_tax"][h] for h in HEADS}
    credit = {h: out["itc_available"][h] + task.opening_credit[h] for h in HEADS}
    return {**out, "total_cash_payable": sum(minimum_cash(liab, credit).values())}


DEGENERATE = {"zeros": zeros, "constant": constant, "random_guess": random_guess}
SHORTCUT = {
    "sales_only": sales_only,
    "first_page_only": first_page_only,
    "all_purchases_eligible": all_purchases_eligible,
    "all_intra_state": all_intra_state,
    "ignore_ledger": ignore_ledger,
    "pooled_setoff": pooled_setoff,
    "rcm_offset_by_credit": rcm_offset_by_credit,
}


def audit(n_tasks: int) -> dict[str, dict[str, tuple[float, float]]]:
    """Returns {strategy: {difficulty: (mean_reward, pass_rate)}}."""
    results = {}
    for name, fn in {**DEGENERATE, **SHORTCUT}.items():
        results[name] = {}
        for d in DIFFICULTIES:
            graded = [grade(t, fn(t)) for t in (generate_task(5000 + i, d) for i in range(n_tasks))]
            results[name][d] = (
                sum(g["reward"] for g in graded) / n_tasks,
                sum(g["passed"] for g in graded) / n_tasks,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=200)
    args = parser.parse_args()
    results = audit(args.tasks)
    print(f"{'strategy':<24}" + "".join(f"{d:>20}" for d in DIFFICULTIES))
    for name, row in results.items():
        cells = "".join(f"{f'{r:.2f} / {p:.0%}':>20}" for r, p in row.values())
        kind = "degenerate" if name in DEGENERATE else "shortcut"
        print(f"{name:<24}{cells}   ({kind})")
    print("\ncells = mean reward / pass rate")


if __name__ == "__main__":
    main()
