"""Seeded generator for monthly GSTR-3B filing tasks."""

import random
from dataclasses import asdict, dataclass, field

from .rules import (
    BLOCKED_CATEGORIES,
    ELIGIBLE_CATEGORIES,
    HOME_STATE,
    RATES,
    RCM_CATEGORIES,
    STATES,
    VALUE_STEP,
)

DIFFICULTIES = {
    # n_sales, n_purchases, inter-state, nil-rated, blocked ITC, reverse charge, opening credit ledger
    "easy": dict(n_sales=(3, 5), n_purchases=(2, 4), inter=False, nil=False, blocked=False, rcm=False, ledger=False),
    "medium": dict(n_sales=(6, 10), n_purchases=(4, 7), inter=True, nil=True, blocked=False, rcm=False, ledger=False),
    "hard": dict(n_sales=(10, 18), n_purchases=(8, 14), inter=True, nil=True, blocked=True, rcm=True, ledger=True),
}


@dataclass
class Invoice:
    invoice_id: str
    kind: str  # "sale" or "purchase"
    counterparty_gstin: str | None  # None for B2C sales
    counterparty_state: str
    taxable_value: int
    rate: int
    category: str | None = None  # purchases only
    reverse_charge: bool = False


@dataclass
class Task:
    task_id: str
    difficulty: str
    period: str
    filer_gstin: str
    invoices: list[Invoice] = field(default_factory=list)
    # Electronic credit ledger balance brought forward from earlier periods.
    opening_credit: dict[str, int] = field(default_factory=lambda: {"igst": 0, "cgst": 0, "sgst": 0})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(**{**d, "invoices": [Invoice(**i) for i in d["invoices"]]})


def _gstin(rng: random.Random, state: str) -> str:
    letters = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
    digits = "".join(rng.choice("0123456789") for _ in range(4))
    return f"{state}{letters}{digits}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}1Z{rng.choice('0123456789')}"


def _value(rng: random.Random, lo: int = 5_000, hi: int = 500_000) -> int:
    return rng.randint(lo // VALUE_STEP, hi // VALUE_STEP) * VALUE_STEP


def _state(rng: random.Random, inter_share: float) -> str:
    if rng.random() < inter_share:
        return rng.choice([s for s in STATES if s != HOME_STATE])
    return HOME_STATE


def generate_task(seed: int, difficulty: str = "medium", period: str = "2026-09") -> Task:
    cfg = DIFFICULTIES[difficulty]
    rng = random.Random(f"{difficulty}:{seed}")
    task = Task(
        task_id=f"gstr3b-{difficulty}-{seed:05d}",
        difficulty=difficulty,
        period=period,
        filer_gstin=_gstin(rng, HOME_STATE),
    )
    taxed_rates = [r for r in RATES if r]
    # Some businesses trade mostly within the state, others mostly across states.
    inter_share = rng.uniform(0.0, 0.6) if cfg["inter"] else 0.0

    for i in range(rng.randint(*cfg["n_sales"])):
        b2b = rng.random() < 0.7
        state = _state(rng, inter_share)
        rate = 0 if cfg["nil"] and rng.random() < 0.15 else rng.choice(taxed_rates)
        task.invoices.append(Invoice(
            invoice_id=f"S-{i + 1:03d}",
            kind="sale",
            counterparty_gstin=_gstin(rng, state) if b2b else None,
            counterparty_state=state,
            taxable_value=_value(rng),
            rate=rate,
        ))

    for i in range(rng.randint(*cfg["n_purchases"])):
        state = _state(rng, inter_share)
        if cfg["rcm"] and rng.random() < 0.15:
            category = rng.choice(list(RCM_CATEGORIES))
            rate, rcm = RCM_CATEGORIES[category][1], True
        else:
            pool = list(ELIGIBLE_CATEGORIES)
            if cfg["blocked"] and rng.random() < 0.2:
                pool = list(BLOCKED_CATEGORIES)
            category, rate, rcm = rng.choice(pool), rng.choice(taxed_rates), False
        task.invoices.append(Invoice(
            invoice_id=f"P-{i + 1:03d}",
            kind="purchase",
            counterparty_gstin=_gstin(rng, state),
            counterparty_state=state,
            taxable_value=_value(rng, hi=300_000),
            rate=rate,
            category=category,
            reverse_charge=rcm,
        ))

    # Uneven carried-forward balances make the head-wise set-off rules and the
    # cash-only rule for reverse charge actually change the answer.
    if cfg["ledger"]:
        heavy, light = rng.sample(["cgst", "sgst"], 2)
        task.opening_credit = {
            "igst": rng.choice([0, rng.randint(1, 200_000)]),
            heavy: rng.randint(100_000, 600_000),
            light: rng.randint(0, 50_000),
        }
    return task
