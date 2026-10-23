"""Ground-truth GSTR-3B computation, including Rule 88A ITC utilisation."""

from .generator import Invoice, Task
from .rules import BLOCKED_CATEGORIES, HOME_STATE

HEADS = ("igst", "cgst", "sgst")


def split_tax(inv: Invoice) -> dict[str, int]:
    tax = inv.taxable_value * inv.rate // 100
    if inv.counterparty_state == HOME_STATE:
        return {"igst": 0, "cgst": tax // 2, "sgst": tax // 2}
    return {"igst": tax, "cgst": 0, "sgst": 0}


def _add(acc: dict[str, int], tax: dict[str, int]) -> None:
    for h in HEADS:
        acc[h] += tax[h]


def _zero() -> dict[str, int]:
    return {h: 0 for h in HEADS}


def _cash_after_credit(liab: dict[str, int], itc: dict[str, int], igst_to_cgst: int) -> dict[str, int]:
    """Apply credit in the Rule 88A order for one choice of how leftover IGST
    credit is split between CGST and SGST. Returns cash payable per head."""
    liab, itc = dict(liab), dict(itc)

    # 1. IGST credit first against IGST, then CGST/SGST (split is the free choice).
    used = min(itc["igst"], liab["igst"])
    liab["igst"] -= used
    itc["igst"] -= used
    to_c = min(igst_to_cgst, itc["igst"], liab["cgst"])
    liab["cgst"] -= to_c
    itc["igst"] -= to_c
    to_s = min(itc["igst"], liab["sgst"])
    liab["sgst"] -= to_s
    itc["igst"] -= to_s

    # 2. CGST credit against CGST then IGST; SGST credit against SGST then IGST.
    #    CGST and SGST credit can never cross-utilise each other.
    for head in ("cgst", "sgst"):
        used = min(itc[head], liab[head])
        liab[head] -= used
        itc[head] -= used
        used = min(itc[head], liab["igst"])
        liab["igst"] -= used
        itc[head] -= used
    return liab


def minimum_cash(liab: dict[str, int], itc: dict[str, int]) -> dict[str, int]:
    """Cash payable per head under the split of leftover IGST credit that
    minimises total cash. The optimum lies on a breakpoint, so only those are checked."""
    spare = max(itc["igst"] - liab["igst"], 0)
    points = [
        0,
        spare,
        liab["cgst"],
        spare - liab["sgst"],
        liab["cgst"] - itc["cgst"],
        spare - liab["sgst"] + itc["sgst"],
    ]
    best = None
    for x in sorted({min(max(p, 0), spare) for p in points}):
        cash = _cash_after_credit(liab, itc, x)
        if best is None or sum(cash.values()) < sum(best.values()):
            best = cash
    return best


def solve(task: Task) -> dict:
    outward = _zero()  # 3.1(a) tax on taxable outward supplies
    rcm = _zero()  # 3.1(d) tax on inward supplies under reverse charge
    itc = _zero()  # net eligible ITC (including ITC on RCM paid)
    taxable_value = nil_value = blocked_itc = 0

    for inv in task.invoices:
        tax = split_tax(inv)
        if inv.kind == "sale":
            if inv.rate == 0:
                nil_value += inv.taxable_value
            else:
                taxable_value += inv.taxable_value
                _add(outward, tax)
        elif inv.reverse_charge:
            _add(rcm, tax)
            _add(itc, tax)
        elif inv.category in BLOCKED_CATEGORIES:
            blocked_itc += sum(tax.values())
        else:
            _add(itc, tax)

    # RCM tax must be paid in cash; ITC can only be set off against forward-charge tax.
    cash = minimum_cash(outward, itc)
    for h in HEADS:
        cash[h] += rcm[h]

    return {
        "outward_taxable_value": taxable_value,
        "nil_exempt_value": nil_value,
        "outward_tax": outward,
        "rcm_tax": rcm,
        "itc_available": itc,
        "blocked_itc": blocked_itc,
        "cash_payable": cash,
        "total_cash_payable": sum(cash.values()),
    }
