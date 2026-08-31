"""Lever A discussion note: does routing survive its own proof obligations?

Four gates per pair (`csx_probe.experiments.lever_a`), each a claim that has to
hold before "routing helps" is trustworthy rather than merely measured:

    drift               the generalist score used elsewhere is reproduced exactly
    affine_identity     oracle routing is affine-invariant; real routing is not
    ece_falls           calibration improves inside each band after pooling
    train_only_provenance  both Platt parameters came from train rows only

A pair failing any gate means its Lever A numbers should not be read as
evidence -- the routing machinery itself is unverified for that pair.
"""

from __future__ import annotations

import pandas as pd

GATES: tuple[str, ...] = ("drift", "affine_identity", "ece_falls",
                          "train_only_provenance")


def gate_table(gates: pd.DataFrame) -> list[str]:
    pairs = sorted(gates["pair"].unique())
    out = ["| pair | " + " | ".join(GATES) + " | all pass |",
           "|---" * (len(GATES) + 2) + "|"]
    all_pass = True
    for p in pairs:
        d = gates[gates["pair"] == p].set_index("gate")["passed"]
        cells = [("PASS" if d.get(g, False) else "**FAIL**") for g in GATES]
        pair_pass = all(d.get(g, False) for g in GATES)
        all_pass &= pair_pass
        out.append(f"| {p} | " + " | ".join(cells) + f" | {'yes' if pair_pass else 'NO'} |")
    out.append("")
    out.append(f"**{'All gates pass across the cohort.' if all_pass else 'AT LEAST ONE GATE FAILS -- see above.'}**")
    out.append("")
    return out


def ece_detail(gates: pd.DataFrame) -> list[str]:
    d = gates[gates["gate"] == "ece_falls"]
    if not len(d):
        return []
    out = ["## Calibration detail (`ece_falls`)", "",
           "| pair | detail |", "|---|---|"]
    for _, r in d.sort_values("pair").iterrows():
        out.append(f"| {r['pair']} | {r['detail']} |")
    out.append("")
    return out


def report(gates: pd.DataFrame, *, cohort: str | None = None,
           dropped: list[str] | None = None) -> str:
    """`DISCUSSION_NOTE.md` analogue -- the routing proof obligations, per pair."""
    pairs = sorted(gates["pair"].unique())
    lines = ["# Lever A — discussion note: do the routing proof obligations hold?",
             ""]
    lines.append(f"**Group:** {cohort or 'custom group'} — {len(pairs)} pair(s): "
                 f"{', '.join(pairs)}")
    if dropped:
        lines.append(f"> Dropped (no results): {', '.join(sorted(dropped))}")
    lines.append("")
    lines += [
        "Each row is one pair's `hs_wide`/`all` unit, evaluated against 4 "
        "gates. `affine_identity` is the interesting one: under **oracle** "
        "routing each cell is scored by exactly one expert, and AUROC within "
        "one band is invariant to a positive affine map, so `z` and "
        "`platt_prior` must be bit-identical there -- and under a **real** "
        "router (a mixture of two experts) that identity must instead break, "
        "so the gate asserts the break rather than the match.", "",
    ]
    lines += gate_table(gates)
    lines += ece_detail(gates)
    return "\n".join(lines) + "\n"
