"""Is this group's evidence complete enough to median over?

Two different things can make a report partial, and only one of them was ever
visible:

**Missing pairs.** A named pair with no results at all. `cohorts.resolve`
already returns these as `dropped` and `render` prints them in a banner, so this
axis was never silent -- but it also never stopped the report being written.

**Ragged coverage.** A pair that IS present but was fit over a subset of the
families or segments the others have. This one is genuinely invisible: the pair
appears in the header count, contributes to some cells and not others, and the
median's denominator changes from row to row with nothing on the page saying so.
It is what a smoke run leaves behind -- one family, one segment, real numbers,
indistinguishable from a full unit except by counting.

The rule here is that a report over an incomplete group is refused by default
rather than annotated. An annotated partial report still gets read as the
result; a refusal gets fixed. `--allow-partial` exists for the case where the
group really is all there will ever be, and stamps the shortfall into the
document instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# What makes two units comparable. A pair missing a whole (family, segment) that
# its peers have is ragged; differing row counts within a cell are not, since
# arms legitimately differ in size across pairs.
COVERAGE_KEYS: tuple[str, ...] = ("family", "segment")


@dataclass
class Coverage:
    """What a group has, against what its best-covered member has."""
    present: list[str]
    dropped: list[str]
    ragged: dict[str, list[str]] = field(default_factory=dict)
    reference: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.dropped and not self.ragged

    def reasons(self) -> list[str]:
        out: list[str] = []
        if self.dropped:
            out.append(f"{len(self.dropped)} pair(s) named but absent: "
                       f"{', '.join(sorted(self.dropped))}")
        for pair, missing in sorted(self.ragged.items()):
            shown = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
            out.append(f"{pair}: missing {len(missing)} of "
                       f"{len(self.reference)} (family, segment) cells "
                       f"[{shown}]")
        return out


def _cells(df: pd.DataFrame) -> set[tuple]:
    keys = [k for k in COVERAGE_KEYS if k in df.columns]
    if not keys:
        return set()
    return set(map(tuple, df[keys].drop_duplicates().itertuples(index=False)))


def check(per_pair: pd.DataFrame, present: list[str],
          dropped: list[str]) -> Coverage:
    """Compare each present pair's (family, segment) cells against the fullest.

    The reference is the union over the group, not a configured expectation: the
    families a pair *should* have depend on its modality (text pairs have no
    image segment), so a fixed list would flag every text pair as ragged. Taking
    the union means "ragged" always reads as "some peer in this very group has
    evidence this pair lacks", which is exactly the condition that makes a
    median's denominator vary by row.
    """
    sub = per_pair[per_pair["pair"].isin(present)]
    per = {p: _cells(g) for p, g in sub.groupby("pair")}
    reference: set[tuple] = set().union(*per.values()) if per else set()

    ragged = {}
    for pair, cells in per.items():
        missing = reference - cells
        if missing:
            ragged[pair] = sorted("/".join(map(str, c)) for c in missing)

    return Coverage(present=list(present), dropped=list(dropped),
                    ragged=ragged,
                    reference=sorted("/".join(map(str, c)) for c in reference))


def banner(cov: Coverage) -> str:
    """A markdown blockquote for a report that was written anyway."""
    if cov.complete:
        return ""
    lines = ["> **PARTIAL GROUP -- this is not a complete result.**", ">"]
    lines += [f"> - {r}" for r in cov.reasons()]
    lines += [">",
              "> Cells above are medianed over whichever pairs have them, so "
              "the denominator varies by row. Re-run the missing units and "
              "regenerate before citing any number here."]
    return "\n".join(lines) + "\n"
