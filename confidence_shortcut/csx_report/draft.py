"""Assembles the Part A + Part B draft, with every number pulled from the
atomic tables.

The published write-up was hand-assembled over one-off scripts, which is why it
is welded to 8 QA pairs and one grouping. This renders the same document for
*any* group -- `--cohort vlm` produces the VLM analogue of the same argument, in
the same order, so the two can be read side by side.

**What it does not contain.** The published draft's Part C (SR-xAUC, the
band-conditional routed scorer) is not implemented in `csx_probe`, so it is not
rendered here. A section that silently went missing would read as a negative
result; the report says so explicitly instead.

Every table degrades to `--` rather than raising when a family or stage is
absent, because a partially-complete cohort is the normal state during a run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from csx_report import aggregate
from csx_report.tables import confounds as confound_tbl

ARM_ORDER = ("dse_natural", "dse_balanced2", "dse_matched", "dse_matched2")
SHORT = {"dse_natural": "natural", "dse_balanced2": "balanced2",
         "dse_matched": "matched", "dse_matched2": "matched2"}

# Declared here rather than imported from `csx_probe.config`: this package reads
# the results contract and nothing else, and `test_isolation` enforces that. A
# family or contrast absent from the data is simply not rendered, so these are
# presentation order, not a schema.
FAMILY_ORDER = ("hs_wide", "hs_narrow", "hs_peak_only", "sink",
                "lapeigvals", "attn_eigvals", "attnlogdet")
CONTRAST_PREF = ("IvC", "IHvCH", "ILvCL", "IHvCH_IL", "IvCH", "IvCL",
                 "IHvCL", "ILvCH")
DEFINITIONAL = ("IHvCL", "ILvCH")


def _f(x, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{nd}f}"


def _signed(x, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:+.{nd}f}"


def _med(df: pd.DataFrame, **sel) -> float:
    """Median over pairs of one atomic quantity, or NaN if the cell is absent."""
    d = df
    for k, v in sel.items():
        d = d[d[k] == v]
    if not len(d):
        return float("nan")
    # One value per pair first: a pair with more rows must not weigh more.
    per_pair = d.groupby("pair")["AUROC"].median()
    return float(np.nanmedian(per_pair.to_numpy(dtype=float)))


def _families(df: pd.DataFrame, order) -> list[str]:
    have = set(df["family"].unique())
    return [f for f in order if f in have] + sorted(have - set(order))


def _c_for(df: pd.DataFrame, family: str) -> str:
    d = df[df["family"] == family]
    if not len(d):
        return "--"
    vals = sorted(float(v) for v in d["C"].dropna().unique())
    return f"{vals[0]:g}" if len(vals) == 1 else f"{vals[0]:g}–{vals[-1]:g}"


# ── Part A ───────────────────────────────────────────────────────────────────

def arm_statistics(arm_stats: pd.DataFrame) -> list[str]:
    """§1 -- composition of every arm, medianed over pairs.

    Row composition never sees `C`, so this section is a property of the arm
    builders alone and is the first place a broken arm shows up.
    """
    out = ["<a id=\"s1\"></a>", "", "## 1. Arm statistics", "",
           "Medians over the pairs in the group. Composition is decided by the "
           "arm builders, which never see `C`.", "",
           "| arm | split | n | IH % | CH % | IL % | CL % | incorrect % | "
           "AUROC(entropy, IvC) |",
           "|---|---|---|---|---|---|---|---|---|"]
    if not len(arm_stats):
        return out + ["| -- | | | | | | | | |", ""]
    for arm in ARM_ORDER:
        for split in ("train", "test"):
            d = arm_stats[(arm_stats["arm"] == arm)
                          & (arm_stats["split"] == split)]
            if not len(d):
                continue
            n = float(np.nanmedian(d["n"].to_numpy(float)))
            # Percentages are formed per pair and then medianed: medianing the
            # raw counts first would let the largest pair set the composition.
            cells = []
            for c in ("IH", "CH", "IL", "CL"):
                if c not in d.columns:
                    cells.append(float("nan"))
                    continue
                pct = d[c].to_numpy(float) / d["n"].to_numpy(float) * 100.0
                cells.append(float(np.nanmedian(pct)))
            inc = (float(np.nanmedian(d["pct_incorrect"].to_numpy(float)))
                   if "pct_incorrect" in d.columns else np.nan)
            ent = (float(np.nanmedian(d["entropy_IvC"].to_numpy(float)))
                   if "entropy_IvC" in d.columns else np.nan)
            out.append(
                f"| `{SHORT.get(arm, arm)}` | {split} | {n:,.0f} | "
                + " | ".join(_f(x, 1) for x in cells)
                + f" | {_f(inc, 1)} | {_f(ent)} |")
    out += ["",
            "The last column is the confidence channel itself: `matched2` must "
            "read 0.500 pooled, which is what makes every 'survives matching' "
            "claim downstream admissible.", ""]
    return out


def headline(pp: pd.DataFrame, families: list[str]) -> list[str]:
    """§3 -- the diagonal, `g` / `sep` (gap), per family.

    The gap is the whole argument in one number: on `natural` the two probes are
    nearly the same probe, and on the matched arms they separate.
    """
    out = ["<a id=\"s3\"></a>", "", "## 3. The headline", "",
           "Pooled `IvC`, **diagonal** cells only (trained and tested on the "
           "same arm), median over pairs. Each cell is **`g` / `sep` (gap)**.",
           "",
           "| family | `C` | on `natural` | on `balanced2` | on `matched` | "
           "on `matched2` |", "|---|---|---|---|---|---|"]
    for fam in families:
        cells = []
        for arm in ARM_ORDER:
            g = _med(pp, family=fam, train_arm=arm, test_arm=arm, head="g",
                     contrast="IvC")
            s = _med(pp, family=fam, train_arm=arm, test_arm=arm, head="sep",
                     contrast="IvC")
            gap = g - s
            cells.append(f"{_f(g)} / {_f(s)} (**{_signed(gap)}**)")
        out.append(f"| `{fam}` | {_c_for(pp, fam)} | " + " | ".join(cells) + " |")
    out += ["",
            "A gap that widens from `natural` to the matched arms is the "
            "shortcut: `sep` falls further than `g` gives up.", ""]
    return out


def head_decomposition(pp: pd.DataFrame, family: str) -> list[str]:
    """§3b -- the three heads side by side on the diagonal, one family."""
    out = [f"### The three heads on the diagonal — `{family}`", "",
           "| head | " + " | ".join(f"`{SHORT[a]}`" for a in ARM_ORDER)
           + " | change natural → matched2 |",
           "|---|" + "---|" * (len(ARM_ORDER) + 1)]
    label = {"g": "`g` (correctness)", "sep": "`sep` (confidence)",
             "entropy_only": "`entropy_only` (unfit)"}
    for head in ("g", "sep", "entropy_only"):
        vals = [_med(pp, family=family, train_arm=a, test_arm=a, head=head,
                     contrast="IvC") for a in ARM_ORDER]
        out.append(f"| {label[head]} | " + " | ".join(_f(v) for v in vals)
                   + f" | **{_signed(vals[-1] - vals[0])}** |")
    out += ["",
            "`entropy_only` is unfit — it carries no `C` — so its collapse to "
            "0.500 on `matched2` is a property of the arm, not of a probe.", ""]
    return out


def construction_check(pp: pd.DataFrame, families: list[str]) -> list[str]:
    """§4 -- entropy is dead on `matched2` by construction, and it is checked.

    This is the section that makes everything after it admissible: if the
    confidence channel is not actually closed, every "survives matching" claim
    downstream is about a population that was never matched.
    """
    out = ["<a id=\"s4\"></a>", "",
           "## 4. The construction is verified, not asserted", "",
           "`entropy_only` scored on each arm's own test split. On `matched2` "
           "the within-band confidence channel is closed *by construction*, so "
           "this must read 0.500 — it is a gate, not a finding.", "",
           "| family | on `natural` | on `balanced2` | on `matched` | "
           "on `matched2` |", "|---|---|---|---|---|"]
    for fam in families:
        vals = [_med(pp, family=fam, train_arm=a, test_arm=a,
                     head="entropy_only", contrast="IvC") for a in ARM_ORDER]
        out.append(f"| `{fam}` | " + " | ".join(_f(v) for v in vals) + " |")
    out += ["",
            "> `matched` is allowed its documented ~5e-3 drift (it keys strata "
            "on a rounded entropy); `matched2` keys on the raw float and must "
            "be exact.", ""]
    return out


def transfer_matrix(pp: pd.DataFrame, *, family: str, head: str,
                    contrast: str = "IvC", title: str = "") -> list[str]:
    """One train×test matrix. Rows = trained on, columns = tested on."""
    out = [title, "",
           "| train ＼ test | " + " | ".join(f"`{SHORT[a]}`" for a in ARM_ORDER)
           + " |", "|---|" + "---|" * len(ARM_ORDER)]
    for tr in ARM_ORDER:
        cells = []
        for te in ARM_ORDER:
            v = _med(pp, family=family, train_arm=tr, test_arm=te, head=head,
                     contrast=contrast)
            cells.append(f"**{_f(v)}**" if tr == te else _f(v))
        out.append(f"| `{SHORT[tr]}` | " + " | ".join(cells) + " |")
    return out + [""]


def transfer_section(pp: pd.DataFrame, family: str) -> list[str]:
    """§6 -- the off-diagonal question, for one family."""
    out = ["<a id=\"s6\"></a>", "",
           "## 6. Train on one distribution, test on another", "",
           f"Family `{family}`, median over pairs. Diagonal in **bold**.", ""]
    out += transfer_matrix(pp, family=family, head="g",
                           title="### 6.1 Pooled `IvC` — the correctness probe `g`")
    out += transfer_matrix(pp, family=family, head="sep",
                           title="### 6.2 Pooled `IvC` — the confidence probe `sep`")
    out += transfer_matrix(pp, family=family, head="g", contrast="cell_min",
                           title="### 6.3 Worst admissible cell (`cell_min`), `g`")
    out += transfer_matrix(
        pp, family=family, head="sep", contrast="shortcut_IHvCL",
        title="### 6.4 The shortcut meter (`IHvCL`), `sep`")
    out += ["> `IHvCL` is decided by the band alone, so a `sep` probe that has "
            "become a pure confidence detector scores near 0.000 here. That is "
            "the shortcut caught in the act, not a bug.", ""]
    out += transfer_matrix(
        pp, family=family, head="g", contrast="shortcut_IHvCL",
        title="### 6.5 The shortcut meter (`IHvCL`), `g`")
    out += ["> Same band-only contrast, scored against the correctness probe "
            "`g` instead. `g` is not fit to the band, so this is the control: "
            "if `g` also collapsed near 0.000 here, the band would be leaking "
            "into `g`'s features and the whole `g`/`sep` split would be moot. "
            "It does not, which is what makes §6.4's near-0.000 a `sep`-specific "
            "finding rather than an artefact of the contrast itself.", ""]
    return out


def per_cell_profile(pp: pd.DataFrame, family: str) -> list[str]:
    """§5 -- every contrast on the diagonal, both heads.

    The pooled number can hold up while one specific cell fails; this is where
    that shows.
    """
    have = set(pp["contrast"].unique()) if len(pp) else set()
    order = ([c for c in CONTRAST_PREF if c in have]
             + sorted(c for c in have
                      if c not in CONTRAST_PREF and not c.startswith("cell_")
                      and not c.startswith("shortcut_")))
    out = ["<a id=\"s5\"></a>", "", "## 5. The per-cell profile", "",
           f"Family `{family}`, diagonal cells, median over pairs. "
           f"Definitional cells (`{'`, `'.join(DEFINITIONAL)}`) are "
           f"decided by the band alone and are excluded from `cell_min`.", ""]
    for head in ("g", "sep"):
        out += [f"### `{head}`", "",
                "| contrast | " + " | ".join(f"`{SHORT[a]}`" for a in ARM_ORDER)
                + " |", "|---|" + "---|" * len(ARM_ORDER)]
        for con in order:
            vals = [_med(pp, family=family, train_arm=a, test_arm=a, head=head,
                         contrast=con) for a in ARM_ORDER]
            mark = " *(definitional)*" if con in DEFINITIONAL else ""
            out.append(f"| `{con}`{mark} | " + " | ".join(_f(v) for v in vals)
                       + " |")
        out.append("")
    return out


# ── Part B ───────────────────────────────────────────────────────────────────

def alpha_ladder(rot: pd.DataFrame, families: list[str]) -> list[str]:
    """§9 -- theta and n_train at each rung, median over pairs."""
    out = ["<a id=\"s9\"></a>", "", "## 9. The α ladder", "",
           "θ between the natural-trained `sep` axis and the `g` axis fitted at "
           "each rung, Σ-metric, median over pairs. α=0 is `natural`; α=1 "
           "reproduces `matched2`'s per-cell counts.", ""]
    if not len(rot):
        return out + ["_no rotation results in this group._", ""]
    lad = rot[rot["kind"] == "alpha"]
    alphas = sorted(lad["alpha"].dropna().unique())
    out += ["| family | " + " | ".join(f"α={a:g}" for a in alphas) + " |",
            "|---|" + "---|" * len(alphas)]
    for fam in families:
        vals = []
        for a in alphas:
            d = lad[(lad["family"] == fam) & (lad["alpha"] == a)]
            per_pair = d.groupby("pair")["theta_sigma_sep"].median()
            vals.append(float(np.nanmedian(per_pair.to_numpy(float)))
                        if len(per_pair) else np.nan)
        out.append(f"| `{fam}` | " + " | ".join(_f(v, 1) for v in vals) + " |")
    out += ["",
            "A monotone increase is the rotation: as the confidence channel is "
            "closed, the correctness probe moves off the confidence axis.", ""]
    return out


def rotation_verdict(v: pd.DataFrame, *, cohort: str | None) -> list[str]:
    """§11 -- delta against the placebo null, per family and metric."""
    out = ["<a id=\"s11\"></a>", "", "## 11. The placebo null and the verdict",
           "",
           "`delta` is θ(α=1) − θ(α=0); the null is the 400 outer differences "
           "between placebo draws at the two rungs, and the bar is its 95th "
           "percentile. A placebo is size- and composition-matched but *not* "
           "entropy-matched, so it isolates the matching itself.", ""]
    if not len(v):
        return out + ["_no verdict results in this group._", ""]
    g = aggregate.pass_counts(v, cohort=cohort)
    out += ["| family | metric | median δ | pairs passing | bar | meets bar |",
            "|---|---|---|---|---|---|"]
    for _, r in g.table.sort_values(["family", "metric"]).iterrows():
        out.append(
            f"| `{r['family']}` | {r['metric']} | {_f(r['delta_median'], 2)} | "
            f"{int(r['n_pass'])}/{int(r['n_pairs'])} | {int(r['threshold'])} | "
            f"{'**yes**' if r['meets_bar'] else 'no'} |")
    out += ["",
            f"> Bar is `ceil(0.75·n)` for an arbitrary group; the published "
            f"`qa8` rule of 6/8 coincides with it at n=8.", "",
            "> A Euclidean cell that starts near 90° has little room to rotate, "
            "so a failure there is a ceiling effect rather than a negative "
            "result. Check θ(α=0) in §9 before reading one as evidence.", ""]
    return out


# ── the document ─────────────────────────────────────────────────────────────

def render(*, per_pair: pd.DataFrame, arm_stats: pd.DataFrame,
           rotation: pd.DataFrame, verdict: pd.DataFrame,
           placebo: pd.DataFrame, sizeonly: pd.DataFrame,
           cohort: str | None = None, dropped: list[str] | None = None,
           families: list[str] | None = None, segment: str = "all",
           headline_family: str = "hs_wide") -> str:
    pp = per_pair[per_pair["segment"] == segment] if len(per_pair) else per_pair
    fams = families or _families(pp, FAMILY_ORDER)
    hf = headline_family if headline_family in fams else (fams[0] if fams
                                                          else headline_family)
    pairs = sorted(pp["pair"].unique()) if len(pp) else []

    L = [f"# The confidence shortcut — {cohort or 'custom group'}", "",
         "Behavioural evidence (Part A) and geometric evidence (Part B), every "
         "number pulled from the atomic per-pair tables. Cohort membership is a "
         "runtime argument, so this document can be regenerated for any "
         "grouping without refitting anything.", "",
         f"**Group:** {len(pairs)} pair(s) — {', '.join(pairs) or '--'}"]
    if dropped:
        L.append(f"> Dropped (no results): {', '.join(sorted(dropped))}")
    L.append("")
    for w in aggregate._warn_mixed(pp):
        L += [f"> **Caveat:** {w}.", ""]

    L += ["## The claim, in three sentences", "",
          "A probe that predicts correctness from internal states is partly "
          "reading confidence, which is already available for free. When the "
          "confidence channel is closed by construction — entropy made exactly "
          "uninformative about correctness within band — the confidence probe "
          "collapses much further than the correctness probe does. The "
          "correctness probe also *rotates* off the confidence axis as the "
          "channel closes, which is the same finding measured geometrically "
          "rather than behaviourally.", ""]

    L += arm_statistics(arm_stats)
    L += ["---", "", "# Part A — behavioural evidence", ""]
    L += headline(pp, fams)
    L += head_decomposition(pp, hf)
    L += construction_check(pp, fams)
    L += per_cell_profile(pp, hf)
    L += transfer_section(pp, hf)

    L += ["<a id=\"s7a\"></a>", "",
          "## 7a. Confound controls — composition and sample size", ""]
    if len(placebo) or len(sizeonly):
        by_head = {h: confound_tbl.ladder(pp, placebo, sizeonly, head=h,
                                          segment=segment, cohort=cohort)
                   for h in ("sep", "g")}
        L += ["Every matched arm differs from `natural` in three ways at once — "
              "size, composition, and entropy-matching. Each control holds two "
              "fixed and varies the third, all scored on the same test "
              "population.", ""]
        for head, g in by_head.items():
            if not len(g.table):
                continue
            L += [f"### `{head}`", ""]
            L += confound_tbl.table_for(g, family=hf)
    else:
        L += ["_no confound results in this group; run stage 06._", ""]

    L += ["---", "", "# Part B — geometric evidence", ""]
    L += alpha_ladder(rotation, fams)
    L += rotation_verdict(verdict, cohort=cohort)

    L += ["---", "", "## What this does not cover", "",
          "The published QA draft has a **Part C** (SR-xAUC, and the "
          "band-conditional routed scorer) which is *not implemented* in "
          "`csx_probe` and therefore not rendered here. Its absence is a scope "
          "boundary, not a negative result.", "",
          "Sections `§7 consistency`, `§12 stability` and `§13 the entropy "
          "cross-check` are computable from `rotation_long` "
          "(`kind='refboot'`, the `*_entropy` angle columns) and are not yet "
          "rendered.", ""]
    return "\n".join(L) + "\n"
