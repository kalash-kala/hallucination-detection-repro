"""SR-xAUC: the worst confidence-conditioned AUROC a scorer survives.

A pooled `IvC` of 0.88 can be almost entirely composition: if the confident rows
are mostly correct and the uncertain rows mostly incorrect, a probe that has
learned only "which band is this" scores well without distinguishing correctness
inside either band. SR-xAUC removes that escape route by conditioning on the
band and taking the WORST case:

    §5, per evaluation unit (a model × dataset pair):
        minimise AUROC over 4 atomic cells × 4 test arms = 16 candidates
    then:
        median that per-unit minimum across units

> **The trap that would silently inflate every number in Part C: `min` does not
> commute with `median`.** Minimising *inside* each unit and then medianing is
> the §5 quantity. Medianing each cell across units first and then minimising is
> a different, systematically LARGER number — every unit gets to be rescued by
> the others in whichever cell it happens to fail. Legacy `58` carries the
> commuted version as `approx_min_of_medians` and **never calls it SR-xAUC**. We
> keep both, under both names, and `test_srxauc.py` builds a frame where they
> differ and asserts we report the smaller one.

Three companions are rendered non-optionally, because each one is load-bearing:

**`G` (stability)** — the max across test arms of a cell's spread, medianed over
units. A *constant* scorer has `G = 0` and SR-xAUC = 0.5, so flatness on its own
is worthless; low `G` means something only beside a high SR-xAUC. Kept in its own
column and never folded into the level.

**`gap = pooled IvC − SR-xAUC`** — how much of the headline was resting on
favourable confidence composition. This column *is* the metric's argument.

**`asym = IHvCL − ILvCH`** — the confidence-direction ladder, in ONE convention
(the report's). Legacy prose uses the reversed sign in places; that discrepancy
must not survive the port, so the sign is fixed here and asserted in the tests.

**The §17.3 guard.** `entropy_only` scores ~0.000 on the band-conditioned cells
*by construction* — the bands are cut from entropy, so within a band entropy has
no remaining signal to give. It carries a mandatory footnote and is refused a
place in any ranked comparison. Reporting semantic-entropy methods as "defeated"
on this table is the single easiest way for this work to be wrong in public.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from csx_report.aggregate import _warn_mixed

# The constants come from the SHARED config, read directly -- never by importing
# csx_probe. `configs/` is the one thing all three components may read, and
# tests/test_isolation.py enforces that csx_report's import graph stays free of
# the other two packages: this table has to run on a box that has only pandas.
import yaml  # noqa: E402

from csx_common import paths  # noqa: E402

_ALL = yaml.safe_load((paths.CONFIG_DIR / "frozen_constants.yaml").read_text())
_FROZEN = _ALL["srxauc"]

ATOMIC: tuple[str, ...] = tuple(_FROZEN["atomic"])
TEST_ARMS: tuple[str, ...] = tuple(_FROZEN["test_arms"])
UNRANKED: tuple[str, ...] = tuple(_FROZEN["unranked"])
SEED: int = int(_ALL["seed"])
TRAIN_PRIMARY: tuple[str, ...] = tuple(_FROZEN["train_primary"])
TRAIN_STRESS: tuple[str, ...] = tuple(_FROZEN["train_stress"])
N_BOOT: int = int(_FROZEN["n_boot"])
CI: tuple[float, float] = tuple(_FROZEN["ci"])

SHORT: dict[str, str] = {a: a.replace("dse_", "") for a in TEST_ARMS}
UNIT_KEYS: tuple[str, ...] = ("family", "method")


class SRxAUCError(Exception):
    """The input frame cannot support an SR-xAUC computation."""


def method_key(df: pd.DataFrame) -> pd.Series:
    """`scorer` for unrouted designs, `scorer@router` for routed ones.

    A routed scorer is a different method under each router -- that is the
    deployability axis -- so collapsing them would average a deployable design
    with an undeployable one.
    """
    if "router" not in df.columns:
        return df["scorer"].astype(str)
    r = df["router"].astype(str)
    routed = r.notna() & (r != "") & (r.str.lower() != "none")
    return np.where(routed, df["scorer"].astype(str) + "@" + r,
                    df["scorer"].astype(str))


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    need = {"family", "pair", "train_arm", "test_arm", "contrast", "AUROC"}
    missing = need - set(df.columns)
    if missing:
        raise SRxAUCError(
            f"SR-xAUC needs {sorted(missing)}; got {sorted(df.columns)[:12]}")
    d = df.copy()
    if "segment" in d.columns:
        # Part C is scoped to the `all` segment only -- it is the only one the
        # 8-QA parity target ever had (text pairs have no image/text split).
        # `groupby(["family", "method", "pair"])` below has no `segment` key,
        # so a stray `image`/`text` row for a VLM pair would silently mix into
        # the same group as `all` and corrupt the min/median with an
        # arbitrary cross-segment comparison. Filtering here, once, up front,
        # means every downstream groupby stays correct without having to
        # remember to add `segment` to each one.
        d = d[d["segment"] == "all"]
    if "method" not in d.columns:
        d["method"] = method_key(d)
    return d


# ── the metric ───────────────────────────────────────────────────────────────

def per_unit_min(df: pd.DataFrame, train_arms=TRAIN_PRIMARY) -> pd.DataFrame:
    """§5 step 1: the worst atomic xAUC *within* each evaluation unit.

    Also records WHICH candidate was limiting. That is the interpretable half of
    the result: a method limited by `IHvCL@natural` is failing the
    anti-confidence comparison, which is a different disease from one limited by
    `ILvCL@matched`.
    """
    d = _prepare(df)
    a = d[d["contrast"].isin(ATOMIC) & d["train_arm"].isin(list(train_arms))]
    a = a[a["AUROC"].notna()]
    if not len(a):
        return pd.DataFrame(columns=["family", "method", "pair", "worst",
                                     "limiting"])
    i = a.groupby(["family", "method", "pair"])["AUROC"].idxmin()
    keep = [c for c in ("family", "method", "pair", "AUROC", "train_arm",
                        "test_arm", "contrast", "n", "n_pos") if c in a.columns]
    w = a.loc[i, keep].rename(columns={"AUROC": "worst"})
    return w.assign(limiting=w["contrast"] + "@"
                    + w["test_arm"].map(lambda x: SHORT.get(x, x)))


def boot_ci(worst: np.ndarray, *, n_boot: int = N_BOOT, seed: int = SEED
            ) -> tuple[float, float]:
    """Unit-level bootstrap ONLY: resamples units over their per-unit minima.

    The guide's §10 interval also resamples examples *within* a unit, which is
    impossible to reconstruct from per-cell AUROCs. So this interval is narrower
    than §10's, and every place it is rendered says so.
    """
    worst = np.asarray(worst, dtype=float)
    if not len(worst):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(worst), size=(n_boot, len(worst)))
    reps = np.median(worst[idx], axis=1)
    return (float(np.percentile(reps, CI[0])), float(np.percentile(reps, CI[1])))


def stability_gap(df: pd.DataFrame, train_arms=TRAIN_PRIMARY) -> pd.DataFrame:
    """§9's `G_m`: max over atomic cells of the spread across test arms."""
    d = _prepare(df)
    a = d[d["contrast"].isin(ATOMIC) & d["train_arm"].isin(list(train_arms))]
    if not len(a):
        return pd.DataFrame(columns=["family", "method", "G_stability"])
    s = (a.groupby(["family", "method", "pair", "contrast"])["AUROC"]
          .agg(lambda x: x.max() - x.min()))
    g = s.groupby(["family", "method", "pair"]).max()
    return (g.groupby(["family", "method"]).median()
             .rename("G_stability").reset_index())


def approx_min_of_medians(df: pd.DataFrame, train_arms=TRAIN_PRIMARY
                          ) -> pd.DataFrame:
    """§8: min over (test arm, contrast) of the MEDIAN across units.

    The COMMUTED order. Reported only to quantify how far it strays, and never
    labelled SR-xAUC anywhere in this codebase.
    """
    d = _prepare(df)
    a = d[d["contrast"].isin(ATOMIC) & d["train_arm"].isin(list(train_arms))]
    if not len(a):
        return pd.DataFrame(columns=["family", "method", "approx_min_of_medians"])
    m = a.groupby(["family", "method", "test_arm", "contrast"])["AUROC"].median()
    return (m.groupby(["family", "method"]).min()
             .rename("approx_min_of_medians").reset_index())


def pooled_ivc(df: pd.DataFrame) -> pd.DataFrame:
    """The level at the observed deployment prevalence: train=test=natural."""
    d = _prepare(df)
    a = d[(d["contrast"] == "IvC") & (d["train_arm"] == "dse_natural")
          & (d["test_arm"] == "dse_natural")]
    if not len(a):
        return pd.DataFrame(columns=["family", "method", "pooled_IvC_natural"])
    return (a.groupby(["family", "method"])["AUROC"].median()
             .rename("pooled_IvC_natural").reset_index())


def asymmetry(df: pd.DataFrame, train_arms=TRAIN_PRIMARY) -> pd.DataFrame:
    """`asym = IHvCL − ILvCH`, medianed over units. ONE sign convention.

    Positive means the scorer ranks a confident-incorrect row above an
    uncertain-correct one -- i.e. it is reading correctness against the
    confidence gradient rather than with it.
    """
    d = _prepare(df)
    a = d[d["contrast"].isin(["IHvCL", "ILvCH"])
          & d["train_arm"].isin(list(train_arms))
          & (d["test_arm"] == "dse_natural")]
    if not len(a):
        return pd.DataFrame(columns=["family", "method", "asym"])
    p = (a.pivot_table(index=["family", "method", "pair"], columns="contrast",
                       values="AUROC", aggfunc="median"))
    if not {"IHvCL", "ILvCH"} <= set(p.columns):
        return pd.DataFrame(columns=["family", "method", "asym"])
    p = (p["IHvCL"] - p["ILvCH"]).rename("asym").reset_index()
    return p.groupby(["family", "method"], as_index=False)["asym"].median()


def compute(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(results, atomic)` — the §5 metric plus the per-cell median breakdown."""
    d = _prepare(df)
    rows: list[dict] = []
    for label, train_arms in (("primary", TRAIN_PRIMARY),
                              ("stress", TRAIN_STRESS)):
        w = per_unit_min(d, train_arms)
        if not len(w):
            continue
        for (fam, meth), grp in w.groupby(["family", "method"]):
            worst = grp["worst"].to_numpy(dtype=float)
            lo, hi = boot_ci(worst)
            mode = grp["limiting"].mode()
            rows.append({
                "variant": label, "family": fam, "method": meth,
                "sr_xauc": float(np.median(worst)),
                "ci_lo": lo, "ci_hi": hi,
                "sr_xauc_star": 2.0 * float(np.median(worst)) - 1.0,
                "limiting_modal": mode.iloc[0] if len(mode) else "",
                "limiting_n_units": (int((grp["limiting"] == mode.iloc[0]).sum())
                                     if len(mode) else 0),
                "worst_unit": float(worst.min()),
                "best_unit": float(worst.max()),
                "n_units": int(len(worst)),
            })
    res = pd.DataFrame(rows)
    if not len(res):
        return res, pd.DataFrame()

    extra = stability_gap(d, TRAIN_PRIMARY)
    for f in (approx_min_of_medians, pooled_ivc, asymmetry):
        extra = extra.merge(f(d), on=["family", "method"], how="outer")
    res = res.merge(extra, on=["family", "method"], how="left")
    # The §8 approximation and the descriptive stats are natural-trained only;
    # leaving them on the stress rows would attach a primary-variant number to a
    # differently-trained result.
    res.loc[res["variant"] == "stress",
            ["G_stability", "approx_min_of_medians", "pooled_IvC_natural",
             "asym"]] = np.nan
    res["gap"] = res["pooled_IvC_natural"] - res["sr_xauc"]
    res["unranked"] = res["method"].map(
        lambda m: str(m).split("@")[0] in UNRANKED)

    atomic = (d[d["contrast"].isin(ATOMIC) & (d["train_arm"] == "dse_natural")]
              .groupby(["family", "method", "test_arm", "contrast"])["AUROC"]
              .median().rename("median_AUROC").reset_index())
    return res, atomic


def rankable(res: pd.DataFrame) -> pd.DataFrame:
    """The rows a ranked comparison may include — §17.3, enforced in code.

    `entropy_only` is excluded because its ~0.000 on the band-conditioned cells
    is definitional, not a defeat: the bands ARE the entropy cut.
    """
    return res[~res["unranked"]].copy()


def footnotes(res: pd.DataFrame) -> list[str]:
    """The notes that must accompany any rendering of this table."""
    out = [
        "SR-xAUC is the §5 quantity: the minimum is taken over 16 candidates "
        "(4 atomic cells × 4 test arms) **inside** each unit, and only then "
        "medianed across units. `approx_min_of_medians` is the commuted §8 "
        "order, is systematically larger, and is **not** SR-xAUC.",
        "CIs are unit-level bootstraps over per-unit minima only; the guide's "
        "§10 within-unit example resampling cannot be reconstructed from "
        "per-cell AUROCs, so these intervals are narrower than §10's.",
        "`G` is a stability measure, not a level: a constant scorer has G = 0 "
        "and SR-xAUC = 0.5, so a low G means something only beside a high "
        "SR-xAUC.",
    ]
    if bool(res.get("unranked", pd.Series(dtype=bool)).any()):
        names = sorted({m for m, u in zip(res["method"], res["unranked"]) if u})
        out.append(
            f"{', '.join('`' + n + '`' for n in names)} scores ~0.000 on the "
            f"band-conditioned cells **by construction** — the bands are cut "
            f"from entropy, so no within-band signal remains to measure. It is "
            f"shown for reference and is excluded from every ranked comparison; "
            f"reporting it as 'defeated' here would be wrong.")
    return out + _warn_mixed(res) if "c_mode" in res.columns else out


# ── rendering ────────────────────────────────────────────────────────────────

def _fmt(x, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{nd}f}"


def table_for(res: pd.DataFrame, *, family: str, variant: str = "primary"
             ) -> list[str]:
    d = res[(res["family"] == family) & (res["variant"] == variant)]
    if not len(d):
        return []
    # unranked rows (entropy_only) sink to the bottom regardless of level.
    d = pd.concat([d[~d["unranked"]].sort_values("sr_xauc", ascending=False),
                   d[d["unranked"]]])
    out = [f"### {family} ({variant})", "",
           "| method | SR-xAUC | 95% CI | gap | asym | G | limiting | n |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        m = r["method"] + (" _(unranked)_" if r["unranked"] else "")
        out.append(
            f"| {m} | {_fmt(r['sr_xauc'])} | "
            f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | {_fmt(r['gap'])} | "
            f"{_fmt(r['asym'])} | {_fmt(r['G_stability'])} | "
            f"{r['limiting_modal']} ({int(r['limiting_n_units'])}/{int(r['n_units'])}) "
            f"| {int(r['n_units'])} |")
    out.append("")
    return out


def report(res: pd.DataFrame, *, cohort: str | None = None,
           pairs: list[str] | None = None,
           dropped: list[str] | None = None,
           families: list[str] | None = None) -> str:
    """`SR_XAUC_RESULTS.md` analogue — the §5 metric, worst case first."""
    lines = ["# SR-xAUC — the worst confidence-conditioned AUROC a scorer survives",
             ""]
    lines.append(f"**Group:** {cohort or 'custom group'}"
                 + (f" — {len(pairs)} pair(s): {', '.join(sorted(pairs))}"
                    if pairs else ""))
    if dropped:
        lines.append(f"> Dropped (no results): {', '.join(sorted(dropped))}")
    lines.append("")
    lines += [f"> {n}" for n in footnotes(res)]
    lines.append("")
    lines += [
        "`pooled_IvC_natural − sr_xauc = gap` is the metric's argument: a large "
        "gap means the pooled headline was resting on confidence composition, "
        "not on distinguishing correctness within a band.", "",
    ]
    fams = families or sorted(res["family"].unique())
    for f in fams:
        lines += table_for(res, family=f, variant="primary")
    stress = res[res["variant"] == "stress"]
    if len(stress):
        lines += ["## stress variant (train on natural + matched2 jointly)", ""]
        for f in fams:
            lines += table_for(res, family=f, variant="stress")
    return "\n".join(lines) + "\n"
