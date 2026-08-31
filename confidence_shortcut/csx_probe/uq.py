"""L2 ingest: every run CSV -> one row-level parquet.

This is the cheapest and most useful layer in the store. It needs no GPU and no
internal states, so it covers every registered pair on day one -- which means arm
construction, the entropy_only verification and the alpha-ladder leak table are
available for VLM pairs long before their extraction finishes.

The run CSVs come in several column shapes (12 / 14 with C_metric+category / 15
with a leading unnamed index). Normalising them here means nothing downstream has
to know which variant a pair came from.

Row identity is `id`, of the form `train::N` / `validation::N` -- the *source
dataset's* split and index, not the experiment's train/test split. The experiment
split is assigned later, by the arm builders.

Gates (verified against llama_sciq, qwen25vl_vqav2 and llama_8b_nq before being
written as assertions):
  * DSE_b_entropy == 0  <=>  entropy <= tau  <=>  category ends in `_high`.
    This pins the orientation the whole study depends on: H = LOW entropy =
    HIGH confidence. Several conclusions invert if it is wrong, so it is checked
    per pair rather than assumed.
  * correctness comes from LLM_verdict, NOT from `accuracy` -- the two disagree
    on real rows, and the published categories follow the verdict.
  * tau is a single value per pair.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

from csx_common import paths, registry

# M20: the 7 text-only NLI metrics, left-joined in read_pair() when
# paths.uq_metrics(pair) exists -- NaN for a pair whose M20 pass hasn't run
# yet, so ingest never blocks on it.
UQ_METRIC_COLUMNS = ("num_set", "lexical_sim", "sum_eigv", "degree",
                    "eccentricity", "luq", "snne")

# Canonical output column order for the row table.
COLUMNS = [
    "pair", "model", "dataset", "modality",
    "id", "split_src", "src_index",
    "question", "ground_truth", "greedy",
    "category", "correct", "band",
    "entropy", "tau",
    "llm_verdict", "accuracy", "p_true",
    "n_gen", "c_metric", "category_cmetric",
    *UQ_METRIC_COLUMNS,
]

# Source -> canonical renames. Anything not listed is dropped.
_RENAME = {
    "low_t_generation": "greedy",
    "cluster_assignment_entropy": "entropy",
    "DSE_threshold": "tau",
    "DSE_b_entropy": "b_entropy",
    "category_dse": "category_long",
    "LLM_verdict": "llm_verdict",
    "C_metric": "c_metric",
    "category": "category_cmetric",
}


class IngestError(RuntimeError):
    """A pair's CSV violated an invariant the rest of the study relies on."""


@dataclass(frozen=True)
class PairIngest:
    pair: str
    rows: pd.DataFrame
    generations: pd.DataFrame | None  # id -> the 10 sampled answer strings


def _as_bool(s: pd.Series) -> pd.Series:
    """LLM_verdict arrives as bool, or as the strings 'True'/'False'."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def _split_id(ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    """`train::1278` -> ('train', 1278). The prefix is the SOURCE dataset split,
    not the experiment split -- both appear inside one file."""
    parts = ids.astype(str).str.split("::", n=1, expand=True)
    if parts.shape[1] != 2:
        raise IngestError("ids are not all of the form '<split>::<index>'")
    return parts[0], pd.to_numeric(parts[1], errors="coerce").astype("Int64")


def _parse_generations(raw: pd.Series) -> pd.Series:
    """`n_generations` is a stringified Python list of the 10 sampled answers."""
    def one(v):
        if isinstance(v, list):
            return v
        if not isinstance(v, str):
            return []
        try:
            out = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
        return list(out) if isinstance(out, (list, tuple)) else []
    return raw.map(one)


def read_pair(pair_key: str, *, with_generations: bool = False) -> PairIngest:
    """Normalise one run CSV, enforcing the invariants above."""
    p = registry.get(pair_key)
    path = p.csv_path
    if not path.exists():
        raise IngestError(f"{pair_key}: CSV not found at {path}")

    raw = pd.read_csv(path)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed:")])
    df = raw.rename(columns=_RENAME)

    required = {"id", "greedy", "entropy", "tau", "b_entropy",
                "category_long", "llm_verdict"}
    missing = required - set(df.columns)
    if missing:
        raise IngestError(f"{pair_key}: CSV missing required columns {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    out["pair"] = pair_key
    out["model"] = p.model.key
    out["dataset"] = p.dataset.key
    out["modality"] = p.model.modality
    out["id"] = df["id"].astype(str)
    out["split_src"], out["src_index"] = _split_id(out["id"])
    out["question"] = df.get("question", pd.Series("", index=df.index)).astype(str)
    out["ground_truth"] = df.get("ground_truth", pd.Series("", index=df.index)).astype(str)
    out["greedy"] = df["greedy"].astype(str)

    cat_long = df["category_long"].astype(str)
    out["category"] = cat_long.map(registry.CAT_SHORT)
    out["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
    out["tau"] = pd.to_numeric(df["tau"], errors="coerce")
    out["llm_verdict"] = _as_bool(df["llm_verdict"])
    out["accuracy"] = pd.to_numeric(df.get("accuracy"), errors="coerce")
    out["p_true"] = pd.to_numeric(df.get("p_true"), errors="coerce")
    out["c_metric"] = pd.to_numeric(df.get("c_metric"), errors="coerce")
    out["category_cmetric"] = df.get("category_cmetric", pd.Series(pd.NA, index=df.index))

    gens = _parse_generations(df["n_generations"]) if "n_generations" in df else None
    out["n_gen"] = gens.map(len).astype("Int64") if gens is not None else pd.NA

    # Derived, and the two axes everything downstream conditions on.
    out["correct"] = ~cat_long.str.startswith("incorrect")
    out["band"] = np.where(cat_long.str.endswith("_high"), "H", "L")

    metrics_path = paths.uq_metrics(pair_key)
    if metrics_path.exists():
        m = pd.read_parquet(metrics_path)
        # left join, not concat: M20 may still be mid-run (checkpointed rows
        # only) or ahead of a stale roster, so id is the join key, not order.
        out = out.merge(m[["id", *UQ_METRIC_COLUMNS]], on="id", how="left")

    _gate(pair_key, out, df["b_entropy"], cat_long)

    out = out.reindex(columns=COLUMNS)
    gen_df = None
    if with_generations and gens is not None:
        gen_df = pd.DataFrame({"pair": pair_key, "id": out["id"], "generations": gens})
    return PairIngest(pair=pair_key, rows=out, generations=gen_df)


def _gate(pair: str, out: pd.DataFrame, b_entropy: pd.Series, cat_long: pd.Series) -> None:
    unknown = sorted(set(cat_long) - set(registry.CAT_SHORT))
    if unknown:
        # 'boundary' rows exist upstream in principle and must never be silently
        # bucketed -- 01_make_dse_splits.py asserts their absence too.
        raise IngestError(f"{pair}: unmappable category_dse values {unknown}")

    if out["entropy"].isna().any():
        raise IngestError(f"{pair}: {int(out['entropy'].isna().sum())} rows with no entropy")

    taus = out["tau"].dropna().unique()
    if len(taus) != 1:
        raise IngestError(f"{pair}: expected one DSE threshold, got {len(taus)}")

    # Orientation: H is LOW entropy. Asserted, not assumed.
    hi_from_entropy = out["entropy"] <= float(taus[0])
    if not hi_from_entropy.eq(b_entropy.astype(int) == 0).all():
        raise IngestError(f"{pair}: DSE_b_entropy disagrees with entropy <= tau")
    if not hi_from_entropy.eq(out["band"] == "H").all():
        raise IngestError(
            f"{pair}: category band disagrees with entropy <= tau -- "
            f"H must mean LOW entropy (high confidence)"
        )

    # Correctness follows the LLM verdict. `accuracy` is a different quantity and
    # genuinely disagrees; using it would silently relabel rows.
    if not out["correct"].eq(out["llm_verdict"].astype(bool)).all():
        raise IngestError(f"{pair}: category correctness disagrees with LLM_verdict")

    if out["id"].duplicated().any():
        raise IngestError(f"{pair}: duplicate ids")


def build(pair_keys: list[str] | None = None, *, with_generations: bool = False,
          strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None, list[dict]]:
    """Ingest many pairs. Returns (rows, generations|None, problems).

    With `strict=False` a failing pair is recorded in `problems` and skipped
    rather than aborting the run -- useful while data is still landing.
    """
    pairs = registry.resolve(pair_keys)
    frames, gens, problems = [], [], []
    for p in pairs:
        try:
            got = read_pair(p.key, with_generations=with_generations)
        except IngestError as exc:
            if strict:
                raise
            problems.append({"pair": p.key, "error": str(exc)})
            continue
        frames.append(got.rows)
        if got.generations is not None:
            gens.append(got.generations)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    gen_df = pd.concat(gens, ignore_index=True) if gens else None
    return rows, gen_df, problems


def write(rows: pd.DataFrame, generations: pd.DataFrame | None = None) -> list:
    """Write L2. Generations go to a sidecar, not the main table: they are bulky
    (10 answer strings per row) and are only needed by the 10-sample extraction,
    whereas the row table is read by every arm build."""
    written = []
    target = paths.uq_table()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(target, index=False)
    written.append(target)
    if generations is not None and len(generations):
        for pair, g in generations.groupby("pair", sort=False):
            side = target.parent / "generations" / f"{pair}.parquet"
            side.parent.mkdir(parents=True, exist_ok=True)
            g.drop(columns=["pair"]).to_parquet(side, index=False)
            written.append(side)
    return written


def load(pair_keys: list[str] | None = None) -> pd.DataFrame:
    """Read L2 back, optionally restricted to some pairs."""
    t = paths.uq_table()
    if not t.exists():
        raise FileNotFoundError(
            f"L2 not built yet ({t}); run cli_probe/00_ingest_uq.py --run"
        )
    df = pd.read_parquet(t)
    if pair_keys:
        df = df[df["pair"].isin(pair_keys)].reset_index(drop=True)
    return df


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-pair cell counts and tau -- the table that shows, at a glance, whether
    a pair has enough rows in every cell to support the four arms."""
    g = rows.groupby("pair", sort=False)
    out = g.agg(
        model=("model", "first"),
        dataset=("dataset", "first"),
        modality=("modality", "first"),
        n=("id", "size"),
        tau=("tau", "first"),
    )
    counts = rows.pivot_table(index="pair", columns="category", values="id",
                              aggfunc="size", fill_value=0)
    for c in registry.CATS:
        out[c] = counts[c] if c in counts else 0
    out["pct_incorrect"] = (100 * (out["IH"] + out["IL"]) / out["n"]).round(1)
    return out.reset_index()
