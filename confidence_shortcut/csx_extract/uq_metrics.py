"""Extract the 7 text-only NLI uncertainty metrics (Part C, M20).

`num_set`, `lexical_sim`, `sum_eigv`, `degree`, `eccentricity`, `luq`, `snne` --
each a function of `(question, 10 sampled-answer strings)` through a DeBERTa
entailment matrix. Reuses `scripts/snne_baseline/snne_core.py` and
`run_baselines.per_question_scores` unchanged; this module is the VLM-cohort
data loader and checkpointed driver around that already-proven QA code.

**Text-only, no vision tower.** The 10 answer strings for `id` come straight
from the source run CSV's `n_generations` column (a parsed Python-list literal,
same field `sampled_manifest.py` reads for M19) -- no image is loaded, no VLM
forward pass runs. Settings are frozen to the QA run: entailment similarity,
`variant=only_denom`, `temp=1.0`, `condition_on_question=True` (baked into
`per_question_scores`, not re-specified here).

**Checkpointed like `sampled_extract.py`, but row-structured rather than
array-structured.** Each row costs ~90 batched DeBERTa forward passes (10
generations -> 10*9 directed pairs), which is small compared to a VLM pass but
still adds up to double-digit GPU-hours over the full cohort -- worth surviving
a crash rather than restarting. Because the output here is 7 floats per row
rather than a big array, checkpointing is a periodically-flushed partial
parquet per worker shard instead of a memmap.

**Row-sharded across `n_workers` processes on one GPU, not just one loop.**
DeBERTa-v2-xlarge is ~3.6G of weights against 80G of VRAM and each row's
compute is tiny (short text, no vision tower) -- the "GPU stages don't
parallelize" convention elsewhere in this repo is calibrated for the VLM
extraction jobs (15-50G/instance, 1-2 fit per GPU), which does not describe
this job. `--n-workers` spawns that many independent `EntailmentDeberta`
instances via `loky` (fresh processes, safe with CUDA -- never fork after
`torch.cuda` is touched), each scoring a `rows[i::n_workers]` slice with its
own checkpoint file, so a `n_workers`-change between runs cannot corrupt
another worker's resume state. Cross-GPU fan-out is still the shell layer's
job (one `CUDA_VISIBLE_DEVICES`-pinned process per GPU, `--n-workers` inside
each), matching how stage 26 is already launched.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from csx_common import paths, registry

from .sampled_manifest import _parse_generations

# scripts/snne_baseline/{snne_core,run_baselines}.py are script-style modules
# (bare `import snne_core`, no package __init__.py), so importing them means
# putting their own directory on sys.path, the same way snne_core.py does for
# the vendored SNNE repo -- not a `from scripts.snne_baseline import ...`.
_SNNE_BASELINE_DIR = str(
    Path(__file__).resolve().parents[2] / "scripts" / "snne_baseline")
if _SNNE_BASELINE_DIR not in sys.path:
    sys.path.insert(0, _SNNE_BASELINE_DIR)

METHODS = ("num_set", "lexical_sim", "sum_eigv", "degree", "eccentricity",
          "luq", "snne")

CHECKPOINT_EVERY = 100  # rows between partial-parquet flushes


class UQMetricsError(Exception):
    """The pair's source CSV is inconsistent with what this module needs."""


def _tmp_dir(pair_key: str) -> Path:
    return paths.raw_dir(pair_key) / "_tmp_uq_metrics"


def _shard_checkpoint_path(pair_key: str, shard_id: int, n_shards: int) -> Path:
    return _tmp_dir(pair_key) / f"partial_shard{shard_id}of{n_shards}.parquet"


def _all_shard_checkpoints(pair_key: str) -> list[Path]:
    d = _tmp_dir(pair_key)
    return sorted(d.glob("partial_shard*.parquet")) if d.exists() else []


def _all_done_ids(pair_key: str) -> set[str]:
    """Union of every id already scored, across ALL shard files present --
    not just the current run's `n_shards`. A `--n-workers` change between
    runs must not lose a previous run's progress: each worker still only
    APPENDS to its own file for this run's sharding, but every worker skips
    an id any prior sharding already finished."""
    ids: set[str] = set()
    for p in _all_shard_checkpoints(pair_key):
        ids |= set(pd.read_parquet(p, columns=["id"])["id"])
    return ids


def _load_rows(pair_key: str) -> pd.DataFrame:
    """id, question, generations(list[10]) -- roster-filtered, same join
    `sampled_manifest.build` performs, so a row missing here is missing there
    too rather than silently diverging."""
    pair = registry.get(pair_key)
    rows_path = paths.raw_rows(pair_key)
    if not rows_path.exists():
        raise UQMetricsError(
            f"{pair_key}: no {rows_path} -- run the greedy extraction "
            f"(cli_extract/20-23) first.")
    roster = pd.read_parquet(rows_path)
    ids = set(roster["id"].astype(str))

    csv_path = pair.csv_path
    if not csv_path.exists():
        raise UQMetricsError(f"{pair_key}: missing source CSV {csv_path}")
    df = pd.read_csv(csv_path)
    if "n_generations" not in df.columns:
        raise UQMetricsError(
            f"{pair_key}: {csv_path} has no `n_generations` column -- M20 "
            f"needs the 10 sampled-generation strings, not just the greedy "
            f"answer.")
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(ids)].reset_index(drop=True)
    missing = ids - set(df["id"])
    if missing:
        raise UQMetricsError(
            f"{pair_key}: {len(missing)} of {len(ids)} roster ids are absent "
            f"from {csv_path}, e.g. {sorted(missing)[:3]}")

    out = pd.DataFrame({
        "id": df["id"],
        "question": df["question"].astype(str),
    })
    out["generations"] = [_parse_generations(g) for g in df["n_generations"]]
    bad = out["generations"].map(len) != 10
    if bad.any():
        raise UQMetricsError(
            f"{pair_key}: {int(bad.sum())} rows do not have exactly 10 "
            f"generations, e.g. id {out.loc[bad, 'id'].iloc[0]!r}")
    return out


def _flush_shard(pair_key: str, shard_id: int, n_shards: int,
                 done: pd.DataFrame) -> None:
    _tmp_dir(pair_key).mkdir(parents=True, exist_ok=True)
    p = _shard_checkpoint_path(pair_key, shard_id, n_shards)
    tmp = p.with_suffix(".parquet.tmp")
    done.to_parquet(tmp, index=False)
    tmp.replace(p)


def _extract_shard(pair_key: str, rows_shard: pd.DataFrame, shard_id: int,
                   n_shards: int, global_done_ids: set[str],
                   verbose: bool) -> int:
    """One worker's slice. Own model instance, own checkpoint file. Imports
    are local so a `loky`-spawned worker initializes CUDA fresh rather than
    the parent process ever touching it -- safe regardless of backend."""
    import evaluate
    from rouge_score import tokenizers

    import snne_core as sc
    from run_baselines import per_question_scores

    own_path = _shard_checkpoint_path(pair_key, shard_id, n_shards)
    own_done = (pd.read_parquet(own_path) if own_path.exists()
               else pd.DataFrame(columns=["id", *METHODS]))
    todo = rows_shard[~rows_shard["id"].isin(global_done_ids)]
    if not len(todo):
        return 0

    deberta = sc.EntailmentDeberta()
    rouge = evaluate.load("rouge", keep_in_memory=True)
    tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize

    tag = f"{pair_key}#{shard_id}/{n_shards}"
    new_rows: list[dict] = []
    t0 = time.time()
    n_this_run = 0
    for i, row in enumerate(todo.itertuples(index=False)):
        s = per_question_scores(row.question, row.generations, deberta,
                                rouge, tokenizer)
        # `per_question_scores` casts every method to float() except
        # `lexical_sim`, which stays a 0-dim torch tensor (the sum in
        # `compute_lexical_similarity` accumulates over a tensor similarity
        # matrix) -- pyarrow rejects tensors outright, so coerce defensively
        # here rather than patching the shared/vendored script.
        new_rows.append({"id": row.id,
                         **{k: float(v) for k, v in s.items()}})
        n_this_run += 1

        if len(new_rows) >= CHECKPOINT_EVERY or i + 1 == len(todo):
            batch = pd.DataFrame(new_rows)
            own_done = batch if own_done.empty else pd.concat(
                [own_done, batch], ignore_index=True)
            _flush_shard(pair_key, shard_id, n_shards, own_done)
            new_rows = []
            if verbose:
                rate = n_this_run / (time.time() - t0)
                print(f"  [{tag}] {len(own_done)}/{len(rows_shard)} rows  "
                     f"{rate:.2f}/s  (checkpointed)", flush=True)
    return n_this_run


def extract(pair_key: str, *, limit: int | None = None,
           verbose: bool = True, n_workers: int = 1) -> dict:
    """Score every roster row of `pair_key` on all 7 metrics, checkpointed.

    `n_workers` > 1 spawns that many independent worker processes on the
    (single, already-selected) GPU, each with its own DeBERTa instance and
    checkpoint file -- see the module docstring for why this is safe and
    worthwhile for this specific model, unlike the big VLM extraction jobs.

    Returns summary info; writes the final table to
    `paths.uq_metrics(pair_key)` and removes the checkpoint dir.
    """
    rows = _load_rows(pair_key)
    if limit:
        rows = rows.head(limit).copy()
    n = len(rows)

    global_done = _all_done_ids(pair_key)
    n_already = int(rows["id"].isin(global_done).sum())
    if verbose and n_already:
        print(f"  [{pair_key}] resuming from checkpoint: "
             f"{n_already}/{n} rows already scored", flush=True)

    t0 = time.time()
    n_workers = max(1, n_workers)
    shards = [rows.iloc[i::n_workers].reset_index(drop=True)
             for i in range(n_workers)]
    if n_workers == 1:
        n_scored_this_run = _extract_shard(pair_key, shards[0], 0, 1,
                                           global_done, verbose)
    else:
        from joblib import Parallel, delayed
        counts = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_extract_shard)(pair_key, shards[i], i, n_workers,
                                    global_done, verbose)
            for i in range(n_workers))
        n_scored_this_run = int(sum(counts))

    done_frames = [pd.read_parquet(p) for p in _all_shard_checkpoints(pair_key)]
    done = (pd.concat(done_frames, ignore_index=True)
           if done_frames else pd.DataFrame(columns=["id", *METHODS]))
    done = done.drop_duplicates(subset="id", keep="last")
    still_missing = set(rows["id"]) - set(done["id"])
    if still_missing:
        raise UQMetricsError(
            f"{pair_key}: {len(still_missing)} rows were never scored after "
            f"the run, e.g. {sorted(still_missing)[:3]} -- a worker likely "
            f"crashed silently; rerun to resume from its checkpoint.")

    out_path = paths.uq_metrics(pair_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    done.sort_values("id").reset_index(drop=True).to_parquet(tmp, index=False)
    tmp.replace(out_path)

    import shutil
    shutil.rmtree(_tmp_dir(pair_key), ignore_errors=True)

    return {
        "n_rows": int(n),
        "n_scored_this_run": n_scored_this_run,
        "minutes": round((time.time() - t0) / 60, 1),
        "methods": list(METHODS),
        "n_workers": n_workers,
        "partial_limit": int(limit) if limit else None,
    }
