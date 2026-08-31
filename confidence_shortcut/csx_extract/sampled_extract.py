"""Extract internal states for the 10-sample router / SR-xAUC tier (Part C, M19).

One eager forward pass per unique `(id, text)` row of the manifest built by
`sampled_manifest.py` yields both hidden states and attention in the same pass
-- the legacy `24_extract_sampled.py` design, which halves GPU time relative to
running the greedy pipeline's two separate sdpa/eager sweeps.

**Nothing full-width is ever held per row**, but the accumulator across ALL
rows is still large: `diag_topk`/`lap_topk` are `[n, n_layers, n_q, top_k]`
fp16, which for a 91k-row pair is tens of GB summed over 3 segments x 5 arrays.
Holding that as anonymous RAM (`np.zeros`) is what OOM-killed a 90k/91.5k-row
`qwen25vl_okvqa` run one page from the finish line, with nothing written to
disk because the single `sampled_writer.save_*` call sits after the loop.

**So the accumulators are disk-backed memmaps (`_tmp/*.mmap`), not `np.zeros`.**
Two things follow: (1) their pages are file-backed and reclaimable, so the
kernel can evict them under memory pressure instead of invoking the OOM
killer, and (2) writes already live on disk as the loop runs. A `_progress.json`
checkpoint, updated every `CHECKPOINT_EVERY` rows right after an `.flush()`,
lets a rerun resume at the last checkpointed row instead of starting over --
`extract()` checks for a matching checkpoint (same manifest row count and array
shapes) before allocating fresh buffers. The memmaps are folded into the final
compressed `.npz` and the `_tmp/` directory removed only once `n_done == n`.

**Bucket layers and regions are read from the pair's own `peaks.json`, not
re-derived.** The peak search is a validation-set classifier fit on greedy
TRAIN rows; running it again on sampled rows would be circular (there is no
natural 1:1 correctness label per sampled generation) and would put the sampled
features in a different layer space from the greedy ones they must combine
with (`greedy_mean_std`). So this reads the existing `raw/<pair>/hs/peaks.json`
and reuses its bucket definitions unchanged.

**Per-layer z-score statistics ARE recomputed, not read back.** `reduce_hidden.py`
computes them over greedy-train rows but never persists them, then deletes the
raw shards they were computed from. This module recomputes the identical
statistic (mean/std over the same `_train_mask` split, same layers) by running
one extra sdpa pass, so the sampled features are z-scored on the same footing
as the greedy ones despite the raw greedy shards no longer existing on disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from csx_common import paths, registry
from csx_common.store_schema import HS_SCHEMES

from . import config, models, phase1_hidden, phase2_attention, prompts, \
    reduce_hidden, sampled_writer, spans, writer

CHECKPOINT_EVERY = 200  # rows between memmap flush + progress.json update


class ExtractionError(Exception):
    """The pair's sampled inputs are inconsistent with what extraction needs."""


def _tmp_dir(pair_key: str) -> Path:
    return paths.sampled_dir(pair_key) / "_tmp"


def _progress_path(pair_key: str) -> Path:
    return _tmp_dir(pair_key) / "progress.json"


def _mmap_path(pair_key: str, seg: str, name: str) -> Path:
    return _tmp_dir(pair_key) / f"{seg}__{name}.mmap"


def _shape_sig(hs_shapes: dict, diag_shapes: dict) -> dict:
    """JSON-safe signature of every buffer's shape+dtype, to detect a stale
    checkpoint left behind by a different `n`, model, or top_k/sink_k."""
    def sig(d):
        return {k: [list(shape), str(dtype)] for k, (shape, dtype) in d.items()}
    return {"hs": sig(hs_shapes), "diag": sig(diag_shapes)}


def _open_buffers(pair_key: str, segs: tuple, n: int, hidden_dim: int,
                  n_layers: int, n_q: int, top_k: int, sink_k: int,
                  *, verbose: bool = True) -> tuple[dict, dict, int]:
    """Open (or resume) the disk-backed accumulators. Returns
    `(hs_buf, diag_buf, n_done)` where `n_done` is how many rows are already
    checkpointed -- 0 for a fresh start."""
    hs_shapes = {f"{seg}__{scheme}": ((n, 2 * hidden_dim + 1), np.float16)
                for seg in segs for scheme in HS_SCHEMES}
    diag_dtypes = {"attn_topk": np.float16, "lap_topk": np.float16,
                   "sink_topk": np.float16, "sink_vnorm_topk": np.float32,
                   "attn_logdet": np.float32}
    diag_shapes = {}
    for seg in segs:
        for name, dt in diag_dtypes.items():
            k = sink_k if "sink" in name else top_k
            shape = ((n, n_layers, n_q) if name == "attn_logdet"
                     else (n, n_layers, n_q, k))
            diag_shapes[f"{seg}__{name}"] = (shape, dt)

    tmp = _tmp_dir(pair_key)
    tmp.mkdir(parents=True, exist_ok=True)
    sig = _shape_sig(hs_shapes, diag_shapes)
    prog_path = _progress_path(pair_key)
    n_done = 0
    if prog_path.exists():
        prev = json.loads(prog_path.read_text())
        if prev.get("sig") == sig and prev.get("n_total") == n:
            n_done = prev.get("n_done", 0)
            if verbose and n_done:
                print(f"  [{pair_key}] resuming from checkpoint: "
                     f"{n_done}/{n} rows already extracted", flush=True)
        elif verbose:
            print(f"  [{pair_key}] stale checkpoint (shape/n mismatch) -- "
                 f"starting this pair's buffers over", flush=True)

    def open_one(seg_name: str, shape: tuple, dt) -> np.memmap:
        p = _mmap_path(pair_key, *seg_name.split("__", 1))
        m = "r+" if (n_done and p.exists()) else "w+"
        return np.memmap(p, dtype=dt, mode=m, shape=shape)

    hs_buf: dict[str, dict[str, np.memmap]] = {seg: {} for seg in segs}
    for key, (shape, dt) in hs_shapes.items():
        seg, scheme = key.split("__", 1)
        hs_buf[seg][scheme] = open_one(key, shape, dt)

    diag_buf: dict[str, dict[str, np.memmap]] = {seg: {} for seg in segs}
    for key, (shape, dt) in diag_shapes.items():
        seg, name = key.split("__", 1)
        diag_buf[seg][name] = open_one(key, shape, dt)

    if not prog_path.exists() or n_done == 0:
        prog_path.write_text(json.dumps({"sig": sig, "n_total": n, "n_done": 0}))

    return hs_buf, diag_buf, n_done


def _checkpoint(pair_key: str, hs_buf: dict, diag_buf: dict, sig: dict, n: int,
                n_done: int, n_skipped: int, n_short_guard: int) -> None:
    for d in (hs_buf, diag_buf):
        for seg in d.values():
            for arr in seg.values():
                arr.flush()
    tmp = _progress_path(pair_key).with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "sig": sig, "n_total": n, "n_done": n_done,
        "n_skipped": n_skipped, "n_short_guard": n_short_guard}))
    tmp.replace(_progress_path(pair_key))


def _peaks(pair_key: str) -> dict:
    import json
    from csx_common import paths
    p = paths.raw_dir(pair_key) / "hs" / "peaks.json"
    if not p.exists():
        raise ExtractionError(
            f"{pair_key}: no {p} -- run the greedy pipeline (stages 20-22) "
            f"first. Sampled extraction reuses the greedy pass's peak layers "
            f"and bucket definitions so the two feature spaces combine.")
    return json.loads(p.read_text())


def _needed_layers(peaks: dict) -> list[int]:
    wide = peaks["wide"]
    return sorted(set(wide["mid"]) | set(wide["late"]))


# ── phase A: layer z-score statistics, over greedy-train rows ────────────────

def compute_layer_stats(pair_key: str, needed_layers: list[int], segs: tuple,
                        *, verbose: bool = True
                        ) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """{segment: {layer: (mu, sd)}}, from ONE sdpa pass over greedy-train rows.

    One row's forward pass is pooled for every segment at once -- pooling is
    cheap relative to the forward pass itself, so computing all segments' stats
    from a single sweep over train rows costs the same as computing one
    segment's, rather than `len(segs)` times as much.

    Streamed as float64 accumulators so memory stays O(dim), not O(n_train x dim)
    -- reduce_hidden.py could afford to hold a whole layer's shard because it
    read it from disk; here every row costs a forward pass, so nothing is kept
    beyond the running sum/sumsq.
    """
    pair = registry.get(pair_key)
    rows = writer.read_rows(pair_key)
    tr_mask = reduce_hidden._train_mask(rows)
    train_rows = rows[tr_mask].reset_index(drop=True)

    loaded = models.load(pair_key, attn_implementation="sdpa")
    n_layers, dim = loaded.n_layers, loaded.hidden_dim

    sums = {s: {L: np.zeros(dim, dtype=np.float64) for L in needed_layers}
            for s in segs}
    sqs = {s: {L: np.zeros(dim, dtype=np.float64) for L in needed_layers}
           for s in segs}
    n = 0
    t0 = time.time()
    for r in range(len(train_rows)):
        row = train_rows.iloc[r]
        inputs, n_prompt, n_answer = models.build_inputs(loaded, row)
        if inputs is None:
            continue
        built_prompt = inputs["input_ids"][0, :n_prompt].tolist()
        if pair.is_vlm:
            prompts.check(pair, built_prompt, row["prompt_token_ids"],
                          row_id=str(row["id"]))
        sp = spans.build(pair, built_prompt, n_answer)
        with torch.no_grad():
            out = loaded.model(**inputs, output_hidden_states=True,
                               use_cache=False)
        for s in segs:
            pooled = phase1_hidden._pool(out.hidden_states, sp.mask(s), s,
                                         n_layers, dim)
            for L in needed_layers:
                v = pooled[L - 1].astype(np.float64)
                sums[s][L] += v
                sqs[s][L] += v * v
        n += 1
        del out
        if verbose and n % 500 == 0:
            rate = n / (time.time() - t0)
            print(f"  [{pair_key}] stats {n}/{len(train_rows)} rows "
                  f"{rate:.1f}/s", flush=True)

    if n == 0:
        raise ExtractionError(f"{pair_key}: no usable train rows")
    out_stats: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for s in segs:
        mu = {L: (sums[s][L] / n).astype(np.float32) for L in needed_layers}
        sd = {L: (np.sqrt(np.maximum(sqs[s][L] / n - (sums[s][L] / n) ** 2, 0.0))
                 .astype(np.float32) + 1e-6) for L in needed_layers}
        out_stats[s] = {L: (mu[L], sd[L]) for L in needed_layers}
    del loaded
    torch.cuda.empty_cache()
    return out_stats


def _bucket_vector(pooled: np.ndarray, buckets: dict, stats: dict,
                   s_ext: float) -> np.ndarray:
    """pooled: [n_layers, dim] for one row -> [2*dim+1] hs_* feature vector,
    the identical mean-of-z-scores construction as reduce_hidden.py."""
    parts = []
    for region in ("mid", "late"):
        lls = buckets[region]
        z = np.mean([(pooled[L - 1] - stats[L][0]) / stats[L][1] for L in lls],
                    axis=0)
        parts.append(z.astype(np.float16))
    return np.concatenate(parts + [np.array([s_ext], dtype=np.float16)])


# ── phase B: one combined eager pass per unique row ───────────────────────────

def extract(pair_key: str, *, limit: int | None = None,
           verbose: bool = True) -> dict:
    import pandas as pd
    from csx_common import paths

    pair = registry.get(pair_key)
    unique = pd.read_parquet(paths.sampled_unique(pair_key))
    if limit:
        unique = unique.head(limit).copy()
    n = len(unique)

    # Join each unique (id, text) row against the greedy roster for the
    # context (image_path, question, prompt_token_ids) that does not change
    # across a row's 10 samples -- only the answer text does.
    greedy = writer.read_rows(pair_key).set_index("id")
    ctx = greedy.loc[unique["id"]].reset_index(drop=True)

    peaks = _peaks(pair_key)
    needed = _needed_layers(peaks)
    segs = pair.segments

    if verbose:
        print(f"[{pair_key}] computing layer stats for segments={segs} "
              f"({len(needed)} layers, one pass)", flush=True)
    stats_by_seg = compute_layer_stats(pair_key, needed, segs, verbose=verbose)
    layer_stats_out: dict[str, np.ndarray] = {}
    for seg in segs:
        for L in needed:
            layer_stats_out[f"{seg}_mu_L{L}"] = stats_by_seg[seg][L][0]
            layer_stats_out[f"{seg}_sd_L{L}"] = stats_by_seg[seg][L][1]
    sampled_writer.save_layer_stats(pair_key, layer_stats_out)

    loaded = models.load(pair_key, attn_implementation="eager")
    n_layers = loaded.n_layers
    n_q, n_kv = loaded.n_q_heads, loaded.n_kv_heads
    group = max(n_q // max(n_kv, 1), 1)
    kv_of_head = np.arange(n_q) // group
    top_k, sink_k = config.EXTRACT_TOP_K, config.SINK_K

    hs_buf, diag_buf, n_done = _open_buffers(
        pair_key, segs, n, loaded.hidden_dim, n_layers, n_q, top_k, sink_k,
        verbose=verbose)
    sig = json.loads(_progress_path(pair_key).read_text())["sig"]

    prog = json.loads(_progress_path(pair_key).read_text())
    n_skipped = prog.get("n_skipped", 0)
    n_short_guard = prog.get("n_short_guard", 0)

    capture = phase2_attention.ValueNormCapture(loaded.model, n_layers)
    t0 = time.time()
    for r in range(n_done, n):
        u_row = unique.iloc[r]
        c_row = ctx.iloc[r]
        synth = {"image_path": c_row["image_path"], "question": c_row["question"],
                 "answer": u_row["text"]}
        inputs, n_prompt, n_answer = models.build_inputs(loaded, synth)
        if inputs is None:
            n_skipped += 1
            continue
        built_prompt = inputs["input_ids"][0, :n_prompt].tolist()
        if pair.is_vlm:
            prompts.check(pair, built_prompt, c_row["prompt_token_ids"],
                          row_id=f"{u_row['id']}#u{r}")
        sp = spans.build(pair, built_prompt, n_answer)
        if sp.seq_len < top_k:
            n_short_guard += 1

        with capture, torch.no_grad():
            out = loaded.model(**inputs, output_hidden_states=True,
                               output_attentions=True, use_cache=False)

        s_ext = phase1_hidden._s_ext(out.logits, inputs["input_ids"],
                                     sp.answer_start, sp.answer_end)
        attn_d = phase2_attention.attention_diagonal(out.attentions).float().cpu().numpy()
        lap_d = phase2_attention.laplacian_diagonal(out.attentions).float().cpu().numpy()
        vnorm = capture.norms().cpu().numpy()

        for seg in segs:
            mask = sp.mask(seg)
            pooled = phase1_hidden._pool(out.hidden_states, mask, seg,
                                         n_layers, loaded.hidden_dim)
            for scheme in HS_SCHEMES:
                buckets = peaks["segments"][seg]["buckets"][scheme]
                hs_buf[seg][scheme][r] = _bucket_vector(
                    pooled, buckets, stats_by_seg[seg], s_ext)
            red = phase2_attention._reduce(attn_d, lap_d, vnorm, mask,
                                           kv_of_head, top_k, sink_k)
            for k, v in red.items():
                diag_buf[seg][k][r] = v

        del out
        if (r + 1) % CHECKPOINT_EVERY == 0 or r + 1 == n:
            _checkpoint(pair_key, hs_buf, diag_buf, sig, n, r + 1,
                       n_skipped, n_short_guard)
            if verbose:
                rate = (r + 1 - n_done) / (time.time() - t0)
                print(f"  [{pair_key}] extract {r + 1}/{n} unique rows  "
                     f"{rate:.2f}/s  (checkpointed)", flush=True)

    for seg in segs:
        # np.memmap arrays are accepted directly by np.savez_compressed --
        # this reads the finished data back off disk, not from RAM.
        sampled_writer.save_hs(pair_key, seg, hs_buf[seg])
        sampled_writer.save_diag(pair_key, seg, diag_buf[seg])

    import shutil
    shutil.rmtree(_tmp_dir(pair_key), ignore_errors=True)

    return {
        "n_unique": int(n),
        "n_skipped_empty_answer": int(n_skipped),
        "n_short_seq_guard": int(n_short_guard),
        "minutes": round((time.time() - t0) / 60, 1),
        "segments": list(segs),
        "top_k": top_k,
        "sink_k": sink_k,
        "partial_limit": int(limit) if limit else None,
    }
