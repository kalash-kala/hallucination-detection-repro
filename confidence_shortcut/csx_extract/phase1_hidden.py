"""Phase 1: hidden states and s_ext, one teacher-forced forward pass per row.

Runs under sdpa. No attention weights are needed here, so this pass is fast and
linear in sequence length -- and it alone unlocks hs_wide / hs_narrow /
hs_peak_only, the draft's headline families. Phase 2 can follow days later.

Per layer, per segment, one pooled vector is kept:

  all    the LAST token's hidden state. This is the published convention
         (`hidden_states[L][0, -1]`, model_utils.py:176), and matching it is what
         puts newly extracted pairs in the same feature space as the eight qa8
         pairs that arrive through the legacy adapter.
  image  the MEAN over image positions
  text   the MEAN over non-image positions

The `all` asymmetry is deliberate and load-bearing: changing it to a mean would
put every new text pair in a different space from the parity pairs.

Per row the pass also records s_ext = mean answer-token log-probability, the +1
column of every hs_* vector.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from csx_common import registry

from . import config, models, prompts, spans, writer


def _pool(hidden_states, mask: np.ndarray, segment: str, n_layers: int,
          dim: int) -> np.ndarray:
    """[n_layers, dim] pooled per layer for one row.

    `hidden_states` is the HF tuple of length n_layers+1; index 0 is the
    embedding output and is never used, so layer L (1-indexed) is element L.

    Stored fp32, NOT fp16. The models run in bfloat16, which has float32's
    exponent range, and gemma-3 exploits it: its mid/late-layer activations
    reach ~1e5, above fp16's 65504 ceiling. Casting to fp16 turned 927 of the
    first 20 rows' values into +-inf, all in layers 27-38 -- while Qwen (max
    492) and Pixtral (max 121) stayed comfortably in range, so a smoke test on
    either alone would have shown nothing. bfloat16 would be the exact-width
    fix, but numpy has no bfloat16 and the pinned env has no ml_dtypes, so
    fp32 it is: these shards are transient (reduce deletes them) and the
    pooled matrices written afterwards are z-scored back to order 1.
    """
    out = np.empty((n_layers, dim), dtype=np.float32)
    idx = torch.from_numpy(np.flatnonzero(mask))
    for L in range(1, n_layers + 1):
        h = hidden_states[L][0]                    # [S, D]
        if segment == "all":
            v = h[-1]
        else:
            v = h.index_select(0, idx.to(h.device)).float().mean(dim=0)
        out[L - 1] = v.float().cpu().numpy()
    return out


def _s_ext(logits: torch.Tensor, input_ids: torch.Tensor, answer_start: int,
           answer_end: int) -> float:
    """Mean log P over the answer tokens (ALPHA = 1.0).

    Position i's logits predict token i+1, so the distribution over answer token
    t sits at index t-1 -- the off-by-one that silently halves or shifts the
    score if it is got wrong.
    """
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    tgt = input_ids[0, answer_start:answer_end]
    pred = lp[answer_start - 1:answer_end - 1]
    tok_lp = pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    n = max(int(tgt.numel()), 1)
    return float(tok_lp.sum().item() / n ** config.S_EXT_ALPHA)


def run(pair_key: str, *, limit: int | None = None, verbose: bool = True) -> dict:
    """Execute phase 1 for one pair. Writes per-layer shards + updated rows."""
    pair = registry.get(pair_key)
    rows = writer.read_rows(pair_key)

    # `--limit` is destructive, and that is not obvious: this function reads and
    # rewrites the SAME table, so a limited pass leaves rows.parquet holding only
    # those N rows. A later full run then reads the truncated table, extracts N
    # rows, and reports success -- which is exactly what happened after the first
    # advqa smoke test, and it looks identical to a real run in the log.
    #
    # The roster size is recorded by stage 20 as subsample.n_kept, so the two
    # disagreeing is detectable rather than merely regrettable.
    full = writer.load_meta(pair_key).n_kept
    if full is not None and len(rows) < full:
        if not limit:
            raise RuntimeError(
                f"{pair_key}: rows.parquet holds {len(rows)} of {full} rows -- it "
                f"was truncated by an earlier --limit pass. Re-run "
                f"`cli_extract/20_rows.py --pairs {pair_key} --force --run` to "
                f"restore the full roster before extracting.")
        print(f"  [{pair_key}] note: roster is already truncated to {len(rows)} "
              f"rows by an earlier --limit pass", flush=True)

    if limit:
        rows = rows.head(limit).copy()

    loaded = models.load(pair_key, attn_implementation="sdpa")
    n_layers, dim = loaded.n_layers, loaded.hidden_dim
    segs = pair.segments
    n = len(rows)

    acc = {s: np.zeros((n, n_layers, dim), dtype=np.float32) for s in segs}
    s_ext = np.full(n, np.nan, dtype=np.float32)
    seq_len = np.full(n, -1, dtype=np.int32)
    a0 = np.full(n, -1, dtype=np.int32)
    a1 = np.full(n, -1, dtype=np.int32)
    i0 = np.full(n, -1, dtype=np.int32)
    i1 = np.full(n, -1, dtype=np.int32)
    skipped: list[str] = []

    t0 = time.time()
    for r in range(n):
        row = rows.iloc[r]
        inputs, n_prompt, n_answer = models.build_inputs(loaded, row)
        if inputs is None:
            skipped.append(str(row["id"]))
            continue

        built_prompt = inputs["input_ids"][0, :n_prompt].tolist()

        # The definitive gate, run on EVERY row: the prompt we just built must
        # equal the one the generation run recorded, token for token. This is
        # what makes s_ext and the hidden states provably measurements of the
        # context the model actually saw. It subsumes the image-span check --
        # a dropped token_type_ids or a mis-packed batch cannot survive it --
        # and it costs nothing, since the comparison is against a column already
        # sitting in rows.parquet.
        if pair.is_vlm:
            prompts.check(pair, built_prompt, row["prompt_token_ids"],
                          row_id=str(row["id"]))

        sp = spans.build(pair, built_prompt, n_answer)

        with torch.no_grad():
            out = loaded.model(**inputs, output_hidden_states=True, use_cache=False)

        for s in segs:
            acc[s][r] = _pool(out.hidden_states, sp.mask(s), s, n_layers, dim)
        s_ext[r] = _s_ext(out.logits, inputs["input_ids"], sp.answer_start,
                          sp.answer_end)
        seq_len[r], a0[r], a1[r] = sp.seq_len, sp.answer_start, sp.answer_end
        i0[r], i1[r] = sp.image_start, sp.image_end

        del out
        if verbose and (r + 1) % 200 == 0:
            rate = (r + 1) / (time.time() - t0)
            print(f"  [{pair_key}] phase1 {r + 1}/{n} rows  {rate:.1f}/s",
                  flush=True)

    # A row whose greedy answer tokenises to nothing has no answer span to pool
    # over. Dropping it is the only honest option, but rows and shards must be
    # filtered by the SAME positional mask or every downstream feature is
    # misaligned by the number of dropped rows -- a silent, uniform corruption.
    keep = np.ones(n, dtype=bool)
    if skipped:
        keep = ~rows["id"].astype(str).isin(set(skipped)).to_numpy()

    # Checked here, at the source, rather than only at the reduce step: a
    # non-finite hidden state means the storage dtype cannot hold what the model
    # produced, and the useful diagnostic is WHICH layer overflowed. By reduce
    # time the layer identity is lost inside a bucket mean.
    for s in segs:
        bad = ~np.isfinite(acc[s][keep])
        if bad.any():
            layers = sorted({int(L) + 1 for L in np.flatnonzero(bad.any(axis=(0, 2)))})
            raise RuntimeError(
                f"{pair_key}/{s}: {int(bad.sum())} non-finite hidden-state "
                f"values in layer(s) {layers}. The pooled value exceeded the "
                f"storage dtype's range -- see the dtype note in _pool.")

    # Shards are written per (segment, layer) so the reduce step streams one
    # layer at a time instead of holding [N, L, D] in memory.
    for s in segs:
        for L in range(1, n_layers + 1):
            writer.save_shard(pair_key, s, L, acc[s][keep, L - 1, :])

    rows = rows.assign(
        s_ext=s_ext, seq_len=seq_len, answer_start=a0, answer_end=a1,
        image_start=i0, image_end=i1,
    )[keep].reset_index(drop=True)
    # `row` is the index into every array the phases write, so it is renumbered
    # to stay exactly 0..n-1 after the drop.
    rows["row"] = np.arange(len(rows), dtype="int32")
    writer.write_rows(pair_key, rows)

    info = {
        "n_rows": int(len(rows)),
        "skipped_empty_answer": skipped,
        "minutes": round((time.time() - t0) / 60, 1),
        "attn_implementation": "sdpa",
        # Carried so stage 22 can refuse to reduce a smoke subset into features
        # that would look, downstream, like a real pair with a small n.
        "partial_limit": int(limit) if limit else None,
    }
    return info
