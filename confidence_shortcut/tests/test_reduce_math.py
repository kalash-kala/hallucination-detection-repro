"""The phase-2 in-loop reduction, tested as pure numpy.

`_reduce` is where [L,H,S] becomes the stored [L,H,k] arrays. It runs on the GPU
box inside a hot loop, but it is ordinary array math, so it is pinned here with
no torch and no model.

The properties that matter are the ones that would otherwise be invisible: that
attn_logdet really is a full-span mean (and so genuinely cannot be rebuilt from
the top-k arrays), that the sink feature ranks and gathers with a single index,
and that a segment mask selects from the full-length diagonal rather than
recomputing on a slice.
"""

from __future__ import annotations

import numpy as np
import pytest

from csx_extract.phase2_attention import _reduce

L, H, S = 2, 3, 20
TOP_K, SINK_K = 5, 4


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    attn = rng.random((L, H, S)).astype(np.float32)
    lap = rng.random((L, H, S)).astype(np.float32)
    vnorm = rng.random((L, 1, S)).astype(np.float32) + 1.0   # 1 kv head
    kv_of_head = np.zeros(H, dtype=int)
    return attn, lap, vnorm, kv_of_head


def _all_mask():
    return np.ones(S, dtype=bool)


def test_shapes_and_keys():
    a, p, v, kv = _inputs()
    out = _reduce(a, p, v, _all_mask(), kv, TOP_K, SINK_K)
    assert set(out) == {"attn_topk", "lap_topk", "sink_topk",
                        "sink_vnorm_topk", "attn_logdet"}
    assert out["attn_topk"].shape == (L, H, TOP_K)
    assert out["sink_topk"].shape == (L, H, SINK_K)
    assert out["attn_logdet"].shape == (L, H)


def test_topk_is_descending():
    a, p, v, kv = _inputs()
    out = _reduce(a, p, v, _all_mask(), kv, TOP_K, SINK_K)
    d = np.diff(out["attn_topk"].astype(np.float32), axis=-1)
    assert (d <= 1e-6).all()


def test_attn_logdet_is_a_full_span_mean_not_a_topk_statistic():
    """The reason the store contract makes attn_logdet a separate required key.
    If it happened to equal a function of the top-k values, storing only top-k
    would be fine; it does not, so it would silently break `attnlogdet`."""
    a, p, v, kv = _inputs()
    out = _reduce(a, p, v, _all_mask(), kv, TOP_K, SINK_K)
    want = np.log(np.clip(a, 1e-12, None)).mean(axis=-1)
    assert np.allclose(out["attn_logdet"], want, atol=1e-6)

    # And it is genuinely NOT recoverable from the stored top-k slice.
    from_topk = np.log(np.clip(out["attn_topk"].astype(np.float32), 1e-12, None)
                       ).mean(axis=-1)
    assert not np.allclose(from_topk, want, atol=1e-3)


def test_sink_ranks_by_score_and_gathers_value_norms_at_the_same_ranks():
    """feature = s_i * ||V_i|| at the SAME position i. Sorting the two arrays
    independently would pair the largest score with the largest norm regardless
    of whether they occur at the same token -- a plausible-looking wrong answer."""
    a, p, v, kv = _inputs()
    out = _reduce(a, p, v, _all_mask(), kv, TOP_K, SINK_K)

    D = a + p
    idx = np.argsort(-D, axis=-1)[:, :, :SINK_K]
    want_d = np.take_along_axis(D, idx, axis=-1)
    vn = v[:, kv, :]
    want = want_d * np.take_along_axis(vn, idx, axis=-1)
    assert np.allclose(out["sink_vnorm_topk"].astype(np.float32), want, atol=1e-2)

    # The independent-sort version differs, which is what makes this a real test.
    wrong = (np.sort(D, axis=-1)[:, :, ::-1][:, :, :SINK_K]
             * np.sort(vn, axis=-1)[:, :, ::-1][:, :, :SINK_K])
    assert not np.allclose(wrong, want, atol=1e-2)


def test_sink_topk_is_the_score_without_the_norm():
    a, p, v, kv = _inputs()
    out = _reduce(a, p, v, _all_mask(), kv, TOP_K, SINK_K)
    D = a + p
    want = np.sort(D, axis=-1)[:, :, ::-1][:, :, :SINK_K]
    assert np.allclose(out["sink_topk"].astype(np.float32), want, atol=1e-2)


def test_segment_mask_selects_from_the_full_diagonal():
    """Segment features must be a selection from the full-length diagonal, not a
    recomputation on a slice -- the Laplacian's denominator is position-dependent,
    so the two are different numbers."""
    a, p, v, kv = _inputs()
    mask = np.zeros(S, dtype=bool)
    mask[3:11] = True
    out = _reduce(a, p, v, mask, kv, TOP_K, SINK_K)
    want = np.sort(a[:, :, 3:11], axis=-1)[:, :, ::-1][:, :, :TOP_K]
    assert np.allclose(out["attn_topk"].astype(np.float32), want, atol=1e-2)


def test_short_segment_is_edge_padded_to_full_width():
    """A segment with fewer positions than k still yields a fixed-width feature,
    so every row of a pair has the same dimensionality."""
    a, p, v, kv = _inputs()
    mask = np.zeros(S, dtype=bool)
    mask[:2] = True                      # 2 positions, k = 5
    out = _reduce(a, p, v, mask, kv, TOP_K, SINK_K)
    assert out["attn_topk"].shape == (L, H, TOP_K)
    tail = out["attn_topk"][:, :, 1:].astype(np.float32)
    assert np.allclose(tail, tail[:, :, :1], atol=1e-3)


def test_gqa_broadcast_maps_query_heads_to_kv_heads():
    """With n_q=4 and n_kv=2, heads 0,1 read kv 0 and heads 2,3 read kv 1."""
    n_q, n_kv = 4, 2
    rng = np.random.default_rng(1)
    a = rng.random((L, n_q, S)).astype(np.float32)
    p = rng.random((L, n_q, S)).astype(np.float32)
    v = np.stack([np.full((L, S), 10.0), np.full((L, S), 20.0)],
                 axis=1).astype(np.float32)      # [L, 2, S]
    kv_of_head = np.arange(n_q) // (n_q // n_kv)
    assert list(kv_of_head) == [0, 0, 1, 1]

    out = _reduce(a, p, v, _all_mask(), kv_of_head, TOP_K, SINK_K)
    ratio = (out["sink_vnorm_topk"].astype(np.float32)
             / np.clip(out["sink_topk"].astype(np.float32), 1e-6, None))
    assert np.allclose(ratio[:, :2], 10.0, atol=0.5)
    assert np.allclose(ratio[:, 2:], 20.0, atol=0.5)


def test_stored_dtypes_match_the_contract():
    """fp16 only where the value is provably bounded.

    attn/lap/sink are attention diagonals in [0, ~2] and top-k keeps the largest
    entries, so fp16 cannot overflow or flush them. sink_vnorm_topk multiplies by
    ||V||, which carries the model's raw activation scale -- the same quantity
    that overflowed fp16 on gemma-3 in phase 1 -- so it is fp32, as is
    attn_logdet.
    """
    a, p, v, kv = _inputs()
    out = _reduce(a, p, v, _all_mask(), kv, TOP_K, SINK_K)
    for k in ("attn_topk", "lap_topk", "sink_topk"):
        assert out[k].dtype == np.float16, k
    for k in ("sink_vnorm_topk", "attn_logdet"):
        assert out[k].dtype == np.float32, k


def test_gemma_scale_value_norms_survive_storage():
    """A value norm at gemma-3's observed activation scale must round-trip.

    Phase 1 measured |h| up to 65408 on gemma-3-12b, right at fp16's 65504
    ceiling. sink x ||V|| at that scale overflows fp16 to inf; this pins that the
    stored array keeps it finite and accurate.
    """
    a, p, v, kv = _inputs()
    v = v * 6e4
    out = _reduce(a, p, v, _all_mask(), kv, TOP_K, SINK_K)
    sv = out["sink_vnorm_topk"]
    assert np.isfinite(sv).all()
    assert sv.max() > 65504, "test no longer exercises the fp16 ceiling"
