"""The sampled aggregation schemes, `cloud` in particular.

`cloud` is the one scheme whose value is a *derived geometry* rather than a
per-dimension statistic, so a transcription slip in it produces a plausible
10-column matrix rather than an error. The reference-equivalence test below is
therefore the point of this file: it re-implements the QA original
(`25_sampled_router_ovr.aggregate(variant="cloud")`) literally, row by row, and
demands our batched version agree with it to float64 precision.

Everything here runs on synthetic arrays -- `cloud_block` takes the unique
matrix, the slot map and the greedy matrix directly, so none of this needs a
store, a GPU, or an extracted pair.
"""

from __future__ import annotations

import numpy as np
import pytest

from csx_probe.store import sampled as sp


# ── a literal transcription of the published implementation ──────────────────
# Deliberately NOT refactored: it is only useful as an oracle if it reads like
# the original. Vectorising it here would be reproducing our own possible bug.
def _reference_cloud(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """`25_sampled_router_ovr.py:152-175`, verbatim, for one row."""
    n = H.shape[0]
    X = H.astype(np.float64)
    G = X @ X.T
    dg2 = np.diag(G)
    d2 = np.maximum(dg2[:, None] + dg2[None, :] - 2.0 * G, 0.0)
    iu = np.triu_indices(n, 1)
    dist = np.sqrt(d2[iu])
    nrm = np.sqrt(np.maximum(dg2, 0)) + 1e-12
    cos = (G / np.outer(nrm, nrm))[iu]
    mu = X.mean(0)
    Gc = (G - np.outer(np.ones(n), X @ mu) - np.outer(X @ mu, np.ones(n))
          + (mu @ mu))
    ev = np.linalg.eigvalsh((Gc + Gc.T) / 2.0 / max(n - 1, 1))[::-1]
    ev = np.log1p(np.maximum(
        np.concatenate([ev, np.zeros(sp.CLOUD_EIG)])[:sp.CLOUD_EIG], 0))
    gg = g.astype(np.float64)
    dgv = np.sqrt(np.maximum(dg2 - 2.0 * (X @ gg) + gg @ gg, 0.0))
    return np.concatenate([
        [dist.mean(), dist.std(), cos.mean(), cos.std()],
        ev, [float(ev.sum()), dgv.mean(), dgv.std()],
    ])


def _synth(n_rows=37, n_slots=10, n_feat=23, n_unique=61, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_unique, n_feat)).astype(np.float32)
    idx = rng.integers(0, n_unique, size=(n_rows, n_slots))
    greedy = rng.normal(size=(n_rows, n_feat)).astype(np.float32)
    return U, idx, greedy


def test_cloud_matches_reference_implementation():
    """The load-bearing test: batched == the published per-row original."""
    U, idx, greedy = _synth()
    got = sp.cloud_block(U, idx, greedy)
    want = np.array([_reference_cloud(U[idx[i]], greedy[i])
                     for i in range(idx.shape[0])])
    assert got.shape == (idx.shape[0], sp.CLOUD_DIM)
    np.testing.assert_allclose(got, want.astype(np.float32), rtol=0, atol=2e-5)


def test_cloud_is_ten_dimensional_regardless_of_feature_width():
    """The whole claim about `cloud` is that its width is free of the family's.

    If this ever fails, the "competitive at 10 dimensions against 14k" framing
    in the write-up is no longer describing the feature being computed.
    """
    assert sp.CLOUD_DIM == 10
    for n_feat in (5, 23, 500):
        U, idx, greedy = _synth(n_feat=n_feat)
        assert sp.cloud_block(U, idx, greedy).shape[1] == 10


def test_cloud_chunking_does_not_change_the_answer(monkeypatch):
    """Chunk size is a memory knob and must never be visible in the output."""
    U, idx, greedy = _synth(n_rows=53)
    full = sp.cloud_block(U, idx, greedy)
    monkeypatch.setattr(sp, "_CLOUD_CHUNK_BYTES", 1)      # forces chunk == 1
    np.testing.assert_array_equal(sp.cloud_block(U, idx, greedy), full)


def test_cloud_of_identical_slots_is_a_degenerate_cloud():
    """10 copies of one answer: no spread, and that must show as exact zeros.

    This is the high-confidence extreme the feature exists to detect, so a
    non-zero dispersion here would mean the statistic is picking up numerical
    noise rather than semantic spread.
    """
    rng = np.random.default_rng(1)
    U = rng.normal(size=(4, 12)).astype(np.float32)
    idx = np.zeros((6, 10), dtype=int)                   # every slot -> row 0
    greedy = np.repeat(U[:1], 6, axis=0)                 # greedy == that row
    out = sp.cloud_block(U, idx, greedy)
    dist_mean, dist_std, _, cos_std = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    assert np.allclose(dist_mean, 0.0, atol=1e-5)
    assert np.allclose(dist_std, 0.0, atol=1e-5)
    assert np.allclose(cos_std, 0.0, atol=1e-5)
    # eigenvalues of a rank-0 centred Gram, and zero displacement from greedy
    assert np.allclose(out[:, 4:4 + sp.CLOUD_EIG], 0.0, atol=1e-5)
    assert np.allclose(out[:, -2:], 0.0, atol=1e-5)


def test_cloud_separates_tight_from_diffuse_clouds():
    """The feature must be monotone in actual spread, not merely non-constant."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=(1, 16))
    tight = (base + 0.01 * rng.normal(size=(10, 16))).astype(np.float32)
    diffuse = (base + 3.00 * rng.normal(size=(10, 16))).astype(np.float32)
    U = np.concatenate([tight, diffuse]).astype(np.float32)
    idx = np.array([np.arange(10), np.arange(10, 20)])
    greedy = np.repeat(base.astype(np.float32), 2, axis=0)
    out = sp.cloud_block(U, idx, greedy)
    assert out[0, 0] < out[1, 0], "mean pairwise distance must order the clouds"
    assert out[0, 4] < out[1, 4], "leading Gram eigenvalue must order them too"


def test_cloud_requires_at_least_two_slots():
    U, idx, greedy = _synth(n_slots=1)
    with pytest.raises(sp.SampledError, match="undefined at n_slots=1"):
        sp.cloud_block(U, idx, greedy)


def test_cloud_is_registered_as_a_scheme():
    assert "cloud" in sp.SCHEMES
    with pytest.raises(sp.SampledError, match="unknown scheme"):
        sp.aggregate("nonexistent_pair", "hs_wide", scheme="not_a_scheme")
