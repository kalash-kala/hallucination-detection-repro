"""Contract 1: a well-formed store entry validates, and every malformed one fails
with a *named* reason.

These are the tests that let the two components be developed apart. csx_extract
can be written against `make_entry` here without any experiment code existing,
and csx_probe can be written against the same fixture without a GPU.

The failure cases are the valuable half. Each one encodes a way an entry could be
wrong while looking fine -- an inverted span, diagonals whose L disagrees with the
model, a missing attn_logdet -- and demands that the verifier catch it rather than
letting it flow into the numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from csx_common import store_schema as ss
from csx_common.store_schema import ContractError, EntryMeta

from csx_extract.config import SINK_K

N, L, H, K = 12, 4, 3, 50
DIM = 2 * 8 + 1


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway store root, so nothing touches the real one."""
    from csx_common import paths
    monkeypatch.setitem(paths.ROOTS, "store", tmp_path)
    return tmp_path


def make_entry(store, pair="qwen25vl_vqav2", *, modality="vlm",
               segments=("all", "image", "text"), phase1=True, phase2=True,
               n=N, layers=L, top_k=K, mutate_rows=None, mutate_meta=None,
               drop_diag_key=None, diag_L=None):
    """Write a synthetic but contract-shaped L0 entry."""
    from csx_common import paths
    d = paths.raw_dir(pair)
    (d / "hs").mkdir(parents=True, exist_ok=True)
    (d / "diag").mkdir(parents=True, exist_ok=True)

    meta = ss.new_meta(
        pair=pair,
        model={"key": pair.split("_")[0], "hf_id": "fake/model", "layers": layers,
               "n_q_heads": H, "n_kv_heads": 1, "hidden_dim": 8},
        dataset=pair.split("_")[-1],
        modality=modality,
        prompt_template="chat" if modality == "vlm" else "ranking",
        segments=list(segments),
        n_rows=n, n_pool=n, subsample=None, top_k=top_k, sink_k=10,
        extractor_version="test",
    )
    meta["phase1"]["done"] = phase1
    meta["phase2"]["done"] = phase2
    if mutate_meta:
        mutate_meta(meta)
    (d / "meta.json").write_text(json.dumps(meta))

    is_vlm = modality == "vlm"
    rows = pd.DataFrame({
        "id": [f"train::{i}" for i in range(n)],
        "row": np.arange(n, dtype="int32"),
        "category": np.tile(["IH", "CH", "IL", "CL"], n // 4).astype(object),
        "entropy": np.linspace(0.0, 1.0, n),
        "s_ext": np.zeros(n, dtype="float32"),
        "seq_len": np.full(n, 100, dtype="int32"),
        "answer_start": np.full(n, 80, dtype="int32"),
        "answer_end": np.full(n, 90, dtype="int32"),
        "image_start": np.full(n, 0 if is_vlm else -1, dtype="int32"),
        "image_end": np.full(n, 60 if is_vlm else -1, dtype="int32"),
        "image_path": ["/img.jpg" if is_vlm else ""] * n,
    })
    if mutate_rows:
        rows = mutate_rows(rows)
    rows.to_parquet(d / "rows.parquet", index=False)

    for seg in segments:
        if phase1:
            np.savez(d / "hs" / f"{seg}.npz",
                     **{s: np.zeros((n, DIM), dtype="float16") for s in ss.HS_SCHEMES})
            (d / "hs" / "peaks.json").write_text(json.dumps({seg: {"mid": 2, "late": 3}}))
        if phase2:
            LL = diag_L if diag_L is not None else layers
            # The sink arrays are stored at SINK_K, not top_k -- see
            # phase2_attention._reduce, which the fixture must mirror.
            arrays = {k: np.zeros(
                (n, LL, H, SINK_K if k in ss.DIAG_SINK_KEYS else top_k),
                dtype="float16") for k in ss.DIAG_TOPK_KEYS}
            arrays["attn_logdet"] = np.zeros((n, LL, H), dtype="float32")
            if drop_diag_key:
                arrays.pop(drop_diag_key)
            np.savez(d / "diag" / f"{seg}.npz", **arrays)
    return pair


# ── the happy path ───────────────────────────────────────────────────────────

def test_well_formed_entry_validates(store):
    from csx_extract.verify import verify_entry
    pair = make_entry(store)
    assert verify_entry(pair, check_l2=False) == []


def test_text_entry_validates(store):
    from csx_extract.verify import verify_entry
    pair = make_entry(store, pair="llama_sciq", modality="text", segments=("all",))
    assert verify_entry(pair, check_l2=False) == []


def test_phase1_only_is_usable(store):
    """A pair with hidden states but no attention pass is valid: it simply has the
    three hs_* families and none of the spectral ones. This is what lets probing
    start before the expensive eager pass finishes."""
    from csx_extract.verify import verify_entry
    pair = make_entry(store, phase2=False)
    assert verify_entry(pair, check_l2=False) == []
    meta = EntryMeta.load(store / "raw" / pair / "meta.json")
    assert set(meta.available_families()) == set(ss.HS_SCHEMES)


def test_neither_phase_done_is_flagged(store):
    from csx_extract.verify import verify_entry
    pair = make_entry(store, phase1=False, phase2=False)
    assert any("phase" in str(p) for p in verify_entry(pair, check_l2=False))


# ── failures, each with a named reason ───────────────────────────────────────

def test_missing_entry(store):
    from csx_extract.verify import verify_entry
    probs = verify_entry("llama_sciq", check_l2=False)
    assert len(probs) == 1 and probs[0].check == "exists"


def test_unknown_schema_version_is_refused(store):
    """A newer extractor may have changed what a field means; a best-effort read
    would be quietly wrong, so this is a hard stop."""
    pair = make_entry(store, mutate_meta=lambda m: m.update(schema_version=99))
    with pytest.raises(ContractError, match="schema v99"):
        EntryMeta.load(store / "raw" / pair / "meta.json")


def test_row_index_not_contiguous(store):
    from csx_extract.verify import verify_entry
    def bad(df):
        df.loc[0, "row"] = 999
        return df
    pair = make_entry(store, mutate_rows=bad)
    assert any("0..n-1" in str(p) for p in verify_entry(pair, check_l2=False))


def test_row_count_disagrees_with_meta(store):
    from csx_extract.verify import verify_entry
    pair = make_entry(store, mutate_meta=lambda m: m.update(n_rows=N + 5))
    assert any("n_rows" in str(p) for p in verify_entry(pair, check_l2=False))


def test_inverted_answer_span(store):
    from csx_extract.verify import verify_entry
    def bad(df):
        df.loc[0, "answer_start"] = 95
        df.loc[0, "answer_end"] = 90
        return df
    pair = make_entry(store, mutate_rows=bad)
    assert any(p.check == "spans" for p in verify_entry(pair, check_l2=False))


def test_image_span_overlapping_answer(store):
    """Image tokens precede the answer. If they overlapped, the `image` and `text`
    segment features would both be wrong while looking plausible."""
    from csx_extract.verify import verify_entry
    def bad(df):
        df["image_end"] = 95
        return df
    pair = make_entry(store, mutate_rows=bad)
    assert any("overlapping" in str(p) for p in verify_entry(pair, check_l2=False))


def test_text_pair_must_not_declare_image_span(store):
    from csx_extract.verify import verify_entry
    def bad(df):
        df["image_start"] = 0
        df["image_end"] = 10
        return df
    pair = make_entry(store, pair="llama_sciq", modality="text",
                      segments=("all",), mutate_rows=bad)
    assert any("image_start/end = -1" in str(p)
               for p in verify_entry(pair, check_l2=False))


def test_missing_attn_logdet_is_caught_and_explained(store):
    """The single most dangerous omission: attnlogdet is a mean over ALL
    positions, so it cannot be rebuilt from the top-k arrays. Without this check
    the family would be silently wrong rather than missing."""
    from csx_extract.verify import verify_entry
    pair = make_entry(store, drop_diag_key="attn_logdet")
    probs = [str(p) for p in verify_entry(pair, check_l2=False)]
    assert any("attn_logdet" in p and "cannot be derived" in p for p in probs)


def test_missing_topk_key(store):
    from csx_extract.verify import verify_entry
    pair = make_entry(store, drop_diag_key="sink_vnorm_topk")
    assert any("sink_vnorm_topk" in str(p) for p in verify_entry(pair, check_l2=False))


def test_diag_layers_disagree_with_model(store):
    from csx_extract.verify import verify_entry
    pair = make_entry(store, diag_L=L + 1)
    assert any("model.layers" in str(p) for p in verify_entry(pair, check_l2=False))


def test_topk_width_disagrees_with_meta(store):
    from csx_extract.verify import verify_entry
    pair = make_entry(store, top_k=25, mutate_meta=lambda m: m.update(top_k=50))
    assert any("top-k width" in str(p) for p in verify_entry(pair, check_l2=False))


def test_segments_must_match_modality(store):
    """A VLM entry with only `all` has thrown away the image/text split, and it is
    not recoverable without re-extracting."""
    from csx_extract.verify import verify_entry
    pair = make_entry(store, segments=("all",))
    assert any(p.check == "segments" for p in verify_entry(pair, check_l2=False))


def test_phase1_done_but_hs_absent(store):
    from csx_extract.verify import verify_entry
    pair = make_entry(store)
    (store / "raw" / pair / "hs" / "all.npz").unlink()
    assert any(p.check == "hs" for p in verify_entry(pair, check_l2=False))


def test_l2_cross_check_flags_unknown_ids(store, monkeypatch):
    from csx_common import paths
    from csx_extract.verify import verify_entry
    pair = make_entry(store)
    t = paths.uq_table()
    t.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pair": [pair] * 3, "id": ["train::0", "train::1", "train::2"]}) \
        .to_parquet(t, index=False)
    assert any(p.check == "l2" for p in verify_entry(pair, check_l2=True))


def test_cli_exit_codes(store, capsys):
    from csx_extract.verify import main
    pair = make_entry(store)
    assert main(["verify", "--pair", pair, "--no-l2"]) == 0
    (store / "raw" / pair / "rows.parquet").unlink()
    assert main(["verify", "--pair", pair, "--no-l2"]) == 1
