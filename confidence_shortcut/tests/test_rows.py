"""The CSV x generations join, and the gates that make a bad roster entry loud.

Every check here corresponds to a way the join could succeed while describing the
wrong data. A partial generations folder yields a short pair with shifted band
proportions; a mismatched question means the two files are different runs that
merely share an id space; a path rewrite that silently no-ops yields a pair that
dies thousands of rows into a GPU pass instead of immediately.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from csx_common import paths, registry
from csx_extract import rows as rows_mod

PAIR = "qwen25vl_advqa"
N = 8


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setitem(paths.ROOTS, "store", tmp_path)
    monkeypatch.setitem(paths.ROOTS, "generations", tmp_path / "gens")
    return tmp_path


def _write_l2(store, pair=PAIR, n=N, questions=None):
    ids = [f"validation::{i}" for i in range(n)]
    qs = questions or [f"question {i}?" for i in range(n)]
    df = pd.DataFrame({
        "pair": [pair] * n,
        "id": ids,
        "question": qs,
        "greedy": [f"answer {i}" for i in range(n)],
        "category": np.tile(["IH", "CH", "IL", "CL"], n // 4),
        "entropy": np.linspace(0, 1, n),
    })
    t = paths.uq_table()
    t.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(t, index=False)
    return ids, qs


def _write_gens(store, folder, ids, questions, images, *, prefix=""):
    d = paths.ROOTS["generations"] / folder
    d.mkdir(parents=True, exist_ok=True)
    with (d / "combined_generations.jsonl").open("w") as fh:
        for sid, q, img in zip(ids, questions, images):
            fh.write(json.dumps({sid: {
                "question": q,
                "image_path": prefix + str(img),
                "prompt_token_ids": [1, 151655, 151655, 2],
                "most_likely_answer": {"response": "yes"},
            }}) + "\n")


def _images(tmp_path, n):
    d = tmp_path / "img"
    d.mkdir(exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"{i}.jpg"
        p.write_bytes(b"x")
        out.append(p)
    return out


def _folder_of(pair=PAIR):
    return registry.get(pair).generations


# ── happy path ───────────────────────────────────────────────────────────────

def test_builds_rows_for_a_vlm_pair(store, tmp_path):
    ids, qs = _write_l2(store)
    imgs = _images(tmp_path, N)
    _write_gens(store, _folder_of(), ids, qs, imgs)

    built = rows_mod.build(PAIR)
    assert built.n == N
    assert list(built.rows["row"]) == list(range(N))
    assert built.rows["image_path"].str.endswith(".jpg").all()
    assert (built.rows["seq_len"] == -1).all()      # phase 1 fills these


def test_row_index_is_contiguous_after_subsampling(store, tmp_path):
    ids, qs = _write_l2(store)
    imgs = _images(tmp_path, N)
    _write_gens(store, _folder_of(), ids, qs, imgs)
    built = rows_mod.build(PAIR, n_target=4)
    assert built.n == 4
    assert list(built.rows["row"]) == [0, 1, 2, 3]
    assert built.n_pool == N


def test_subsample_preserves_cell_proportions(store, tmp_path):
    ids, qs = _write_l2(store)
    imgs = _images(tmp_path, N)
    _write_gens(store, _folder_of(), ids, qs, imgs)
    built = rows_mod.build(PAIR, n_target=4)
    # 2 of each cell in the pool -> 1 of each kept.
    assert built.rows["category"].value_counts().to_dict() == {
        "IH": 1, "CH": 1, "IL": 1, "CL": 1}


# ── the gates ────────────────────────────────────────────────────────────────

def test_missing_l2_names_the_stage_to_run(store):
    with pytest.raises(rows_mod.RowsError, match="00_ingest_uq"):
        rows_mod.build(PAIR)


def test_partial_generations_folder_is_refused(store, tmp_path):
    """Extracting the intersection would drop rows and shift band proportions,
    which is invisible downstream."""
    ids, qs = _write_l2(store)
    imgs = _images(tmp_path, N)
    _write_gens(store, _folder_of(), ids[:-2], qs[:-2], imgs[:-2])
    with pytest.raises(rows_mod.RowsError, match="absent from"):
        rows_mod.build(PAIR)


def test_question_mismatch_is_refused(store, tmp_path):
    ids, qs = _write_l2(store)
    imgs = _images(tmp_path, N)
    bad = list(qs)
    bad[3] = "a completely different question"
    _write_gens(store, _folder_of(), ids, bad, imgs)
    with pytest.raises(rows_mod.RowsError, match="question text disagrees"):
        rows_mod.build(PAIR)


def test_whitespace_differences_are_tolerated(store, tmp_path):
    """The CSV round-trips through pandas, so whitespace is not evidence of a
    mismatch -- only real text differences are."""
    ids, qs = _write_l2(store)
    imgs = _images(tmp_path, N)
    _write_gens(store, _folder_of(), ids, [f"  {q}  " for q in qs], imgs)
    assert rows_mod.build(PAIR).n == N


def test_missing_image_file_is_refused(store, tmp_path):
    ids, qs = _write_l2(store)
    imgs = _images(tmp_path, N)
    imgs[2].unlink()
    _write_gens(store, _folder_of(), ids, qs, imgs)
    with pytest.raises(rows_mod.RowsError, match="do not exist"):
        rows_mod.build(PAIR)


def test_duplicate_id_in_generations_is_refused(store, tmp_path):
    ids, qs = _write_l2(store)
    imgs = _images(tmp_path, N)
    _write_gens(store, _folder_of(), ids, qs, imgs)
    d = paths.ROOTS["generations"] / _folder_of() / "combined_generations.jsonl"
    with d.open("a") as fh:
        fh.write(json.dumps({ids[0]: {"question": qs[0], "image_path": str(imgs[0]),
                                      "prompt_token_ids": [1],
                                      "most_likely_answer": {"response": "y"}}}) + "\n")
    with pytest.raises(rows_mod.RowsError, match="duplicate id"):
        rows_mod.build(PAIR)


# ── the path rewrite ─────────────────────────────────────────────────────────

REMAP_PAIR = "gemma3_12b_okvqa"


def test_path_rewrite_is_applied(store, tmp_path, monkeypatch):
    p = registry.get(REMAP_PAIR)
    assert p.path_rewrite, "this pair is supposed to declare a rewrite"
    src, _ = p.path_rewrite

    ids, qs = _write_l2(store, pair=REMAP_PAIR)
    imgs = _images(tmp_path, N)
    # Store paths under the FOREIGN prefix, and point the rewrite at tmp_path.
    monkeypatch.setattr(
        registry.Pair, "path_rewrite",
        property(lambda self: (src, str(tmp_path / "img"))))
    _write_gens(store, p.generations, ids, qs,
                [f"{src}/{i}.jpg" for i in range(N)])

    built = rows_mod.build(REMAP_PAIR)
    assert built.n == N
    assert all(pp.startswith(str(tmp_path)) for pp in built.rows["image_path"])


def test_rewrite_that_does_not_apply_is_reported_as_such(store, tmp_path):
    """A no-op rewrite is the dangerous case: paths look untouched and every file
    is missing. The error must say the rewrite failed to apply, not just that
    files are absent."""
    p = registry.get(REMAP_PAIR)
    ids, qs = _write_l2(store, pair=REMAP_PAIR)
    src, _ = p.path_rewrite
    _write_gens(store, p.generations, ids, qs,
                [f"{src}/{i}.jpg" for i in range(N)])
    with pytest.raises(rows_mod.RowsError, match="did not apply|files are absent"):
        rows_mod.build(REMAP_PAIR)


def test_registry_rewrite_leaves_foreign_paths_untouched():
    """A path not starting with the declared prefix is returned unchanged rather
    than mangled, so the missing-file gate catches it instead of silently
    producing a plausible-looking wrong path."""
    p = registry.get(REMAP_PAIR)
    assert p.rewrite_path("/somewhere/else/x.jpg") == "/somewhere/else/x.jpg"
    src, dst = p.path_rewrite
    assert p.rewrite_path(f"{src}/coco/a.jpg") == f"{dst}/coco/a.jpg"


# ── the --limit truncation trap ──────────────────────────────────────────────

def test_meta_exposes_roster_size_and_partial_flag():
    """The invariant that makes a truncated roster detectable.

    Phase 1 reads and rewrites the same rows.parquet, so a `--limit N` pass
    leaves the table holding only N rows. A later full run then reads the
    truncated table and reports success on N rows -- indistinguishable, in the
    log, from a real pass. The guard compares the live row count against
    `subsample.n_kept`, which stage 20 records before any phase runs, so both
    numbers must survive on the entry.
    """
    from csx_common.store_schema import EntryMeta

    meta = EntryMeta(raw={
        "schema_version": 1, "pair": "p", "modality": "vlm",
        "segments": ["all"], "n_rows": 20,
        "subsample": {"n_kept": 3000},
        "phase1_pass": {"partial_limit": 20},
    })
    assert meta.n_rows == 20
    assert meta.n_kept == 3000
    assert meta.partial_limit == 20
    assert meta.n_rows < meta.n_kept  # what the guard trips on

    full = EntryMeta(raw={
        "schema_version": 1, "pair": "p", "modality": "vlm",
        "segments": ["all"], "n_rows": 3000,
        "subsample": {"n_kept": 3000},
        "phase1_pass": {"partial_limit": None},
    })
    assert full.partial_limit is None
    assert full.n_rows == full.n_kept


def test_meta_without_subsample_block_does_not_crash_the_guard():
    """A pair that was never subsampled has no n_kept; the guard must skip,
    not raise -- otherwise it would block the pairs it is meant to protect."""
    from csx_common.store_schema import EntryMeta

    meta = EntryMeta(raw={
        "schema_version": 1, "pair": "p", "modality": "text",
        "segments": ["all"], "n_rows": 500,
    })
    assert meta.n_kept is None
    assert meta.partial_limit is None
