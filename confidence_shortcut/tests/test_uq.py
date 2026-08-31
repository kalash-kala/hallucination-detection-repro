"""L2 ingest: column-variant handling, and the gates that must not be skippable.

The gates matter more than the parsing. Orientation (H = LOW entropy) and the
choice of LLM_verdict over `accuracy` are both places where a wrong-but-plausible
reading would flow silently into every downstream number, so each has a test that
feeds it deliberately corrupted input and demands a failure.
"""

from __future__ import annotations

import pandas as pd
import pytest

from csx_common import registry
from csx_probe import uq


def _csv(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _good_frame(n=40, tau=0.5, variant="vlm"):
    """A minimal well-formed run CSV, alternating the four cells."""
    ent, cat, verdict = [], [], []
    for i in range(n):
        hi = i % 2 == 0
        correct = (i // 2) % 2 == 0
        ent.append(0.1 if hi else 0.9)
        cat.append(f"{'correct' if correct else 'incorrect'}_{'high' if hi else 'low'}")
        verdict.append(correct)
    df = pd.DataFrame({
        "id": [f"train::{i}" for i in range(n)],
        "ground_truth": ["['x']"] * n,
        "low_t_generation": ["x"] * n,
        "accuracy": [1.0] * n,
        "n_generations": [str(["a"] * 10)] * n,
        "cluster_assignment_entropy": ent,
        "question": ["q?"] * n,
        "p_true": [float("nan")] * n,
        "LLM_verdict": verdict,
        "DSE_threshold": [tau] * n,
        "DSE_b_entropy": [0 if e <= tau else 1 for e in ent],
        "category_dse": cat,
    })
    if variant == "text":
        df["C_metric"] = 0.4
        df["category"] = cat
    return df


@pytest.fixture
def patched(tmp_path, monkeypatch):
    """Point one registry entry at a synthetic CSV in tmp_path."""
    def install(df, name="fake.csv", pair="llama_sciq"):
        path = _csv(tmp_path, name, df)
        real = registry.get(pair)
        monkeypatch.setattr(type(real), "csv_path", property(lambda self: path))
        return pair
    return install


def test_reads_vlm_variant(patched):
    pair = patched(_good_frame(variant="vlm"))
    got = uq.read_pair(pair)
    assert len(got.rows) == 40
    assert set(got.rows["category"]) == set(registry.CATS)
    assert got.rows["c_metric"].isna().all()          # absent in the 12-col shape
    assert (got.rows["n_gen"] == 10).all()


def test_reads_text_variant_with_cmetric(patched):
    pair = patched(_good_frame(variant="text"))
    got = uq.read_pair(pair)
    assert (got.rows["c_metric"] == 0.4).all()
    assert got.rows["category_cmetric"].notna().all()


def test_uq_metrics_are_nan_when_m20_has_not_run(patched):
    """A pair with no uq_metrics.parquet yet must still ingest cleanly -- M20
    landing per-pair, not all-at-once, must never block L2 ingest."""
    pair = patched(_good_frame(variant="vlm"))
    got = uq.read_pair(pair)
    for c in uq.UQ_METRIC_COLUMNS:
        assert c in got.rows.columns
        assert got.rows[c].isna().all()


def test_uq_metrics_left_join_by_id_with_partial_coverage(patched, monkeypatch):
    """M20 may be mid-run (checkpointed rows only): rows present in
    uq_metrics.parquet get real values, rows absent stay NaN -- never dropped,
    never mismatched by position."""
    pair = patched(_good_frame(n=6, variant="vlm"))
    m = pd.DataFrame({
        "id": ["train::1", "train::4"],
        **{c: [0.1, 0.2] for c in uq.UQ_METRIC_COLUMNS},
    })
    _write_metrics(m, monkeypatch)
    got = uq.read_pair(pair)
    have = got.rows.set_index("id")
    assert have.loc["train::1", "num_set"] == pytest.approx(0.1)
    assert have.loc["train::4", "num_set"] == pytest.approx(0.2)
    assert pd.isna(have.loc["train::0", "num_set"])
    assert pd.isna(have.loc["train::2", "num_set"])


def _write_metrics(m, monkeypatch):
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    p = d / "uq_metrics.parquet"
    m.to_parquet(p, index=False)
    monkeypatch.setattr("csx_probe.uq.paths.uq_metrics", lambda pair: p)
    return p


def test_drops_leading_unnamed_index(patched):
    df = _good_frame()
    df.insert(0, "Unnamed: 0", range(len(df)))
    pair = patched(df)
    got = uq.read_pair(pair)
    assert "Unnamed: 0" not in got.rows.columns
    assert len(got.rows) == 40


def test_band_and_correct_are_derived_consistently(patched):
    pair = patched(_good_frame())
    r = uq.read_pair(pair).rows
    assert (r.loc[r["band"] == "H", "entropy"] <= r["tau"].iloc[0]).all()
    assert (r["correct"] == r["category"].isin(["CH", "CL"])).all()


def test_split_src_parsed(patched):
    df = _good_frame()
    df.loc[0, "id"] = "validation::7"
    pair = patched(df)
    r = uq.read_pair(pair).rows
    assert r.loc[0, "split_src"] == "validation"
    assert r.loc[0, "src_index"] == 7


def test_generations_sidecar_only_on_request(patched):
    pair = patched(_good_frame())
    assert uq.read_pair(pair).generations is None
    g = uq.read_pair(pair, with_generations=True).generations
    assert len(g) == 40 and len(g.iloc[0]["generations"]) == 10


# ── the gates ────────────────────────────────────────────────────────────────

def test_rejects_inverted_band_orientation(patched):
    """H must mean LOW entropy. If a future file flipped that, every conclusion
    about the conflict cell would invert, so this has to be fatal."""
    df = _good_frame()
    df["category_dse"] = df["category_dse"].str.replace("_high", "_TMP") \
                                           .str.replace("_low", "_high") \
                                           .str.replace("_TMP", "_low")
    pair = patched(df)
    with pytest.raises(uq.IngestError, match="entropy <= tau"):
        uq.read_pair(pair)


def test_rejects_b_entropy_disagreeing_with_threshold(patched):
    df = _good_frame()
    df.loc[0, "DSE_b_entropy"] = 1 - df.loc[0, "DSE_b_entropy"]
    pair = patched(df)
    with pytest.raises(uq.IngestError, match="DSE_b_entropy"):
        uq.read_pair(pair)


def test_rejects_verdict_disagreeing_with_category(patched):
    """Correctness follows LLM_verdict, not `accuracy` -- verified on real data,
    where the two genuinely disagree."""
    df = _good_frame()
    df.loc[0, "LLM_verdict"] = not df.loc[0, "LLM_verdict"]
    pair = patched(df)
    with pytest.raises(uq.IngestError, match="LLM_verdict"):
        uq.read_pair(pair)


def test_rejects_multiple_thresholds(patched):
    df = _good_frame()
    df.loc[0, "DSE_threshold"] = 0.7
    pair = patched(df)
    with pytest.raises(uq.IngestError, match="one DSE threshold"):
        uq.read_pair(pair)


def test_rejects_boundary_category(patched):
    """'boundary' rows must never be silently bucketed into one of the four
    cells -- 01_make_dse_splits.py asserts their absence too."""
    df = _good_frame()
    df.loc[0, "category_dse"] = "boundary"
    pair = patched(df)
    with pytest.raises(uq.IngestError, match="unmappable"):
        uq.read_pair(pair)


def test_rejects_duplicate_ids(patched):
    df = _good_frame()
    df.loc[1, "id"] = df.loc[0, "id"]
    pair = patched(df)
    with pytest.raises(uq.IngestError, match="duplicate ids"):
        uq.read_pair(pair)


def test_rejects_missing_required_column(patched):
    df = _good_frame().drop(columns=["DSE_threshold"])
    pair = patched(df)
    with pytest.raises(uq.IngestError, match="missing required columns"):
        uq.read_pair(pair)


def test_summarise_shape(patched):
    pair = patched(_good_frame())
    s = uq.summarise(uq.read_pair(pair).rows)
    assert len(s) == 1
    assert set(registry.CATS) <= set(s.columns)
    assert s.loc[0, "n"] == 40


# ── against the real files ───────────────────────────────────────────────────

@pytest.mark.needs_artifacts
def test_every_active_pair_ingests_cleanly():
    from csx_common import paths
    if not paths.ROOTS["uq_csv_dir"].is_dir():
        pytest.skip("run CSVs not present on this machine")
    rows, _, problems = uq.build(strict=False)
    assert not problems, problems
    assert rows["pair"].nunique() == len(registry.resolve())
    counts = rows.pivot_table(index="pair", columns="category", values="id",
                              aggfunc="size", fill_value=0)
    thin = counts[(counts < 10).any(axis=1)]
    assert thin.empty, f"pairs with a cell under 10 rows cannot support the arms:\n{thin}"
