"""The registry must be a complete, unambiguous census of what is on disk.

These tests are the reason `configs/pairs.yaml` can be trusted as the single
source of truth: if a CSV appears, disappears, or is renamed, one of them fails
rather than the roster quietly drifting out of sync with the data.
"""

from __future__ import annotations

import pytest

from csx_common import paths, registry
from csx_common import cohorts as cohorts_mod
from csx_probe import config


def _csvs_on_disk() -> set[str]:
    d = paths.ROOTS["uq_csv_dir"]
    if not d.is_dir():
        pytest.skip(f"run CSVs not present at {d}")
    return {p.name for p in d.glob("uncertainty_run_*_dse_output.csv")}


def test_every_csv_is_classified():
    """Every run CSV is either a pair or explicitly out of scope. No silent gaps."""
    on_disk = _csvs_on_disk()
    claimed = {p.csv for p in registry.pairs().values()}
    excluded = set(registry.out_of_scope_csvs())
    unclassified = on_disk - claimed - excluded
    assert not unclassified, (
        "CSVs on disk that are neither a registered pair nor listed under "
        f"out_of_scope in pairs.yaml: {sorted(unclassified)}"
    )


def test_active_pairs_have_their_csv():
    """A pair marked active must actually have its file; a missing file means the
    entry should be `status: pending` instead."""
    on_disk = _csvs_on_disk()
    missing = {
        p.key: p.csv
        for p in registry.pairs().values()
        if p.status == "active" and p.csv not in on_disk
    }
    assert not missing, f"active pairs whose CSV is absent: {missing}"


def test_pending_pairs_are_genuinely_absent():
    """Once pending data lands, the entry must be flipped to active -- otherwise
    it stays invisible to every default selector."""
    on_disk = _csvs_on_disk()
    arrived = {
        p.key: p.csv
        for p in registry.pairs().values()
        if p.status == "pending" and p.csv in on_disk
    }
    assert not arrived, (
        "these pairs are marked pending but their CSV has arrived -- remove the "
        f"`status: pending` line in pairs.yaml: {arrived}"
    )


def test_no_duplicate_csv_claims():
    seen: dict[str, str] = {}
    for p in registry.pairs().values():
        assert p.csv not in seen, f"{p.csv} claimed by both {seen[p.csv]} and {p.key}"
        seen[p.csv] = p.key


def test_alias_map_is_unambiguous():
    """Built eagerly so a duplicate alias is a load-time error, not last-one-wins."""
    amap = registry.alias_to_model()
    assert amap["qwen_14b"] == "qwen3_14b", (
        "qwen_14b in the nq files is Qwen3-14B, not a Qwen2.5 variant -- "
        "confirmed against the nq__Qwen__Qwen3-14B__* run directory"
    )
    assert amap["llama_8b"] == "llama"
    assert amap["gemma_27b"] == "gemma3_27b"


def test_vlm_pairs_declare_image_provenance():
    """`image_path` is the one field no run CSV carries, so a VLM pair needs
    either a generations folder or a loader branch to reconstruct it from."""
    for p in registry.pairs().values():
        if not p.is_vlm:
            continue
        assert p.generations or p.dataset.loader_branch, (
            f"{p.key}: VLM pair with neither a generations folder nor a "
            f"loader_branch -- image_path would be unrecoverable"
        )


def test_text_pairs_need_no_generations():
    """Text extraction is CSV-driven: question, greedy answer and the 10 sampled
    strings are all in the run CSV. This is why nq needed no transfers."""
    for p in registry.pairs().values():
        if not p.is_vlm:
            assert not p.needs_generations(), p.key


def test_segments_by_modality():
    for p in registry.pairs().values():
        assert p.segments == (("all", "image", "text") if p.is_vlm else ("all",))


def test_cohort_members_all_exist():
    known = set(registry.pairs())
    for c in cohorts_mod.all_cohorts().values():
        unknown = set(c.pairs) - known
        assert not unknown, f"cohort {c.key} names unknown pairs: {sorted(unknown)}"


def test_qa8_is_frozen_and_literal():
    """qa8 is the parity target: its membership and its pre-registered 6/8 bar
    must not be re-derived from a fraction that merely happens to equal 6."""
    qa8 = cohorts_mod.all_cohorts()["qa8"]
    assert qa8.frozen
    assert len(qa8.pairs) == 8
    assert qa8.min_passing(8) == 6
    with pytest.raises(ValueError):
        qa8.min_passing(7)  # a literal rule must refuse a different n


def test_fraction_pass_rule_coincides_at_eight():
    vlm = cohorts_mod.all_cohorts()["vlm"]
    assert vlm.min_passing(8) == 6      # ceil(0.75 * 8)
    assert vlm.min_passing(7) == 6      # ceil(0.75 * 7) = 6
    assert vlm.min_passing(18) == 14    # ceil(0.75 * 18)


def test_families_and_kinds():
    assert len(config.FAMILIES) == 7
    assert config.kind_of("hs_wide") == "hs"
    assert config.kind_of("lapeigvals") == "spectral"
    with pytest.raises(ValueError):
        config.kind_of("not_a_family")
