"""The layer-bucket rule must reproduce the published table exactly.

`classifier/build_features.py:37` hardcodes mid/late regions for the four qa8
backbones only. Every other pair -- the VLMs, nq, gemma3_27b, qwen3_14b -- needs
the same regions at a different depth, so config.wide_buckets recovers the rule
as a fraction of the block count.

If that formula did not reproduce all four published configs it would be a fresh
invention wearing the old name, and hs_* features for new pairs would sit in a
different space from the parity pairs while looking identical. Hence this test.
"""

from __future__ import annotations

import pytest

from csx_extract import config

# Verbatim from BUCKET_VARIANTS["wide"].
PUBLISHED = {
    "llama":   (32, list(range(11, 22)), list(range(22, 33))),
    "mistral": (32, list(range(11, 22)), list(range(22, 33))),
    "qwen":    (28, list(range(10, 19)), list(range(19, 29))),
    "gemma":   (48, list(range(17, 33)), list(range(33, 49))),
}


@pytest.mark.parametrize("model", sorted(PUBLISHED))
def test_reproduces_published_buckets(model):
    n_layers, mid, late = PUBLISHED[model]
    got = config.wide_buckets(n_layers)
    assert got["mid"] == mid, f"{model} mid"
    assert got["late"] == late, f"{model} late"


@pytest.mark.parametrize("n_layers", [28, 32, 40, 48, 62])
def test_buckets_are_contiguous_and_cover_the_upper_model(n_layers):
    b = config.wide_buckets(n_layers)
    assert b["mid"][-1] + 1 == b["late"][0]
    assert b["late"][-1] == n_layers
    assert b["mid"] and b["late"]


@pytest.mark.parametrize("n_layers", [28, 40, 48, 62])
def test_no_bucket_reaches_the_embedding_layer(n_layers):
    """Layers are 1-indexed against hidden_states, where index 0 is the embedding
    output and is never a valid choice."""
    b = config.wide_buckets(n_layers)
    assert min(b["mid"]) >= 1


def test_narrow_is_peak_plus_minus_two_clipped_to_region():
    wide = config.wide_buckets(32)
    peaks = {"mid": 15, "late": 32}
    nb = config.narrow_buckets(peaks, wide)
    assert nb["mid"] == [13, 14, 15, 16, 17]
    # Clipped at the top of the model, not extended past it.
    assert nb["late"] == [30, 31, 32]


def test_peak_only_is_a_single_layer():
    wide = config.wide_buckets(32)
    pk = config.peak_only_buckets({"mid": 15, "late": 29}, wide)
    assert pk == {"mid": [15], "late": [29]}


def test_published_narrow_bucket_reproduces_for_llama_sciq():
    """llama_sciq's published narrow/peak_only entries are mid=[13..17],
    late=[27..31] with peaks 15/29. Given those peaks, the rule must regenerate
    the same layers -- a check that narrow/peak_only were not redefined."""
    wide = config.wide_buckets(32)
    nb = config.narrow_buckets({"mid": 15, "late": 29}, wide)
    assert nb["mid"] == [13, 14, 15, 16, 17]
    assert nb["late"] == [27, 28, 29, 30, 31]


def test_scheme_dispatch():
    wide = config.wide_buckets(28)
    peaks = {"mid": 12, "late": 22}
    assert config.buckets_for("hs_wide", peaks, wide) == wide
    assert config.buckets_for("hs_peak_only", peaks, wide)["mid"] == [12]
    with pytest.raises(KeyError):
        config.buckets_for("nope", peaks, wide)


def test_too_few_layers_is_refused():
    with pytest.raises(ValueError):
        config.wide_buckets(2)


# ── plan-mode honesty ────────────────────────────────────────────────────────

def test_unit_blocked_covers_content_preconditions_not_just_files(tmp_path):
    """--plan must not promise work that will die on the first row.

    Stage 23 reads each row's span from rows.parquet instead of recomputing it,
    so it depends on stage 21 having run. But rows.parquet exists from stage 20,
    so a pure file-existence check reports the unit ready. `unmet` carries the
    content-level precondition that existence cannot express.
    """
    from csx_common.cli import Unit

    present = tmp_path / "rows.parquet"
    present.write_text("x")

    ready = Unit(key="p", outputs=[tmp_path / "out.npz"], inputs=[present])
    assert ready.blocked == []

    gated = Unit(key="p", outputs=[tmp_path / "out.npz"], inputs=[present],
                 unmet=["phase 1 has not run"])
    assert gated.blocked == ["phase 1 has not run"]

    missing = Unit(key="p", outputs=[tmp_path / "out.npz"],
                   inputs=[tmp_path / "absent.parquet"],
                   unmet=["phase 1 has not run"])
    # Both kinds of blocker surface, file-existence first.
    assert len(missing.blocked) == 2
    assert missing.blocked[0].startswith("missing ")
