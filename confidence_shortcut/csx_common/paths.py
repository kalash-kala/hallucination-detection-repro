"""Path resolution for every artifact the study touches.

One module owns every root, so relocating the store or pointing at a copy of the
reference results is a config edit or an environment variable, never a code
change. Roots come from `configs/pairs.yaml:roots`, each overridable by an
environment variable named `CSX_<KEY>` (upper-cased).

This module is imported by csx_probe and csx_report. csx_extract has its own
thin copy (csx_extract/paths.py) that reads the same YAML -- the two packages do
not import each other, by design; see store_spec/STORE_CONTRACT.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# confidence_shortcut/csx_probe/paths.py -> confidence_shortcut/
PKG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PKG_ROOT / "configs"
STORE_SPEC = PKG_ROOT / "store_spec" / "STORE_CONTRACT.md"
RESULTS_SPEC = PKG_ROOT / "results_spec" / "RESULTS_CONTRACT.md"

_ROOT_KEYS = (
    "uq_csv_dir",
    "store",
    "generations",
    "legacy_full_natural",
    "legacy_dse_arms",
    "reference_results",
)


def _load_roots() -> dict[str, Path]:
    with (CONFIG_DIR / "pairs.yaml").open() as fh:
        raw = yaml.safe_load(fh)["roots"]
    out = {}
    for key in _ROOT_KEYS:
        env = os.environ.get(f"CSX_{key.upper()}")
        out[key] = Path(env if env else raw[key])
    return out


ROOTS = _load_roots()


# ── store layout (contract 1) ────────────────────────────────────────────────
def store_root() -> Path:
    return ROOTS["store"]


def uq_table() -> Path:
    """L2: one parquet holding every row of every in-scope run CSV."""
    return store_root() / "uq" / "uq_rows.parquet"


def raw_dir(pair: str) -> Path:
    """L0: reduced internal states, written by csx_extract."""
    return store_root() / "raw" / pair


def raw_meta(pair: str) -> Path:
    return raw_dir(pair) / "meta.json"


def raw_rows(pair: str) -> Path:
    return raw_dir(pair) / "rows.parquet"


def uq_metrics(pair: str) -> Path:
    """M20: the 7 text-only NLI metrics (num_set/lexical_sim/sum_eigv/degree/
    eccentricity/luq/snne), one row per id -- written by csx_extract's
    uq_metrics.py, read back by csx_probe.uq.read_pair as an optional
    left-join (absent for a pair until its M20 pass has run)."""
    return raw_dir(pair) / "uq_metrics.parquet"


def sampled_dir(pair: str) -> Path:
    """Internal states of the 10 sampled generations, written by csx_extract.

    A labeled subtree distinct from the permanent raw/ L0 entries, but kept
    (not purged) once written -- see docs/PART_C_ROUTING_PLAN.md.
    """
    return store_root() / "sampled" / pair


def sampled_meta(pair: str) -> Path:
    return sampled_dir(pair) / "meta.json"


def sampled_manifest(pair: str) -> Path:
    """id, slot(0..9) -> urow -- which unique-text feature row each sampled
    generation slot reads from."""
    return sampled_dir(pair) / "manifest.parquet"


def sampled_unique(pair: str) -> Path:
    """One row per unique (id, text) extracted -- the manifest's payload
    table: id, urow, text."""
    return sampled_dir(pair) / "unique.parquet"


def feature_dir(pair: str, family: str, segment: str = "all") -> Path:
    """L1: derived family feature matrices, written by csx_probe.store.build."""
    return store_root() / "features" / pair / family / segment


def manifest() -> Path:
    return store_root() / "manifest.parquet"


# ── results layout (contract 2) ──────────────────────────────────────────────
def results_root() -> Path:
    return store_root() / "results"


def results_table(name: str) -> Path:
    """Atomic per-pair tables written by csx_probe.experiments.

    `name` is one of the tables named in results_spec/RESULTS_CONTRACT.md,
    e.g. 'per_pair_long', 'rotation_long', 'verdict', 'c_selection'.
    """
    return results_root() / f"{name}.parquet"


def results_units(name: str, pair: str) -> Path:
    """Per-unit checkpoint, so a killed run resumes instead of restarting."""
    return results_root() / "units" / name / f"{pair}.parquet"


def report_dir() -> Path:
    return store_root() / "reports"


# ── legacy artifacts (read-only; the qa8 parity path) ────────────────────────
def legacy_hs_npz(pair: str, scheme: str, split: str) -> Path:
    """full_natural pooled hidden-state features: hs_{scheme}_{split}.npz."""
    return ROOTS["legacy_full_natural"] / "features" / pair / f"hs_{scheme}_{split}.npz"


def legacy_peaks(pair: str) -> Path:
    return ROOTS["legacy_full_natural"] / "features" / pair / "peaks.json"


def legacy_diags(pair: str, split: str) -> Path:
    """Attention/Laplacian diagonals [L,H,S] on the natural splits."""
    return (ROOTS["legacy_full_natural"] / "results" / "lapeigvals_baseline"
            / "diags" / f"{pair}_{split}.pt")


def legacy_value_norms(pair: str, split: str) -> Path:
    return (ROOTS["legacy_full_natural"] / "results" / "sinkhole"
            / "value_norms" / f"{pair}_{split}.pt")


def legacy_bundle(pair: str, family: str) -> Path:
    """The published 24 GB pickle cache. Read only to pin CV-selected
    (top_k, pca_dim) and as the store-parity reference; never written."""
    return (ROOTS["legacy_full_natural"] / "routing_analysis"
            / "routed_specialist_grid" / "_bundle_cache" / f"{pair}__{family}.pkl")


def legacy_arm_split(arm_dir: str, pair: str, split: str) -> Path:
    """Published arm row-sets, e.g. arm_dir='natural' or 'placebo/matched_d00'."""
    return (ROOTS["legacy_dse_arms"] / arm_dir / f"ranking_experiment_{pair}"
            / "splits" / f"{split}.jsonl")


def reference(*parts: str) -> Path:
    """A published artifact under dse_results/, for the parity gate."""
    return ROOTS["reference_results"].joinpath(*parts)


# ── inputs ───────────────────────────────────────────────────────────────────
def uq_csv(filename: str) -> Path:
    return ROOTS["uq_csv_dir"] / filename


def generations_jsonl(folder: str) -> Path:
    return ROOTS["generations"] / folder / "combined_generations.jsonl"
