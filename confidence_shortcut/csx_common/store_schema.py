"""The store contract, in code.

Lives in csx_common because both sides of contract 1 need it: csx_extract writes
entries and verifies them, csx_probe reads them. Keeping one definition means the
two packages cannot drift apart without a test noticing -- which is the whole
point of having a contract rather than a convention.

Prose version, with the reasoning: store_spec/STORE_CONTRACT.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# rows.parquet
ROW_COLUMNS = {
    "id": "string",
    "row": "int32",
    "category": "string",
    "entropy": "float64",
    "s_ext": "float32",
    "seq_len": "int32",
    "answer_start": "int32",
    "answer_end": "int32",
    "image_start": "int32",   # -1 for text pairs
    "image_end": "int32",     # -1 for text pairs
    "image_path": "string",   # "" for text pairs
}

# diag/<segment>.npz. attn_logdet is REQUIRED whenever phase 2 is done: attnlogdet
# is a mean over ALL positions, so it cannot be recovered from the top-k arrays.
# Its absence would leave that family silently wrong rather than missing.
DIAG_TOPK_KEYS = ("attn_topk", "lap_topk", "sink_topk", "sink_vnorm_topk")
DIAG_KEYS = (*DIAG_TOPK_KEYS, "attn_logdet")

# The sink arrays are stored at SINK_K, the attention/Laplacian ones at
# EXTRACT_TOP_K, and those constants differ (10 vs 50). The two groups are named
# here so a width check compares each against its own constant rather than
# demanding one uniform width across all four.
DIAG_SINK_KEYS = ("sink_topk", "sink_vnorm_topk")

HS_SCHEMES = ("hs_wide", "hs_narrow", "hs_peak_only")

# Which families each phase unlocks. A pair with phase 1 done and phase 2 not is
# valid and usable -- it simply has no spectral families yet.
PHASE_FAMILIES = {
    "phase1": HS_SCHEMES,
    "phase2": ("lapeigvals", "attn_eigvals", "attnlogdet", "sink"),
}

_REQUIRED_META = ("schema_version", "pair", "model", "dataset", "modality",
                  "prompt_template", "segments", "n_rows", "phase1", "phase2")
_REQUIRED_MODEL = ("key", "hf_id", "layers")


class ContractError(Exception):
    """A store entry violates the contract. The message names the reason."""


@dataclass
class Problem:
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.detail}"


@dataclass
class EntryMeta:
    """Parsed meta.json for one L0 entry."""
    raw: dict[str, Any]

    @property
    def pair(self) -> str:
        return self.raw["pair"]

    @property
    def segments(self) -> tuple[str, ...]:
        return tuple(self.raw["segments"])

    @property
    def n_rows(self) -> int:
        return int(self.raw["n_rows"])

    @property
    def n_kept(self) -> int | None:
        """The roster size stage 20 settled on, before any phase ran.

        `n_rows` is whatever the row table currently holds, which a `--limit`
        pass shrinks. This is the number it should be, so the two can be
        compared instead of the truncation going unnoticed.
        """
        v = self.raw.get("subsample", {}).get("n_kept")
        return int(v) if v is not None else None

    @property
    def partial_limit(self) -> int | None:
        """Set when phase 1 last ran under `--limit`; None after a full pass."""
        return self.raw.get("phase1_pass", {}).get("partial_limit")

    @property
    def layers(self) -> int | None:
        v = self.raw.get("model", {}).get("layers")
        return int(v) if v is not None else None

    @property
    def modality(self) -> str:
        return self.raw["modality"]

    def phase_done(self, phase: str) -> bool:
        return bool(self.raw.get(phase, {}).get("done", False))

    def available_families(self) -> tuple[str, ...]:
        out: list[str] = []
        for phase, fams in PHASE_FAMILIES.items():
            if self.phase_done(phase):
                out.extend(fams)
        return tuple(out)

    @classmethod
    def load(cls, path: Path) -> "EntryMeta":
        try:
            raw = json.loads(Path(path).read_text())
        except FileNotFoundError:
            raise ContractError(f"no meta.json at {path}") from None
        except json.JSONDecodeError as exc:
            raise ContractError(f"meta.json at {path} is not valid JSON: {exc}") from None
        missing = [k for k in _REQUIRED_META if k not in raw]
        if missing:
            raise ContractError(f"meta.json missing required keys: {missing}")
        got = int(raw["schema_version"])
        if got != SCHEMA_VERSION:
            # Refuse rather than guess: an entry from a newer extractor may have
            # changed what a field means, and a best-effort read would be wrong
            # quietly.
            raise ContractError(
                f"store schema v{got}, this code speaks v{SCHEMA_VERSION}; "
                f"upgrade csx_probe rather than reading it optimistically"
            )
        mm = [k for k in _REQUIRED_MODEL if k not in raw.get("model", {})]
        if mm:
            raise ContractError(f"meta.json model block missing {mm}")
        return cls(raw=raw)


def new_meta(*, pair: str, model: dict, dataset: str, modality: str,
             prompt_template: str, segments: list[str], n_rows: int,
             n_pool: int, subsample: dict | None, top_k: int, sink_k: int,
             extractor_version: str) -> dict:
    """Build a fresh meta.json body. Phases start not-done and are flipped as
    each pass completes, so an interrupted extraction is never mistaken for a
    finished one."""
    return {
        "schema_version": SCHEMA_VERSION,
        "pair": pair,
        "model": model,
        "dataset": dataset,
        "modality": modality,
        "prompt_template": prompt_template,
        "segments": list(segments),
        "n_rows": int(n_rows),
        "n_pool": int(n_pool),
        "subsample": subsample,
        "phase1": {"done": False},
        "phase2": {"done": False},
        "top_k": int(top_k),
        "sink_k": int(sink_k),
        "extractor_version": extractor_version,
        "notes": [],
    }


# ── L1 ───────────────────────────────────────────────────────────────────────

FEATURE_META_KEYS = ("schema_version", "pair", "family", "segment", "kind",
                     "pca_dim", "dim", "n", "sha256", "source", "builder_version")

MANIFEST_COLUMNS = ("pair", "family", "segment", "status", "n", "dim", "kind",
                    "pca_dim", "source_sha256", "built")

# `missing_raw` is a legitimate state, not a failure: it is how a pair whose
# extraction has not happened (or whose only artifacts are the unusable 1,400-row
# legacy sampled runs) is represented without pretending it is buildable.
MANIFEST_STATUSES = ("ready", "missing_raw", "stale", "error")


@dataclass
class FeatureMeta:
    pair: str
    family: str
    segment: str
    kind: str            # 'hs' | 'spectral'
    pca_dim: int | None
    dim: int
    n: int
    sha256: str
    source: dict
    builder_version: str
    top_k: int | None = None
    schema_version: int = SCHEMA_VERSION
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = {k: getattr(self, k) for k in FEATURE_META_KEYS if hasattr(self, k)}
        d["top_k"] = self.top_k
        d.update(self.extra)
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "FeatureMeta":
        raw = json.loads(Path(path).read_text())
        missing = [k for k in FEATURE_META_KEYS if k not in raw]
        if missing:
            raise ContractError(f"feature meta {path} missing {missing}")
        if int(raw["schema_version"]) != SCHEMA_VERSION:
            raise ContractError(
                f"feature meta {path}: schema v{raw['schema_version']} != "
                f"v{SCHEMA_VERSION}"
            )
        known = {f for f in FEATURE_META_KEYS} | {"top_k"}
        return cls(
            pair=raw["pair"], family=raw["family"], segment=raw["segment"],
            kind=raw["kind"], pca_dim=raw["pca_dim"], dim=int(raw["dim"]),
            n=int(raw["n"]), sha256=raw["sha256"], source=raw["source"],
            builder_version=raw["builder_version"], top_k=raw.get("top_k"),
            extra={k: v for k, v in raw.items() if k not in known},
        )
