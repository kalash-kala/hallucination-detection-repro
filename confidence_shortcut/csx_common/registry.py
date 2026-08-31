"""The pair registry: what pairs exist, and what each one is made of.

Everything downstream addresses a pair by its key (`llama_sciq`,
`qwen25vl_vqav2`). Adding a pair is an entry in `configs/pairs.yaml` plus its
CSV -- never a code change, which matters because the roster is still growing.

Model tags in the source filenames are inconsistent and one is actively
misleading: `qwen_14b` in the nq files is Qwen3-14B, not a Qwen2.5 variant. So
canonical tags are declared in the YAML and resolved here, never parsed out of a
filename.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import paths

CATS = ("IH", "CH", "IL", "CL")
CAT_SHORT = {
    "incorrect_high": "IH",
    "correct_high": "CH",
    "incorrect_low": "IL",
    "correct_low": "CL",
}
I_CATS = ("IH", "IL")
C_CATS = ("CH", "CL")
L_CATS = ("IL", "CL")  # L = high entropy = low confidence


@dataclass(frozen=True)
class Model:
    key: str
    hf_id: str
    modality: str
    aliases: tuple[str, ...]
    layers: int | None
    # Token ids that count as image positions. Empty for text models. This is the
    # ground truth for the image/text segment split: it is read off the stored
    # prompt_token_ids, independently of whatever the processor reports, so the
    # two can be cross-checked instead of one being trusted blindly.
    image_token_ids: tuple[int, ...] = ()

    @property
    def is_vlm(self) -> bool:
        return self.modality == "vlm"


@dataclass(frozen=True)
class Dataset:
    key: str
    modality: str
    prompt_template: str  # 'ranking' | 'chat'
    n_target: int | None  # subsample cap; None = keep the whole pool
    image_root: Path | None
    loader_branch: str | None
    # The exact user-turn wording the generation run used. Required for `chat`
    # datasets; extraction gates on the rebuilt prompt matching the recorded
    # prompt_token_ids token for token.
    prompt_text: str | None = None


@dataclass(frozen=True)
class Pair:
    key: str
    model: Model
    dataset: Dataset
    csv: str
    generations: str | None
    status: str  # 'active' | 'pending'
    # (from, to) prefix substitution for image_path, for runs generated on another
    # machine. Declared per pair; rows.py verifies every rewritten path exists.
    path_rewrite: tuple[str, str] | None = None

    @property
    def is_vlm(self) -> bool:
        return self.model.is_vlm

    def rewrite_path(self, p: str) -> str:
        """Apply the declared prefix substitution. A path that does not start
        with the declared prefix is returned unchanged rather than mangled --
        rows.py then catches it as a missing file."""
        if not self.path_rewrite:
            return p
        src, dst = self.path_rewrite
        return dst + p[len(src):] if p.startswith(src) else p

    @property
    def prompt_template(self) -> str:
        return self.dataset.prompt_template

    @property
    def segments(self) -> tuple[str, ...]:
        """A VLM sequence carries image tokens and text tokens, and the point of
        keeping them apart is that a VLM's linguistic and visual-grounding
        confidence need not agree. Text pairs have one undifferentiated span."""
        return ("all", "image", "text") if self.is_vlm else ("all",)

    @property
    def csv_path(self) -> Path:
        return paths.uq_csv(self.csv)

    @property
    def generations_path(self) -> Path | None:
        return paths.generations_jsonl(self.generations) if self.generations else None

    def needs_generations(self) -> bool:
        """Only VLM pairs do, and only for `image_path` -- the one field no run
        CSV carries. Text pairs are driven entirely from the CSV, which already
        holds question, greedy answer and the 10 sampled answer strings."""
        return self.is_vlm


@functools.lru_cache(maxsize=1)
def _raw() -> dict:
    with (paths.CONFIG_DIR / "pairs.yaml").open() as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def models() -> dict[str, Model]:
    return {
        k: Model(
            key=k,
            hf_id=v["hf_id"],
            modality=v["modality"],
            aliases=tuple(v.get("aliases", (k,))),
            layers=v.get("layers"),
            image_token_ids=tuple(v.get("image_token_ids", ())),
        )
        for k, v in _raw()["models"].items()
    }


@functools.lru_cache(maxsize=1)
def datasets() -> dict[str, Dataset]:
    return {
        k: Dataset(
            key=k,
            modality=v["modality"],
            prompt_template=v["prompt_template"],
            n_target=v.get("n_target"),
            image_root=Path(v["image_root"]) if v.get("image_root") else None,
            loader_branch=v.get("loader_branch"),
            prompt_text=v.get("prompt_text"),
        )
        for k, v in _raw()["datasets"].items()
    }


@functools.lru_cache(maxsize=1)
def pairs() -> dict[str, Pair]:
    ms, ds = models(), datasets()
    out = {}
    for key, v in _raw()["pairs"].items():
        model, dataset = ms[v["model"]], ds[v["dataset"]]
        if model.modality != dataset.modality:
            raise ValueError(
                f"pairs.yaml: {key} pairs a {model.modality} model with a "
                f"{dataset.modality} dataset"
            )
        pr = v.get("path_rewrite")
        out[key] = Pair(
            key=key,
            model=model,
            dataset=dataset,
            csv=v["csv"],
            generations=v.get("generations"),
            status=v.get("status", "active"),
            path_rewrite=(pr["from"], pr["to"]) if pr else None,
        )
    return out


def get(key: str) -> Pair:
    try:
        return pairs()[key]
    except KeyError:
        raise KeyError(
            f"unknown pair {key!r}; add it to configs/pairs.yaml. "
            f"known: {', '.join(sorted(pairs()))}"
        ) from None


def resolve(keys: str | list[str] | None = None, *, include_pending: bool = False,
            modality: str | None = None) -> list[Pair]:
    """Resolve a `--pairs` selector to Pair objects, in registry order.

    `keys` may be None (all), a comma-separated string, or a list. Pending pairs
    are excluded unless asked for, so a roster entry for data that has not landed
    never silently becomes an empty result set.
    """
    if keys is None:
        chosen = list(pairs().values())
    else:
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        chosen = [get(k) for k in keys]
    if not include_pending:
        chosen = [p for p in chosen if p.status == "active"]
    if modality is not None:
        chosen = [p for p in chosen if p.model.modality == modality]
    return chosen


@functools.lru_cache(maxsize=1)
def out_of_scope_csvs() -> frozenset[str]:
    """CSVs on disk that are deliberately not pairs. Declared so the census can
    be asserted complete: every file is either a pair or is named here."""
    return frozenset(_raw().get("out_of_scope", {}).get("csvs", ()))


@functools.lru_cache(maxsize=1)
def alias_to_model() -> dict[str, str]:
    """Every declared alias -> canonical model key. Built eagerly so a duplicate
    alias across two models is a load-time error rather than a silent
    last-one-wins."""
    out: dict[str, str] = {}
    for m in models().values():
        for a in m.aliases:
            if a in out and out[a] != m.key:
                raise ValueError(
                    f"pairs.yaml: alias {a!r} claimed by both {out[a]!r} and {m.key!r}"
                )
            out[a] = m.key
    return out
