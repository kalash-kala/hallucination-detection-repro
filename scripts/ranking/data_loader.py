"""Load CSV into ranking records and split 70/30 by question."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config import ExperimentConfig


@dataclass
class Sample:
    id: str
    question: str
    ground_truth: str
    greedy_prediction: str
    candidate_list: list        # distractors as given in the CSV (may include greedy when verdict=False)
    candidate_pool: list        # {ground_truth} + candidate_list, deduped, order: [GT, *distractors]
    open_text_label: bool       # LLM_verdict
    category: Optional[str] = None       # e.g. correct_high, incorrect_low
    c_metric: Optional[float] = None     # concentration metric
    gt_aliases: list = field(default_factory=list)  # all valid GT aliases (TriviaQA); [ground_truth] for others

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(x: str) -> str:
    return str(x).strip().lower()


def _normalize_quotes(x: str) -> str:
    # The greedy_prediction column uses whichever quote char the model output, while
    # ast.literal_eval on the candidate_list preserves the inner quote chars from the
    # Python list literal. Normalizing to single quotes makes both sides match.
    return x.replace('"', "'")


def _parse_ground_truth(raw: str) -> str:
    """Extract ground truth string from CSV value.

    New-format files store ground_truth as a Python list literal, e.g.
    "['boiling temperature']" (SciQ, Math — single element) or
    "['Aubergine', 'Aubergines', ...]" (TriviaQA — multiple valid aliases).
    We take the first element as the canonical answer.
    Old-format files store a plain string.
    """
    raw = str(raw).strip()
    if raw.startswith("["):
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0]).strip()
    return raw


def load_samples(cfg: ExperimentConfig) -> list[Sample]:
    df = pd.read_csv(cfg.input_csv)
    required = [
        cfg.id_column, cfg.question_column, cfg.gt_column,
        cfg.answer_column, cfg.candidate_column, cfg.verdict_column,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    has_category = "category" in df.columns
    has_c_metric = "C_metric" in df.columns

    samples: list[Sample] = []
    for _, row in df.iterrows():
        raw_candidates = row[cfg.candidate_column]
        if isinstance(raw_candidates, str):
            candidates = ast.literal_eval(raw_candidates)
        else:
            candidates = list(raw_candidates)
        candidates = [str(c).strip() for c in candidates]

        gt = _normalize_quotes(_parse_ground_truth(row[cfg.gt_column]))

        # Parse full alias list for probe alias selection (TriviaQA has many valid aliases)
        raw_gt_str = str(row[cfg.gt_column]).strip()
        if raw_gt_str.startswith("["):
            try:
                parsed_aliases = ast.literal_eval(raw_gt_str)
                if isinstance(parsed_aliases, list) and len(parsed_aliases) > 0:
                    gt_aliases = [_normalize_quotes(str(a).strip()) for a in parsed_aliases]
                else:
                    gt_aliases = [gt]
            except Exception:
                gt_aliases = [gt]
        else:
            gt_aliases = [gt]

        greedy = _normalize_quotes(str(row[cfg.answer_column]).strip())
        candidates = [_normalize_quotes(c) for c in candidates]

        pool = [gt]
        seen = {_norm(gt)}
        for c in candidates:
            if _norm(c) not in seen:
                pool.append(c)
                seen.add(_norm(c))

        verdict = row[cfg.verdict_column]
        if isinstance(verdict, str):
            verdict_bool = verdict.strip().lower() == "true"
        else:
            verdict_bool = bool(verdict)

        cat = str(row["category"]) if has_category and pd.notna(row.get("category")) else None
        c_met = float(row["C_metric"]) if has_c_metric and pd.notna(row.get("C_metric")) else None
        if c_met is not None and math.isnan(c_met):
            c_met = None

        samples.append(Sample(
            id=str(row[cfg.id_column]),
            question=str(row[cfg.question_column]).strip(),
            ground_truth=gt,
            greedy_prediction=greedy,
            candidate_list=candidates,
            candidate_pool=pool,
            open_text_label=verdict_bool,
            category=cat,
            c_metric=c_met,
            gt_aliases=gt_aliases,
        ))
    return samples


def split_samples(samples: list[Sample], cfg: ExperimentConfig) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(cfg.seed)
    correct = [s for s in samples if s.open_text_label]
    incorrect = [s for s in samples if not s.open_text_label]
    train, test = [], []
    for stratum in (correct, incorrect):
        ids = np.array([s.id for s in stratum])
        perm = rng.permutation(len(ids))
        n_test = int(round(len(ids) * cfg.test_fraction))
        test_ids = set(ids[perm[:n_test]].tolist())
        for s in stratum:
            (test if s.id in test_ids else train).append(s)
    return train, test


def save_split(train: list[Sample], test: list[Sample], cfg: ExperimentConfig) -> None:
    splits_dir = cfg.paths()["splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train_ids.json").write_text(json.dumps([s.id for s in train], indent=2))
    (splits_dir / "test_ids.json").write_text(json.dumps([s.id for s in test], indent=2))
    with (splits_dir / "train.jsonl").open("w") as f:
        for s in train:
            f.write(json.dumps(s.to_dict()) + "\n")
    with (splits_dir / "test.jsonl").open("w") as f:
        for s in test:
            f.write(json.dumps(s.to_dict()) + "\n")


def load_split(cfg: ExperimentConfig, name: str) -> list[Sample]:
    path = cfg.paths()["splits"] / f"{name}.jsonl"
    out: list[Sample] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.append(Sample(**d))
    return out
