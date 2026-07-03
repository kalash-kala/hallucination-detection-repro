"""Score every (sample, candidate) pair under the teacher-forced LM.

Because we also need the layer-l hidden state at the last answer token for the
internal probe, we collect both quantities in a single forward pass and cache
them together. Downstream scripts read from this cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import ExperimentConfig
from data_loader import Sample, load_split
from model_utils import load_model, score_and_hidden, score_and_hidden_multilayer


def _cache_paths(cfg: ExperimentConfig):
    cache = cfg.paths()["cache"]
    return {
        "scores": cache / f"scores_layer{cfg.layer_1idx}.jsonl",
        "hidden": cache / f"hidden_layer{cfg.layer_1idx}.npz",
        "index":  cache / f"index_layer{cfg.layer_1idx}.json",
    }


def _cache_paths_for_layer(cfg: ExperimentConfig, layer: int) -> dict:
    cache = cfg.paths()["cache"]
    return {
        "scores": cache / f"scores_layer{layer}.jsonl",
        "hidden": cache / f"hidden_layer{layer}.npz",
        "index":  cache / f"index_layer{layer}.json",
    }


def score_all(cfg: ExperimentConfig, samples: list[Sample]) -> None:
    """Forward-pass every (sample_id, candidate) pair; cache log-prob and hidden state."""
    paths = _cache_paths(cfg)
    paths["scores"].parent.mkdir(parents=True, exist_ok=True)

    if paths["scores"].exists() and paths["hidden"].exists() and paths["index"].exists():
        print(f"[external_scorer] cache exists at {paths['scores'].parent}; skipping.")
        return

    load_model(cfg)  # warm up

    records = []
    hidden_rows = []
    index = {}   # (sample_id, candidate) -> row in hidden_rows

    row_idx = 0
    for s in tqdm(samples, desc="external+hidden"):
        for cand in s.candidate_pool:
            sum_lp, n_tok, h = score_and_hidden(s.question, cand, cfg.layer_1idx, cfg)
            norm_score = sum_lp / (max(n_tok, 1) ** cfg.alpha)
            records.append({
                "id": s.id,
                "candidate": cand,
                "sum_logprob": sum_lp,
                "n_answer_tokens": n_tok,
                "s_ext": norm_score,
                "is_gt": cand.strip().lower() == s.ground_truth.strip().lower(),
                "is_greedy": cand.strip().lower() == s.greedy_prediction.strip().lower(),
            })
            index[f"{s.id}|||{cand}"] = row_idx
            hidden_rows.append(h.astype(np.float32))
            row_idx += 1

        # Score additional GT aliases (aliases[1:]) for probe alias selection.
        # Only applies to TriviaQA-style datasets with multiple valid answers.
        for alias in s.gt_aliases[1:]:
            key = f"{s.id}|||{alias}"
            if key in index:
                continue  # already scored (alias text matches a pool candidate)
            sum_lp, n_tok, h = score_and_hidden(s.question, alias, cfg.layer_1idx, cfg)
            norm_score = sum_lp / (max(n_tok, 1) ** cfg.alpha)
            records.append({
                "id": s.id,
                "candidate": alias,
                "sum_logprob": sum_lp,
                "n_answer_tokens": n_tok,
                "s_ext": norm_score,
                "is_gt": True,
                "is_greedy": False,
            })
            index[key] = row_idx
            hidden_rows.append(h.astype(np.float32))
            row_idx += 1

    with paths["scores"].open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    np.savez_compressed(paths["hidden"], hidden=np.stack(hidden_rows, axis=0))
    paths["index"].write_text(json.dumps(index))
    print(f"[external_scorer] wrote {len(records)} rows to {paths['scores']}")


def score_all_layers(cfg: ExperimentConfig, samples: list[Sample], layers: list) -> None:
    """One forward-pass sweep extracting hidden states for every requested layer simultaneously.

    Compared to calling score_all() N times (once per layer), this reduces forward-pass
    work by N×: output_hidden_states=True already computes all layers internally, so we
    only pay for one sweep regardless of how many layers we extract.

    Saves per-layer caches:
      hidden_layer{L}.npz   — different per layer
      scores_layer{L}.jsonl — identical across layers (log-prob is layer-independent)
      index_layer{L}.json   — identical across layers
    """
    cache_dir = cfg.paths()["cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Skip if every layer's cache already exists.
    missing_layers = [
        L for L in layers
        if not all(
            _cache_paths_for_layer(cfg, L)[k].exists()
            for k in ("scores", "hidden", "index")
        )
    ]
    if not missing_layers:
        print(f"[external_scorer] all {len(layers)} layer caches exist; skipping sweep.")
        return

    print(f"[external_scorer] sweeping {len(missing_layers)} missing layers: {missing_layers}")
    load_model(cfg)

    records = []
    hidden_rows: dict = {L: [] for L in missing_layers}
    index: dict = {}
    row_idx = 0

    for s in tqdm(samples, desc=f"sweep layers={missing_layers}"):
        for cand in s.candidate_pool:
            sum_lp, n_tok, hs = score_and_hidden_multilayer(s.question, cand, missing_layers, cfg)
            norm_score = sum_lp / (max(n_tok, 1) ** cfg.alpha)
            records.append({
                "id": s.id,
                "candidate": cand,
                "sum_logprob": sum_lp,
                "n_answer_tokens": n_tok,
                "s_ext": norm_score,
                "is_gt": cand.strip().lower() == s.ground_truth.strip().lower(),
                "is_greedy": cand.strip().lower() == s.greedy_prediction.strip().lower(),
            })
            index[f"{s.id}|||{cand}"] = row_idx
            for L in missing_layers:
                hidden_rows[L].append(hs[L].astype(np.float32))
            row_idx += 1

        for alias in s.gt_aliases[1:]:
            key = f"{s.id}|||{alias}"
            if key in index:
                continue
            sum_lp, n_tok, hs = score_and_hidden_multilayer(s.question, alias, missing_layers, cfg)
            norm_score = sum_lp / (max(n_tok, 1) ** cfg.alpha)
            records.append({
                "id": s.id,
                "candidate": alias,
                "sum_logprob": sum_lp,
                "n_answer_tokens": n_tok,
                "s_ext": norm_score,
                "is_gt": True,
                "is_greedy": False,
            })
            index[key] = row_idx
            for L in missing_layers:
                hidden_rows[L].append(hs[L].astype(np.float32))
            row_idx += 1

    index_text = json.dumps(index)
    for L in missing_layers:
        p = _cache_paths_for_layer(cfg, L)
        with p["scores"].open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        np.savez_compressed(p["hidden"], hidden=np.stack(hidden_rows[L], axis=0))
        p["index"].write_text(index_text)

    print(f"[external_scorer] wrote {len(records)} rows × {len(missing_layers)} layers to {cache_dir}")


def load_cache(cfg: ExperimentConfig):
    paths = _cache_paths(cfg)
    scores = [json.loads(l) for l in paths["scores"].open()]
    hidden = np.load(paths["hidden"])["hidden"]
    index = json.loads(paths["index"].read_text())
    return scores, hidden, index


def lookup_score(scores: list[dict], sample_id: str, candidate: str) -> dict:
    for r in scores:
        if r["id"] == sample_id and r["candidate"] == candidate:
            return r
    raise KeyError(f"no cached score for ({sample_id}, {candidate})")


def build_score_map(scores: list[dict]) -> dict:
    """(sample_id, candidate) -> score record."""
    return {(r["id"], r["candidate"]): r for r in scores}


if __name__ == "__main__":
    from data_loader import load_samples, split_samples, save_split
    cfg = ExperimentConfig()
    samples = load_samples(cfg)
    train, test = split_samples(samples, cfg)
    save_split(train, test, cfg)
    score_all(cfg, samples)   # score every sample, not just test; probe needs train scores too
