"""Compute SNNE text-only UQ scores on the TRAIN-split generations and dump
per-example scores (same computation as dump_snne_scores.py, new I/O namespace).

Input : results/snne_baseline/generations_train/<pair>.jsonl
Output: results/snne_baseline/scores_train/<pair>.csv
        columns: id, label, num_set, lexical_sim, sum_eigv, degree, eccentricity, luq, snne

Usage:
    /root/miniconda3/envs/snne/bin/python dump_train_scores.py [--pairs llama_sciq ...]
"""
import argparse
import json
import logging

import pandas as pd
from rouge_score import tokenizers
import evaluate

import snne_core as sc
from run_baselines import per_question_scores, METHODS
import snne_train_common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_pair(pair, deberta, rouge, tokenizer, shard=0, nshard=1):
    path = C.GEN_TRAIN_DIR / f"{pair}.jsonl"
    if not path.exists():
        logger.warning(f"[{pair}] no generations file, skipping")
        return
    rows = [json.loads(l) for l in open(path)]
    if nshard > 1:
        rows = rows[shard::nshard]   # interleaved slice -> balanced shards
    recs = []
    for i, r in enumerate(rows):
        gens = r["generations"]
        if len(gens) < 2:
            continue
        s = per_question_scores(r["question"], gens, deberta, rouge, tokenizer)
        rec = {"id": r["id"], "label": 1 if r["label"] else 0}
        rec.update({m: s[m] for m in METHODS})
        recs.append(rec)
        if (i + 1) % 200 == 0:
            logger.info(f"[{pair}] {i + 1}/{len(rows)} questions")
    C.SCORE_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    if nshard > 1:
        out = C.SCORE_TRAIN_DIR / f"{pair}.part{shard}of{nshard}.csv"
    else:
        out = C.SCORE_TRAIN_DIR / f"{pair}.csv"
    pd.DataFrame(recs).to_csv(out, index=False)
    logger.info(f"[{pair}] wrote {len(recs)} rows -> {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    pairs = args.pairs or C.ALL_PAIRS

    logger.info("Loading DeBERTa entailment model + ROUGE...")
    deberta = sc.EntailmentDeberta()
    rouge = evaluate.load("rouge", keep_in_memory=True)
    tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize

    for pair in pairs:
        run_pair(pair, deberta, rouge, tokenizer, shard=args.shard, nshard=args.nshard)


if __name__ == "__main__":
    main()