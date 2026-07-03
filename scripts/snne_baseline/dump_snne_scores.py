"""Recompute SNNE text-only UQ baselines and DUMP per-example scores.

Same computation as run_baselines.py, but instead of only writing aggregate AUROC,
this saves one row per question with the id, label, and all 7 method scores so we
can compute per-CATEGORY (IH/IL/CH/CL) within-band AUROCs downstream.

Output: results/snne_baseline/scores/<pair>.csv
        columns: id, label, num_set, lexical_sim, sum_eigv, degree, eccentricity, luq, snne

Usage:
    /root/miniconda3/envs/snne/bin/python dump_snne_scores.py [--pairs llama_sciq ...]
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from rouge_score import tokenizers
import evaluate

import snne_core as sc
from run_baselines import per_question_scores, ALL_PAIRS, METHODS, GEN_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCORE_DIR = REPO_ROOT / "results" / "snne_baseline" / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)


def run_pair(pair, deberta, rouge, tokenizer):
    path = GEN_DIR / f"{pair}.jsonl"
    if not path.exists():
        logger.warning(f"[{pair}] no generations file, skipping")
        return
    rows = [json.loads(l) for l in open(path)]

    recs = []
    for i, r in enumerate(rows):
        gens = r["generations"]
        if len(gens) < 2:
            continue
        s = per_question_scores(r["question"], gens, deberta, rouge, tokenizer)
        rec = {"id": r["id"], "label": 1 if r["label"] else 0}
        rec.update({m: s[m] for m in METHODS})
        recs.append(rec)
        if (i + 1) % 100 == 0:
            logger.info(f"[{pair}] {i + 1}/{len(rows)} questions")

    df = pd.DataFrame(recs)
    out = SCORE_DIR / f"{pair}.csv"
    df.to_csv(out, index=False)
    logger.info(f"[{pair}] wrote {len(df)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=None)
    args = ap.parse_args()
    pairs = args.pairs or ALL_PAIRS

    logger.info("Loading DeBERTa entailment model + ROUGE...")
    deberta = sc.EntailmentDeberta()
    rouge = evaluate.load("rouge", keep_in_memory=True)
    tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize

    for pair in pairs:
        run_pair(pair, deberta, rouge, tokenizer)


if __name__ == "__main__":
    main()
