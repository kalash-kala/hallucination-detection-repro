"""Compute SNNE text-only UQ baselines on our test questions and score AUROC.

For each pair, reads results/snne_baseline/generations/<pair>.jsonl (10 generations
per question + our open_text_label) and computes, per question, an uncertainty
score for each method, then AUROC/AUARC/PRR of (uncertainty vs incorrectness).

Methods (all unsupervised, text-only):
  num_set      : number of semantic-equivalence clusters among the 10 generations
  lexical_sim  : - mean pairwise ROUGE-L similarity
  sum_eigv     : sum of eigenvalues of the entailment graph Laplacian (SumEigv)
  degree       : degree-matrix uncertainty (Deg)
  eccentricity : eccentricity of the entailment graph (Eccen)
  luq          : LUQ pairwise (1 - max entailment), non-strict / unidirectional
  snne         : soft nearest-neighbour energy (only_denom, temp=1.0, entailment sim)

Settings match SNNE's scripts:
  - condition_on_question = True (CONCAT=True for QA): entailment/lexical sim are
    computed on "<question> <answer>" strings.
  - graph baselines + SNNE use the STRICT entailment matrix; LUQ uses the
    non-strict, exclude-neutral, unidirectional matrix (as in compute_graph_baselines).
  - SNNE: variant=only_denom, temperature=1.0, selfsim=True (exclude_diagonal=False).

AUROC is computed as auroc(is_false, uncertainty) — exactly comparable to our
Classifier A ROC-AUC for correct-vs-incorrect on the same test questions.

Usage:
    python run_baselines.py [--pairs llama_sciq mistral_sciq ...]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import tokenizers
import evaluate

import snne_core as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GEN_DIR = REPO_ROOT / "results" / "snne_baseline" / "generations"
OUT_DIR = REPO_ROOT / "results" / "snne_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_PAIRS = [
    "llama_sciq", "llama_triviaqa", "llama_math",
    "mistral_sciq", "mistral_triviaqa", "mistral_math",
    "qwen_sciq", "qwen_triviaqa", "qwen_math",
    "gemma_sciq", "gemma_triviaqa", "gemma_math",
]
METHODS = ["num_set", "lexical_sim", "sum_eigv", "degree", "eccentricity", "luq", "snne"]


def per_question_scores(question, generations, deberta, rouge, tokenizer):
    """Return {method: uncertainty_score} for one question."""
    deberta.clear_cache()
    texts = [f"{question} {g}" for g in generations]  # condition_on_question=True

    # Similarity matrices (entailment strict for graph+snne; non-strict for LUQ).
    ent_strict = sc.entailment_similarity_matrix(deberta, texts, strict_entailment=True)
    ent_luq = sc.entailment_similarity_matrix(
        deberta, texts, strict_entailment=False, exclude_neutral=True, bidirectional=False)
    lex = sc.lexical_similarity_matrix(rouge, texts, tokenizer=tokenizer)
    sem_ids = sc.get_semantic_ids_using_entailment(texts, deberta)

    ent_np = ent_strict.numpy()
    out = {}
    out["num_set"] = float(max(sem_ids) + 1)
    out["lexical_sim"] = -sc.compute_lexical_similarity(lex)
    out["sum_eigv"] = float(sc.get_spectral_eigv(ent_np.copy()))
    out["degree"] = float(sc.get_degreeuq(ent_np.copy())[0])
    out["eccentricity"] = float(sc.get_eccentricity(ent_np.copy())[0])
    out["luq"] = float(sc.get_luq_pair(ent_luq.numpy().copy())[0])
    out["snne"] = float(sc.snne(
        ent_strict, sem_ids, variant="only_denom", temperature=1.0, exclude_diagonal=False).item())
    return out


def run_pair(pair, deberta, rouge, tokenizer):
    path = GEN_DIR / f"{pair}.jsonl"
    if not path.exists():
        logger.warning(f"[{pair}] no generations file, skipping")
        return None
    rows = [json.loads(l) for l in open(path)]

    labels, scores = [], {m: [] for m in METHODS}
    for i, r in enumerate(rows):
        gens = r["generations"]
        if len(gens) < 2:
            continue
        s = per_question_scores(r["question"], gens, deberta, rouge, tokenizer)
        labels.append(1 if r["label"] else 0)
        for m in METHODS:
            scores[m].append(s[m])
        if (i + 1) % 100 == 0:
            logger.info(f"[{pair}] {i + 1}/{len(rows)} questions")

    is_false = [1 - y for y in labels]
    is_true = labels
    recs = []
    for m in METHODS:
        u = scores[m]
        recs.append({
            "pair": pair, "method": m,
            "auroc": sc.auroc(is_false, u),
            "auarc": sc.auarc(u, is_true),
            "prr": sc.aucpr(u, is_true),
            "n": len(u),
        })
    df = pd.DataFrame(recs)
    df.to_csv(OUT_DIR / f"{pair}_metrics.csv", index=False)
    logger.info(f"[{pair}] done ({len(labels)} q). Best AUROC: "
                f"{df.loc[df.auroc.idxmax(), 'method']}={df.auroc.max():.3f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Subset of pairs to run (default: all available).")
    args = ap.parse_args()
    pairs = args.pairs or ALL_PAIRS

    logger.info("Loading DeBERTa entailment model + ROUGE...")
    deberta = sc.EntailmentDeberta()
    rouge = evaluate.load("rouge", keep_in_memory=True)
    tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize

    all_dfs = []
    for pair in pairs:
        df = run_pair(pair, deberta, rouge, tokenizer)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(OUT_DIR / "all_pairs_metrics.csv", index=False)
        logger.info(f"\nSaved combined metrics -> {OUT_DIR / 'all_pairs_metrics.csv'}")
        # quick per-method mean AUROC across the pairs we ran
        piv = combined.pivot_table(index="method", values="auroc", aggfunc="mean")
        logger.info(f"\nMean AUROC by method (over {len(all_dfs)} pairs):\n{piv.round(4)}")


if __name__ == "__main__":
    main()