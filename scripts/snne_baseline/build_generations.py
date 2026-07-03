"""Build the canonical per-pair generation files the SNNE baselines consume.

Output (one line per test question), written to results/snne_baseline/generations/<pair>.jsonl:
    {
      "id": "train::48139",
      "question": "...",
      "generations": ["ans1", ..., "ans10"],   # 10 stochastic samples (temp 1.0)
      "label": true,                              # OUR open_text_label (greedy correctness)
      "run_accuracy": 1.0                         # the source run's most_likely accuracy (sanity only)
    }

The `label` is OUR open_text_label (the exact target Classifier A predicts), so the
SNNE baselines and our classifier are scored against the same target on the same
test questions.

5 pairs are sourced from existing semantic_uncertainty runs (100% test overlap);
the other 4 (qwen_sciq + triviaqa x3) are produced by regenerate_gaps.py with the
same schema.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"
OUT_DIR = REPO_ROOT / "results" / "snne_baseline" / "generations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SU_DIR = Path("/data/kalashkala/semantic_uncertainty_data/uncertainty")

# Pairs with existing runs at 100% test-set overlap.
READY_RUNS = {
    "llama_sciq":   "sciq__meta-llama__Llama-3.1-8B-Instruct__seed10__pid3815758__20260428_195641",
    "mistral_sciq": "sciq__mistralai__Mistral-7B-Instruct-v0.3__seed10__pid3091374__20260513_143011",
    "llama_math":   "answerable_math__meta-llama__Llama-3.1-8B-Instruct__seed10__pid2950350__20260513_124800",
    "mistral_math": "answerable_math__mistralai__Mistral-7B-Instruct-v0.3__seed10__pid3089414__20260513_142744",
    "qwen_math":    "answerable_math__Qwen__Qwen2.5-7B-Instruct__seed10__pid2950888__20260513_124831",
    "gemma_sciq":     "sciq__google__gemma-3-12b-it__seed10__pid1154278__20260609_192835",
    "gemma_triviaqa": "trivia_qa_nocontext__google__gemma-3-12b-it__seed10__pid1543419__20260610_104719",
    "gemma_math":     "answerable_math__google__gemma-3-12b-it__seed10__pid1175779__20260609_194248",
}

N_GENERATIONS = 10


def load_labels(pair):
    """id -> our open_text_label, from the ranking experiment test split."""
    path = DATA_DIR / f"ranking_experiment_{pair}" / "splits" / "test.jsonl"
    labels = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            labels[d["id"]] = bool(d["open_text_label"])
    return labels


def build_ready_pair(pair, run):
    labels = load_labels(pair)
    test_ids = set(labels)
    src = SU_DIR / run / "combined_generations.jsonl"

    written = 0
    out_path = OUT_DIR / f"{pair}.jsonl"
    with open(src) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            for qid, ex in rec.items():
                if qid not in test_ids:
                    continue
                gens = [str(r[0]).strip() for r in ex.get("responses", [])[:N_GENERATIONS]]
                gens = [g for g in gens if g]  # drop empties
                if len(gens) < 2:
                    continue
                mla = ex.get("most_likely_answer", {}) or {}
                fout.write(json.dumps({
                    "id": qid,
                    "question": ex.get("question", ""),
                    "generations": gens,
                    "label": labels[qid],
                    "run_accuracy": mla.get("accuracy"),
                }) + "\n")
                written += 1
    logger.info(f"[{pair}] wrote {written}/{len(test_ids)} test questions -> {out_path.name}")
    return written


def main():
    for pair, run in READY_RUNS.items():
        build_ready_pair(pair, run)
    logger.info("Done. (Gap pairs qwen_sciq + triviaqa x3 are produced by regenerate_gaps.py.)")


if __name__ == "__main__":
    main()