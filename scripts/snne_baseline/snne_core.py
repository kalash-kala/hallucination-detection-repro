"""Wandb-free core of SNNE's text-only uncertainty measures.

This module provides the SNNE / semantic-uncertainty baseline algorithms WITHOUT
any wandb dependency (the project uses CSV/JSON, not wandb).

Strategy:
  - The pure algorithm functions in `snne.uncertainty.utils.entropy_utils` and
    `snne.uncertainty.utils.eval_utils` do NOT import wandb, so we import them
    directly from the installed `snne` package.
  - `EntailmentDeberta` and `get_semantic_ids_using_entailment` live in
    `semantic_entropy.py`, which imports wandb at module load. We re-implement
    just those two pieces here (verbatim logic, wandb stripped) so nothing in our
    pipeline touches wandb.

Methods covered (all "text-only" — need only candidate texts + correctness label):
  NumSet, LexSim, SumEigv (spectral), Degree, Eccentricity, LUQ, SNNE.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make the SNNE repo importable regardless of cwd (it is not pip-installed; the
# `snne` package only resolves when its repo root is on sys.path).
# [repro patch] default to the copy vendored inside this package.
_DEFAULT_SNNE = str(Path(__file__).resolve().parents[2] / "third_party" / "SNNE")
SNNE_REPO = os.environ.get("SNNE_REPO", _DEFAULT_SNNE)
if SNNE_REPO not in sys.path:
    sys.path.insert(0, SNNE_REPO)

# Pure, wandb-free algorithm functions from the snne package.
from snne.uncertainty.utils.entropy_utils import (  # noqa: F401
    entailment_similarity_matrix,
    lexical_similarity_matrix,
    compute_lexical_similarity,
    get_spectral_eigv,
    get_degreeuq,
    get_eccentricity,
    get_luq_pair,
    snne,
    greedy_clustering,
)
from snne.uncertainty.utils.eval_utils import (  # noqa: F401
    auroc,
    auarc,
    aucpr,
    is_binary_list,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBERTA_MODEL = "microsoft/deberta-v2-xlarge-mnli"


class EntailmentDeberta:
    """DeBERTa-v2-xlarge-MNLI entailment scorer (wandb-free copy of SNNE's class).

    Behaviourally identical to snne.uncertainty.uncertainty_measures.semantic_entropy
    .EntailmentDeberta, with `torch.no_grad()` added (inference-only optimisation,
    no effect on outputs) and defensive truncation (answers are short, so this
    almost never triggers).
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA_MODEL).to(DEVICE)
        self.model.eval()
        # Per-question cache: the strict matrix, LUQ matrix and semantic-id
        # clustering all query the same ordered (text1,text2) pairs. Caching the
        # logits avoids recomputing them. Call clear_cache() between questions.
        self._cache = {}

    def clear_cache(self):
        self._cache.clear()

    def _logits(self, text1, text2):
        key = (text1, text2)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        inputs = self.tokenizer(
            text1, text2, return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        self._cache[key] = logits
        return logits

    def check_implication(self, text1, text2, *args, **kwargs):
        # deberta-mnli: 0=contradiction, 1=neutral, 2=entailment
        logits = self._logits(text1, text2)
        return torch.argmax(F.softmax(logits, dim=1)).cpu().item()

    def get_similarity_score(self, text1, text2, strict_entailment=True, exclude_neutral=True):
        logits = self._logits(text1, text2)
        s = F.softmax(logits, dim=1)
        if strict_entailment:
            return s[:, 2].cpu().item()
        elif exclude_neutral:
            # LUQ paper
            return (s[:, 2] / (s[:, 2] + s[:, 0])).cpu().item()
        else:
            # w = (0, 0.5, 1) as in KLE's paper
            return (s[:, 2] + s[:, 1] * 0.5).cpu().item()


def get_semantic_ids_using_entailment(strings_list, model, strict_entailment=False, example=None):
    """Cluster predictions into semantic-equivalence sets (wandb-free copy)."""

    def are_equivalent(i, j):
        a = model.check_implication(strings_list[i], strings_list[j], example=example)
        b = model.check_implication(strings_list[j], strings_list[i], example=example)
        assert (a in [0, 1, 2]) and (b in [0, 1, 2])
        if strict_entailment:
            return (a == 2) and (b == 2)
        implications = [a, b]
        # not a contradiction, and not both-neutral
        return (0 not in implications) and ([1, 1] != implications)

    return greedy_clustering(strings_list, are_equivalent)