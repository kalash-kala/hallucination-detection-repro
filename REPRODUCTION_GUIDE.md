# Hallucination-Detection Reproduction Package

Self-contained package to reproduce the **Part 9 (band-stratified correctness
AUROC)** and **Part 11 (band-specialist LR)** results of the per-category
hallucination-detection study, for all **12 (model, dataset) pairs**:

- **Models:** Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, gemma-3-12b-it
- **Datasets:** SciQ, TriviaQA (no context), Answerable-Math

`classifier_a` (distractor-feature probe) is **excluded** — it failed validation
and its rows are dropped from reproduction.

---

## 1. What is being reproduced

For each (model, dataset) pair, every test example is a question with the model's
greedy answer, labelled on two axes: **correctness** (LLM verdict vs ground truth)
and **confidence** (semantic-entropy split), giving four categories —
IH (incorrect-high-conf), IL, CH, CL.

**Part 9** asks: given a supervised hallucination detector fit on the train split,
how well does it separate incorrectness *within and across confidence bands*
(IHvCH, ILvCL, pooled IHvC/ILvC/CHvI/CLvI, overall IvC)?

**Part 11** asks: does a *dedicated* classifier trained only on one band's subset
({IH,CH} or {IL,CL}) beat the generalist sliced to that band?

### Methods covered

| Family | Methods | Signal |
|---|---|---|
| Spectral-attention (prior work) | LapEigvals, AttnEigvals, AttnLogDet | top-k eigenvalues / log-det of attention (Laplacian) diagonals |
| Hidden-state LR | hs_lr (wide / narrow / peak_only) | bucketed hidden-state features at CV-picked peak layers |
| Unsupervised baselines | entropy_only, SNNE (best of 7 measures) | sampling-based semantic uncertainty over 10 generations |
| 2-axis combos | lap_only, hidden_only, entropy+lap, entropy+hidden, lap+hidden, entropy+lap+hidden | concatenated features → StandardScaler→LR |
| SNNE combos | snne+lap, snne+hidden, lap+hidden+snne, entropy+lap+hidden+snne | best SNNE score as an extra feature axis |
| Sinkhole | sink×‖V‖+hidden (+ sink-only ablations in Part 9) | attention-sink score × value norms, concatenated with hidden states |

---

## 2. Package layout

```
├── reproduce.sh                  one-command orchestrator (stages below)
├── environment/                  conda specs: semantic_uncertainty.yml, snne.yml
├── data/uncertainty_runs/        12 uncertainty-run CSVs (entropy, verdicts,
│                                 categories, 10 sampled generations per question)
├── recovery-gaps-data/data/ranking_experiment_{model}_{dataset}/
│   └── splits/                   FROZEN train/test splits (the ground truth of
│                                 every experiment; never regenerated)
├── results/snne_baseline/generations{,_train}/   SNNE input generations (shipped)
├── scripts/
│   ├── ranking/                  GPU: hidden-state cache + sidecars
│   ├── lapeigvals_baseline/      GPU: attention diagonals; CPU: baseline CV
│   ├── sinkhole/                 GPU: value norms; CPU: sink features
│   ├── snne_baseline/            GPU (snne env): SNNE scoring
│   ├── classifier/               CPU: hs_lr features + classifiers
│   ├── feature_analysis/         CPU: 2-axis feature loaders/combos
│   ├── per_category_analysis.py            CPU: per-example baseline scores
│   ├── per_category_pairwise_auroc.py      CPU: **Part 9**
│   └── per_category_band_specialist_auroc.py  CPU: **Part 11**
└── third_party/SNNE/             vendored SNNE repo (MIT-licensed; sys.path import)
```

Directory layout inside the package mirrors the original experiment repo, so all
ported scripts run byte-identically; the only patches are (a) the entropy-CSV map
now points at `data/uncertainty_runs/`, (b) the SNNE import path defaults to
`third_party/SNNE/`.

## 3. Data flow (three tiers)

```
TIER 0 (shipped, frozen)            TIER 1 (GPU, run once per system)         TIER 2 (CPU, minutes–hours)
────────────────────────            ──────────────────────────────────         ───────────────────────────
splits/{train,test}.jsonl  ──┬───▶  01 hidden_layer{L}.npz + sidecars   ──┬─▶  05 LapEigvals CV configs
data/uncertainty_runs/*.csv ─┤      02 attention diagonals (diags/*.pt)   ├─▶  06 hs_lr peak layers + LRs
snne generations*.jsonl     ─┴───▶  03 value norms (value_norms/*.pt)     ├─▶  07 per_example_scores.csv
                                    04 SNNE scores (scores*/<pair>.csv)  ─┴─▶  08 Part 9  ▶ 09 Part 11
```

Every Tier 1 output is **cached to disk and never recomputed** (each script
checks for its outputs and skips finished work), which is exactly how the
original experiments amortised GPU cost — rerunning `reproduce.sh extract`
after an interruption resumes where it stopped.

## 4. How to run

```bash
# 0. one-time setup
conda env create -f environment/semantic_uncertainty.yml
conda env create -f environment/snne.yml
export REPRO_GPU=0                     # which GPU to use
# optionally: export REPRO_PY=/path/to/envs/semantic_uncertainty/bin/python
#             export REPRO_PY_SNNE=/path/to/envs/snne/bin/python

# 1. GPU extraction (~several hours/pair for 01; 02–04 are lighter)
./reproduce.sh extract
./reproduce.sh snne

# 2. CPU stages
./reproduce.sh prep
./reproduce.sh analyze
```

Or `./reproduce.sh all`. Final outputs land in `results/per_category_analysis/`:

- `per_category_pairwise_auroc.csv` — **Part 9** (per pair × method × contrast)
- `per_category_band_specialist_auroc.csv` — **Part 11** (specialist vs generalist per band)

### Skipping the GPU entirely (optional artifacts tarball)

A precomputed-artifacts tarball (`repro_artifacts_YYYYMMDD.tar.zst`, ~36 GB,
zstd-compressed; ~39 GB raw — the hidden-state npz caches are already internally
compressed, so zstd gains little) contains every Tier 1 output from the original
run. Restore it
into the package root and jump straight to the CPU stages:

```bash
tar -I zstd -xf repro_artifacts_20260703.tar.zst -C /path/to/this/package
./reproduce.sh prep && ./reproduce.sh analyze
```

## 5. What each stage does (and highlights)

| Stage | Script | What it produces / why it matters |
|---|---|---|
| 01a | `ranking/extract_cache.py` | One teacher-forced forward pass per (sample, candidate), `output_hidden_states=True`; stores the **last-answer-token hidden state at every layer** + length-normalised log-prob (s_ext). Foundation for every hidden-state method. |
| 01b | `ranking/greedy_sidecar.py`, `compute_train_sidecars.py` | Maps each example id to the model's **own greedy answer** and scores it — the deployment-honest view (a detector only ever sees the model's real output). |
| 01c | `ranking/recover_greedy_hidden.py` | Correct samples' greedy answers were never in the candidate pool, so their hidden states are missing from 01a; this fills the gap so hidden-state methods score the **full** test set, not a biased subset. |
| 02 | `lapeigvals_baseline/extract_attention.py` | Teacher-forced pass with **eager attention** capturing attention matrices, reduced on the fly to per-layer/head **diagonals** — all the LapEigvals family needs, at ~1/seq-len the storage. |
| 03 | `sinkhole/extract_value_norms.py` | Per-head **value-vector norms** at the answer positions; combined with attention-sink scores they form the `sink×‖V‖` feature (our best single addition on top of hidden states). |
| 04 | `snne_baseline/dump_{snne,train}_scores.py` | DeBERTa-entailment semantic clustering over the 10 shipped generations → 7 SNNE measures per question (test + train). Runs in the **separate `snne` conda env**. |
| 05 | `lapeigvals_baseline/train_lapeigvals.py` | 5-fold CV per pair over top-k / PCA grids → frozen configs the baselines use downstream (`all_pairs_metrics.csv`). |
| 06 | `classifier/compute_peak_layers.py`, `train_classifier.py` | Finds each model's peak detection layers, fits the bucketed hs_lr classifiers, saves `layer_stats.pkl` (the normalisation stats every later hs_lr evaluation reuses). |
| 07 | `per_category_analysis.py --snne` | Per-example P(hallucination) for the spectral baselines + best-SNNE → `per_example_scores.csv` (Part 9 reads baseline scores from here). |
| 08 | `per_category_pairwise_auroc.py` | **Part 9 table**: for every method × pair, within-band (IHvCH, ILvCL, …) and pooled (IHvC, ILvC, CHvI, CLvI) AUROCs. |
| 09 | `per_category_band_specialist_auroc.py` | **Part 11 table**: HI-band and LO-band specialist LRs vs the generalist, `delta = specialist − generalist`. |

### Key design invariants

- **Frozen splits.** `splits/{train,test}.jsonl` ship with the package and are never
  regenerated — train/test membership is identical to the original study by construction.
- **No target leakage.** All CV/selection (top-k, PCA, peak layers, best SNNE measure)
  happens on train only; band-specialists re-select on the band-restricted train subset.
- **Deterministic CPU stages.** Fixed seeds (42/10) throughout; given identical Tier 1
  artifacts, Tier 2 outputs reproduce exactly.
- **Two environments.** Everything runs in `semantic_uncertainty` except SNNE scoring
  (`snne` env — different torch/transformers pins for the entailment model).

## 6. Headline findings these tables support

1. **Hidden-state features dominate single-signal baselines** in-domain:
   `lap+hidden` ≈ 0.80 pooled IvC vs ≈ 0.74 for LapEigvals alone.
2. **`sink×‖V‖+hidden` is the best overall method** (mean IHvC ≈ +0.9pp over
   `lap+hidden` pooled), especially on confident hallucinations (IH).
3. **Part 11's main result:** band-specialist training helps *only* the hs_lr
   family (+7–8pp in the HI band); for combo methods the generalist is already
   at ceiling — a dedicated band model is not worth it.
4. **Unsupervised SNNE/entropy are strong on low-confidence errors but blind to
   confident hallucinations** (entropy IHvC ≈ 0.50 = chance) — the core motivation
   for supervised internal-state detectors.

## 7. Caveats

- GPU stage needs ≥80 GB VRAM for gemma-3-12b-it at bf16 with eager attention
  (attention capture is memory-hungry); llama/mistral/qwen fit comfortably.
- HF gated-model access is required for Llama and Gemma checkpoints.
- SNNE scoring downloads DeBERTa-v2-xxlarge-MNLI on first run.
- `per_category_analysis.py` skips `classifier_a` automatically when distractor
  sidecars are absent (they are not shipped — the method is excluded).
- Exact-number reproduction holds for fixed Tier 1 artifacts. If you re-extract
  on different hardware/driver stacks, expect tiny bf16 nondeterminism in the
  hidden states; AUROCs should match to ~3 decimals.