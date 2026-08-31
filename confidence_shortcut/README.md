# confidence_shortcut

Structured reimplementation of the confidence-shortcut study: internal-state
extraction, per-pair probing, and report aggregation, across text and VLM
model×dataset pairs.

**Nothing in here runs by accident.** No CLI stage has a default action — `--plan`
prints the work units and exits, `--run` is required to do anything.

---

## Three independent components

Each is a separate package with its own dependencies. **None imports another.**
The only things between them are two versioned on-disk contracts.

```
 1. csx_extract  (GPU)          2. csx_probe  (CPU)           3. csx_report  (CPU)
    torch, transformers, PIL       numpy, pandas, sklearn        pandas only

    model + rows + images          store + arms -> fits          atomic values -> tables
           │                              │                             │
           ▼          STORE               ▼        RESULTS              ▼
     ┌───────────┐   contract      ┌──────────────┐  contract    ┌────────────────┐
     │ L0 raw/   │ ──────────────▶ │ per-pair     │ ───────────▶ │ medians, CIs,  │
     │ L1 feats/ │                 │ ATOMIC rows  │              │ verdicts, MD   │
     └───────────┘                 └──────────────┘              └────────────────┘
      per pair,                     per pair,                     ANY cohort,
      resumable                     no aggregation                chosen at RUNTIME
```

`csx_common` is the shared base — paths, the pair registry, cohorts, frozen
constants. It exists so the three can agree on *what exists* without importing
each other, and it never imports torch.

**Why this shape.** A pair becomes usable by component 2 the moment its store
entry validates, so extraction and probing never block each other. And because
component 2 emits only atomic per-pair rows — no medians, no `cohort` column —
any grouping can be medianed afterwards without refitting: LLM vs VLM,
per-dataset, per-model-size, leave-one-out. Adding a model or dataset costs one
extraction plus one report re-run; nothing already computed is touched.

`tests/test_isolation.py` enforces all of that rather than trusting it.

---

## Install

```bash
# components 2 and 3 (no torch)
pip install -e .

# add component 1
pip install -e '.[extract]'
```

Pin the interpreter the reference numbers were produced under:
`/root/miniconda3/envs/semantic_uncertainty/bin/python` — py3.11.15, numpy 2.2.6,
pandas 3.0.2, sklearn 1.8.0, torch 2.6.0+cu118, transformers 5.5.1.

---

## Configuration

Everything the roster depends on lives in `configs/`, so adding a pair is a CSV
plus a YAML entry, never a code change.

| file | holds |
|---|---|
| `pairs.yaml` | every pair: CSV, canonical model tag, HF id, layers, modality, generations folder, image root, prompt template. Also the storage roots. |
| `cohorts.yaml` | named groupings for component 3. **Conveniences, not commitments** — an arbitrary `--pairs` list works too. |
| `c_policy.yaml` | how `C` is resolved per pair: `pinned` (the 8 parity pairs) / `per_pair` (default) / `group`. |
| `frozen_constants.yaml` | seeds, LR hyperparameters, quotas, alpha ladder, bootstrap sizes, feature grids. Not knobs — the parity gate depends on them. |

Roots are overridable per-run: `CSX_STORE=/somewhere python cli_probe/...`.

### Model tags are declared, never inferred

The same backbone appears under several tags across datasets, and one is actively
misleading: **`qwen_14b` in the nq files is Qwen3-14B**, not a Qwen2.5 variant
(confirmed against the `nq__Qwen__Qwen3-14B__*` run directory). Parsing filenames
would get that wrong, so `pairs.yaml` carries HF ids.

---

## Pairs

24 active, 2 pending. `text` is a 6 × 3 grid once `qwen_nq` lands.

| dataset | models | rows/pair |
|---|---|---|
| `sciq` | llama, mistral, qwen, gemma, qwen3_14b, gemma3_27b | ~13,584 |
| `triviaqa` | same 6 | 50,000 |
| `nq` | 5 present + `qwen_nq` pending | 50,000 |
| `vqav2` | qwen25vl, gemma3_12b | 34,991 |
| `okvqa` | qwen25vl, gemma3_12b, + `pixtral12b_okvqa` pending | 14,055 |
| `advqa` | qwen25vl, gemma3_12b, pixtral12b | 3,000 |

`answerable_math` / `answerable_math_cot` are out of scope, listed explicitly in
`pairs.yaml` so the census can be asserted complete.

### Internal-state availability

| pairs | states | action |
|---|---|---|
| 8 — llama/mistral/qwen/gemma × {sciq, triviaqa} | complete, in `full_natural_data/` | reused via the legacy adapter; the parity cohort |
| 4 — gemma3_27b, qwen3_14b × {sciq, triviaqa} | none usable | extraction |
| 6 — nq | none | extraction |
| 8 — VLM | none | extraction |

> The `certain_mispredictions_results/.../gemma3_27b_*.pt` diagonals look usable
> but are the **legacy 1,400-row sampled runs** (980 train / 420 test) against
> 13.5k–50k-row CSVs, with no `hs_*` at all. The store classifies them
> `missing_raw` rather than quietly building a 1,400-row pair.

Only VLM pairs need a generations folder, and only for `image_path` — the one
field no run CSV carries. Text extraction is CSV-driven: `question`,
`low_t_generation` and the 10 sampled answer strings are all in the CSV.

---

## Usage

```bash
PY=/root/miniconda3/envs/semantic_uncertainty/bin/python

# L2: ingest every run CSV. No GPU, covers all pairs, unblocks arm construction.
$PY cli_probe/00_ingest_uq.py --plan
$PY cli_probe/00_ingest_uq.py --run

# validate a store entry — the handoff between components 1 and 2
csx-store verify --pair qwen25vl_vqav2
csx-store verify --all
```

---

## Contracts

- [`store_spec/STORE_CONTRACT.md`](store_spec/STORE_CONTRACT.md) — components 1 → 2.
  L0/L1/L2 layout, required `meta.json` fields, validation rules.
- [`results_spec/RESULTS_CONTRACT.md`](results_spec/RESULTS_CONTRACT.md) — components 2 → 3.
  Atomic per-pair tables; **no column is named `cohort`**.

Two details in the store contract are worth knowing before reading the code:

**Hidden states are reduced destructively.** Raw `[N, L, D]` is ~10 GB per pair at
gemma-3-27b's 62 layers, so extraction computes peak layers on train rows only,
writes the pooled `hs_*` matrices, and deletes the raw shards — as
`02_reduce_hidden.py` does. Changing the pooling scheme later means re-extracting.

**Spectral diagonals are reduced non-destructively.** Top-50 per head retains
everything all four spectral families need, so that side can be re-derived
without a GPU. But `attn_logdet [L,H]` is stored *separately and required*:
`attnlogdet` is a mean over **all** positions, not a top-k feature, so top-k-only
storage would leave that family silently wrong rather than missing.

---

## Tests

```bash
$PY -m pytest tests/ -q
```

Fixture-based and offline — no GPU, no store, no network. They cover the census,
the component boundary, the L2 gates, the store contract's failure modes, and
subsampling.

The gates are the valuable part. Each encodes a way the data could be
wrong-but-plausible and demands a loud failure: an inverted band orientation
(H must mean **low** entropy), correctness taken from `accuracy` instead of
`LLM_verdict` (they genuinely disagree), a `boundary` category silently bucketed,
an image span overlapping the answer, diagonals whose `L` disagrees with the
model, a missing `attn_logdet`.

---

## Status

| milestone | component | state |
|---|---|---|
| M1 skeleton, registry, config | shared | done |
| M2 L2 UQ ingest | shared | done |
| M3 the two contracts + verifier | shared | done |
| M4 rows, subsample, spans, model loading | 1 | subsample done; rows/spans/models in progress |
| M5 phase 1 (sdpa) + reduce | 1 | not started |
| M6 phase 2 (eager attention) | 1 | not started |
| M7 store read/build + legacy adapter | 2 | not started |
| M8 arms | 2 | not started |
| M9 probes + metrics | 2 | not started |
| M10 experiments | 2 | not started |
| M11 aggregation + rendering | 3 | not started |
| M12 parity + docs | 3 | not started |
