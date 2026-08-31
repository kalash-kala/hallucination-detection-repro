# Part C on the VLM cohort — full parity with the 8-pair QA run

**Status:** plan. Nothing here is implemented yet.
**Goal:** run the *identical* experiments on 9 VLM pairs so that "does it replicate?" is a
question about the phenomenon, not about a difference in method.

**Reference artifacts (the three targets, reproduced section for section):**

| target doc | produced by | what it needs |
|---|---|---|
| `dse_results/routed_vs_generalist_fixedC/DISCUSSION_NOTE.md` | `54` (+`51`, `55`) | sampled internals |
| `dse_results/sr_xauc_platt/SR_XAUC_RESULTS.md` | `58`/`65` over `64`'s merged grid | sampled internals |
| `dse_results/METRIC_ROUTER_REPORT_selected_c.md` | `35`/`36`/`63` | sampled internals **+ 7 NLI metrics** |

---

## 1. What full parity actually requires

The QA Part C rests on two data products that **do not exist for any VLM pair**. Everything else
is CPU work we already have the scaffolding for.

| product | what it is | VLM status |
|---|---|---|
| **sampled internals** | internal states of the **10 temperature-1 generations** per row, for all 7 families | **missing** — GPU |
| **the 7 non-DSE UQ metrics** | `num_set`, `lexical_sim`, `sum_eigv`, `degree`, `eccentricity`, `luq`, `snne` | **missing** — GPU (NLI) |

Confirmed by inspection: `csx_store/raw/<pair>/` holds `hs/` and `diag/` from the single greedy
pass only; `uq/uq_rows.parquet` carries `entropy` and `c_metric` and nothing else.

Without the first, `@sampled`, every `*_cm` cost-matched control, and `spec1_z_hier` are all
undefined — which is 8 of the 12 rows in the SR-xAUC table and the entire DISCUSSION_NOTE.
Without the second, the METRIC_ROUTER report has one metric instead of eight.

**So the plan is GPU-first.** The CPU half is cheap and largely already designed; the honest
critical path is two extraction passes.

### The good news, measured rather than assumed

**The sampled feature store is kept, same as the QA precedent.** Checked: the QA run's sampled
store is still on disk at `/data/kalashkala/dse_data/sampled/features/`, **37G**, across all 12
QA pairs — never deleted, and nothing has needed deleting it. The rationale for keeping is
unchanged: ~65 GPU-hours is the expensive, hard-to-redo part of this plan, and if a later idea
needs the sampled internals again (a new scorer, a different aggregation, an experiment nobody
has specified yet), re-running extraction to get them back would burn the same scarce resource a
second time.

> **Correction (2026-08-26): the earlier 35–45G estimate was wrong — the real figure is ~204G.**
> It was extrapolated from the QA per-pair average, which does not transfer: the QA pairs are
> text-only and therefore have **one** segment (`all`), while every VLM pair stores **three**
> (`all`/`image`/`text`), and the attention `diag/` arrays — not `hs/` — dominate the footprint
> at ~79% of bytes. Measured from the three completed advqa pairs: **496 KB/row** for qwen25vl
> and gemma3_12b, **750 KB/row** for pixtral12b, over 373,328 unique rows → **~204G** against
> 321G free on `/data`. See the segment note in §3/M19 for the 3× lever this exposes.

Kept at
`/data/kalashkala/csx_store/sampled/<pair>/` — a labeled subtree, distinct from the permanent
`raw/`/`hs/`/`diag/` L0 entries, so what it is and why it exists stays legible without implying
it is disposable.

**Sampled extraction is ~3.9×, not 10×.** `24_extract_sampled.py` keys on unique
`(id, answer_text)`, and VQA answers repeat heavily. Measured across the 9 pairs:

| pair | rows in store | unique (id, text) | dedup saving |
|---|---:|---:|---:|
| `qwen25vl_advqa` | 3,000 | 13,715 | 54.3% |
| `gemma3_12b_advqa` | 3,000 | 5,854 | 80.5% |
| `pixtral12b_advqa` | 3,000 | 9,488 | 68.4% |
| `qwen25vl_okvqa` | 14,055 | 91,538 | 34.9% |
| `gemma3_12b_okvqa` | 14,055 | 34,425 | 75.5% |
| `pixtral12b_okvqa` | 14,055 | 40,981 | 70.8% |
| `qwen25vl_vqav2` | 15,000 | ~83,000 | 44.6% |
| `gemma3_12b_vqav2` | 15,000 | ~31,200 | 79.2% |
| `pixtral12b_vqav2` | 15,000 | ~62,700 | 58.2% |
| **total** | **96,165** | **~373,000** | **61%** |

(vqav2 figures scaled by the 15,000/34,991 subsample already applied in the store.)

`gemma3_12b` deduplicates hardest (80%) because it answers VQA in short canonical forms;
`qwen25vl` least (35–54%) because it produces longer, more varied phrasings. That spread is
itself worth a line in the write-up — it is a measure of answer-space concentration per model.

**One eager pass yields all 7 families.** The greedy pipeline runs two sweeps (`21_hidden` sdpa,
`23_attention` eager); `24`'s design does hidden *and* attention in a single eager pass. So the
sampled per-row cost is the eager cost, which is what dominates anyway.

**The NLI metrics need no VLM.** All 7 are functions of `(question, 10 answer strings)` through a
DeBERTa entailment matrix — `scripts/snne_baseline/snne_core.py` + `score_shard.py`, already
written and already run for QA. The VLM CSVs carry `n_generations` (verified: exactly 10 per row,
all 9 pairs). This is a text-only pass over VLM outputs; the vision towers are never loaded.

---

## 2. Measured cost, from our own logs

Greedy extraction rates, taken from the runs that just completed:

| pair | phase2 (eager) rate | full pipeline |
|---|---|---|
| `gemma3_12b_vqav2` | 2.01 rows/s | 134.0 min / 15,000 rows |
| `gemma3_12b_okvqa` | 2.00 rows/s | 126.9 min / 14,055 rows |
| `pixtral12b_vqav2` | 1.64 rows/s | 168.9 min / 15,000 rows |
| `pixtral12b_okvqa` | 1.50 rows/s | — |

Extrapolating to the unique-pass counts above:

| stage | passes | rate | GPU-hours |
|---|---:|---|---:|
| sampled extraction — `qwen25vl` ×3 | 188,300 | ~2.5/s | ~21 |
| sampled extraction — `gemma3_12b` ×3 | 71,500 | ~2.0/s | ~10 |
| sampled extraction — `pixtral12b` ×3 | 113,100 | ~1.6/s | ~20 |
| NLI metrics (DeBERTa-v2-xlarge, 90 pair-calls/row × 96,165 rows ≈ 8.7M) | — | batched | ~12–15 |
| **total** | | | **~63–66 GPU-hours** |

On 2× A100 that is **~32–36 hours wall clock**. Storage: 7 families × 373k rows fp16 ≈ 10–12 GB;
`/data` has 337 G free, so no pressure. `/` is at 85% — nothing may be written there.

### The scheduling reality

**Both GPUs are currently occupied by another user** — `hrishi/TREA-ORCA/finetune/af3.py`, two
processes, 68 GB and 38 GB, 99–100% utilisation, ~3¼ hours in. Our probe stage is CPU-only so
nothing is blocked *today*, but ~65 GPU-hours cannot be scheduled without coordinating. This
needs a conversation before M19 is launched, not after.

---

## 3. Milestones

### Status as of 2026-08-29

| milestone | state | evidence |
|---|---|---|
| **M13** `csx_probe/routing/` | **built** | `experts.py`, `router.py`, `pooling.py`; 22 tests in `tests/test_routing.py` |
| **M14** greedy router == `sep` | **built** | asserted by `test_greedy_router_is_the_sep_probe_with_labels_swapped` |
| **M15** `experiments/routed_grid.py` | **built** | + `cli_probe/07_routed_grid.py`; 8 scorers × 3 routers verified end-to-end |
| **M16** `experiments/lever_a.py` | **built** | + `cli_probe/08_lever_a.py`; 4 gates, non-zero exit on failure |
| **M17** `csx_report/tables/srxauc.py` | **built** | commutation, `G`, `gap`, `asym`, §17.3 guard |
| **M18** validation harness | **built, PASSING** | reproduces the published QA SR-xAUC table to **5.6e-17** (target was 1e-6) |
| — sampled aggregator | **built** | `csx_probe/store/sampled.py`, validated on the real `qwen25vl_advqa` sampled store |
| **M19** sampled extraction | **COMPLETE, 9/9** | `qwen25vl_okvqa` landed 2026-08-28 15:10 (91,538 unique rows, 23.6 h). All 9 verified: `unique.parquet` matches `n_unique`, manifests are exactly 10x the roster, 0 skipped-empty, 0 short-seq-guard hits |
| **M20** NLI metrics | **COMPLETE, 9/9** | all 9 VLM pairs at full roster coverage (3000 advqa / 14055 okvqa / 15000 vqav2, matching `rows.parquet` exactly) |
| **M21** `best_split` thresholds | **built, gate PASSING 9/9** | `csx_probe/routing/best_split.py`, `experiments/band_thresholds.py`, `cli_probe/02_band_thresholds.py`; 13 tests; reproduces every pair's stored `DSE_threshold` exactly |
| **M22** routed grid + `cloud` | **built; grid RUNNING 2026-08-28** | `cloud` was missing entirely until now -- `store/sampled.py` had only `mean`/`mean_std`/`greedy_mean_std`. Ported from `25_sampled_router_ovr.py` at `CLOUD_EIG=3` (10 dims), pinned against a literal transcription of the original in `tests/test_sampled.py`. The routed grid now carries a `sampled_scheme` dimension, sharing the expensive greedy-side fits across schemes, so `cloud` and `greedy_mean_std` are compared under one partition, one `C` and one frozen basis |
| **M23** metric-router grid | **built, first pair PASSING** | `experiments/metric_router.py` + `cli_probe/09_metric_router.py`; 12 tests. `gemma3_12b_advqa`: 324 cells, diagonal 0.932, off-diagonal 0.917, mean off-diagonal band agreement 0.872. Remaining 8 pairs queued behind stage 07/08/06 |
| **M24** | not built | needs M22 + M23 output across all 9 |

**M18 is the load-bearing result here.** Fed the published
`routed_vs_generalist_fixedC/per_pair_long.csv`, our `srxauc.py` reproduces
`58_sr_xauc.py`'s output at machine precision across all 40 rows and all 320 atomic cells —
every scalar column, the unit-level bootstrap CIs (exact, same seed and generator), and the
limiting-cell labels. That is the strongest available evidence that our Part C is the *same
experiment* rather than a plausible-looking neighbour of it, and it was achievable with no GPU
and no new pipeline run.

Ordered so that **every CPU milestone that can be built before the GPUs free up, is**. The code
is written and tested against synthetic fixtures first; the GPU passes then feed finished code.

### Phase I — CPU, buildable now (no GPU dependency)

**M13 — `csx_probe/routing/`: experts, routers, pooling.**

- `experts.py` — `fit_experts(X_tr, cat_tr, *, family, c, mode)` → `{HI: (tf, lr), LO: (tf, lr)}`.
  Transform **refit inside each band** (band subsets have different covariance; sharing the
  natural transform leaks band structure into the expert's input space).
  `mode="same_band"` selects train rows by true band; `mode="hier"` by out-of-fold router band.
- **The leakage trap, in code and in a test.** The `hier` OOF labelling must come from a router
  refit inside each of 5 `StratifiedKFold(y_hi, shuffle, 42)` folds — *the whole pipeline*,
  variance filter and transform included. A router fit on all of train and then asked about those
  same rows predicts at training accuracy, its predicted bands nearly equal the true bands, and
  `hier` silently collapses into `spec1_z`. `test_hier_oof_bands_differ_from_true_bands` asserts
  the disagreement rate tracks the router's held-out error rate — not merely that it is non-zero.
- `router.py` — `greedy`, `sampled`, `oracle`. **`oracle` is computed and never tabled**; it is
  not deployable and exists only because §20's affine-invariance gate is undefined without it.
- `pooling.py` — `z`, `platt`, `platt_prior` (Lever A), `proba`. All four live in one module
  because the paper's argument is that they are **the same 2-parameter affine-per-band family**,
  differing only in how the two numbers are chosen:

  | pooler | map | parameters from |
  |---|---|---|
  | `z` | `(f_b − μ_b)/σ_b` | the raw logit's distribution shape, label-blind |
  | `platt` | `a_b·f_b + b_b` | fit against labels, **train only** |
  | `platt_prior` | `a_b·f_b + (b_b − logit π_b)` | as above, minus the band's counted base rate |
  | `proba` | `predict_proba` | *not* neutral — re-imports the band offset; the negative control |

**M14 — one structural simplification worth recording.** In legacy `54` the greedy router is
`make_lr(kind).fit(Ztr, y_hi)`; our `sep` head is `1[cat in L_CATS]` — **the same binary target
with the labels swapped.** The greedy router *is* the `sep` probe, and its band AUROC is already
in `per_pair_long`. So the 1-generation tier needs no new router machinery, and the write-up gets
a pointed observation: the probe a naive design uses *as* the error scorer is, used correctly,
only a router.

**M15 — `experiments/routed_grid.py`** → `routed_long`, one row per
`(pair, family, segment, C, train_arm, test_arm, router, scorer, contrast, AUROC, …)`.
Schema-compatible with `per_pair_long` plus `router`/`scorer`, so `csx_report` reuses its loaders
and the two concatenate. Same parallel pattern as everywhere else —
`Parallel(n_jobs, backend="loky", inner_max_num_threads=1, max_nbytes="1M")`, plan-then-execute so
bootstrap indices come off the seeded stream serially, never in worker-completion order.

**M16 — `experiments/lever_a.py`, the four gates.** These are what make Lever A a proof rather
than a number that moved:

1. **Drift** — `platt` reproduces the base grid where the designs coincide.
2. **The affine identity** — under **oracle** routing each atomic cell is scored by exactly one
   expert, and AUROC inside one band is invariant to a positive affine map, so `IHvCH` and
   `ILvCL` must be **bit-identical** between `z` and `platt_prior` (published: 0.00e+00, 8/8).
   Under a real router a cell is scored by a *mixture* of two affine maps, which is not affine,
   so the identity **must break** — and the gate asserts it breaks, because if it holds under
   `sampled`/`greedy` then routing is not actually being applied.
3. **ECE falls** per band (~10× in the published run).
4. **Train-only provenance** for both Platt parameters and the prior.

Non-zero exit on any failure.

**M17 — `csx_report/tables/srxauc.py`.** SR-xAUC per unit → median over the group.

> **The trap that would silently inflate every number in Part C.** `min` does not commute with
> `median`. The guide's §5 quantity minimises over 16 candidates (4 atomic cells × 4 test arms)
> **inside** each `model × dataset` unit, and only then medians across units. The commuted
> quantity is systematically larger; legacy `58` carries it as `approx_min_of_medians` and
> **never calls it SR-xAUC**. We keep both columns under both names, and
> `test_srxauc_min_does_not_commute_with_median` builds a frame where they differ and asserts we
> report the smaller one.

Also rendered, non-optionally:
- **`G` (stability)** beside SR-xAUC always — a constant scorer has `G = 0` and SR-xAUC = 0.5, so
  low `G` means something only next to a high SR-xAUC.
- **`gap = pooled IvC − SR-xAUC`** — how much of the headline was resting on favourable
  confidence composition. This column *is* the metric's argument.
- The **`asym = IHvCL − ILvCH`** ladder, in **one** convention (the report's). The legacy prose
  uses the reversed sign in places; that discrepancy must not survive the port.
- **The §17.3 guard, enforced in code.** `entropy_only` scores 0.000 *by construction* — the
  bands are cut from entropy — so the renderer emits a mandatory footnote and refuses to place it
  in a ranked comparison. Reporting semantic-entropy methods as "defeated" here is the single
  easiest way for this work to be wrong in public.

**M18 — validation harness.** Before any VLM number is believed, the same code must reproduce
the *QA* numbers. Once the legacy adapter lands (backlogged to the qa8+nq rerun) this becomes a
parity gate; until then it is a **fixture test** built from the published CSVs: feed
`routed_vs_generalist_fixedC/per_pair_long.csv` through our `srxauc.py` and assert we recover
`SR_XAUC_RESULTS.md`'s table to 1e-6. That is achievable **now**, needs no GPU, and it is the
strongest available evidence that our Part C is the same experiment.

### Phase II — GPU, gated on machine availability

**M19 — sampled generation manifest + extraction** (`cli_extract/26_sampled_manifest.py`,
`27_extract_sampled.py`).

- Manifest: unique `(id, answer_text)` across the 10 sampled generations **plus** the greedy
  answer, with a slot map `id → 10 feature rows` so the aggregator can gather them.
- Extraction: one eager pass per unique text → all 7 families, fp16, row order == manifest key
  order. Per-pair checkpoints, resumable, **launched with `setsid`** (nohup alone dies with the
  Claude Code process).
- **Layer statistics must be recomputed, not inherited.** `22_reduce.py` computed per-layer
  (μ, σ) over natural-train greedy rows and did not persist them, then deleted the raw cache. The
  sampled pass recomputes them by the identical definition in the same process and same eager
  model, so the features it produces are numerically comparable to the greedy ones. Getting this
  wrong is silent — the numbers stay plausible.
- **Sequence-length guard:** a sampled answer can be shorter than the greedy one, so a row with
  `S < top_k` is edge-padded (the `exp3_feature` convention) and *counted in the manifest*. For
  VLM the sequence is image-token-dominated so this should be near-zero, but the count is
  reported rather than assumed.
- **Order `advqa` first** — 29k passes total across all three model families, so every processor
  quirk (gemma-3 `token_type_ids`, Qwen2.5-VL flattened `pixel_values`, Pixtral `image_sizes` /
  `pad_token`) is exercised at the cheapest possible price. These quirks corrupt *silently*
  rather than erroring.
- Written to `/data/kalashkala/csx_store/sampled/<pair>/` — a labeled subtree, distinct from the
  permanent `raw/`/`hs/`/`diag/` L0 entries but **kept**, not purged, matching the QA precedent
  and this run's storage budget (§ above).
- **All 7 families are extracted, not just `hs_*`.** The sampled store mirrors the greedy store's
  array set exactly: `hs/{all,image,text}.npz` carries `hs_wide`/`hs_narrow`/`hs_peak_only`, and
  `diag/{all,image,text}.npz` carries `attn_topk` (→`attn_eigvals`), `lap_topk` (→`lapeigvals`),
  `attn_logdet` (→`attnlogdet`), and `sink_topk`+`sink_vnorm_topk` (→`sink`). This is why M19 uses
  one *combined eager* pass — hidden states and attention come from the same forward.
- **None of the five `diag` arrays is derivable from the others**, so there is no "store one,
  rebuild the rest" saving available: `attn_topk` and `lap_topk` are sorted *independently*, so
  the positional correspondence needed to form `D = A + P` is gone; `sink_topk` is the top-`k` of
  that `D` gathered by `argsort(D)` and `sink_vnorm_topk` multiplies `‖V‖` at those same ranks
  (and `‖V‖` is stored nowhere else); `attn_logdet` is a mean of `log(A)` over *all* masked
  positions, which `phase2_attention.py` flags as silently-wrong-if-reconstructed. The one array
  everything *is* derivable from is the full-length `[L,H,S]` diagonal set — which is **larger**,
  1,393 KB/row vs 606 KB/row for qwen25vl (2.3×), and worse for Pixtral at S≈1,143.
- **The 3× lever is the segments, not the families.** `all`/`image`/`text` cost ~1.8G each per
  pair and the **8-QA parity target only ever had `all`** (text pairs have no image/text split),
  so `image`/`text` are a VLM-only extension. Dropping to `all`-only would take the store from
  ~204G to **~68G**. Decision (2026-08-26): let extraction finish writing all three, prune later.

  > **Correction (2026-08-27): "M22 confirms routing only reads `all`" was never going to happen
  > on its own — the code defaulted the other way.** Checked directly, not assumed: `RunConfig.for_pair`
  > sets `segments=p.segments`, the pair's *full* segment tuple, and stage 07's CLI fell back to
  > `segments or cfg.segments` when `--segments` wasn't passed — i.e. **all 3 segments by default**.
  > The earlier "M18 e2e validated on real data" claim didn't test this either; that scratch script
  > hardcoded `segments=("all",)` explicitly, sidestepping the default path entirely.
  >
  > **This was not hypothetical — it had already happened.** Stage 04 (`per_pair_long`) was run
  > without `--segments all` for 6 of 9 VLM pairs (`*_okvqa`, `*_vqav2`; the 3 `*_advqa` pairs
  > happened to be run with the flag) and had computed `image`/`text` alongside `all` — 3× the
  > intended compute, silently. Worse: `csx_report/tables/srxauc.py`'s `per_unit_min` groups by
  > `["family", "method", "pair"]` with **no `segment` key**, so feeding it this data as-is would
  > have let `idxmin` pick across segments arbitrarily for those 6 pairs — an uncontrolled,
  > unintended cross-segment comparison baked into the SR-xAUC headline.
  >
  > **Decision, resolved:** Part C is scoped to `--segments all` only, matching the 8-QA parity
  > target exactly. Two fixes landed: `_prepare()` in `srxauc.py` now filters to `segment == "all"`
  > up front whenever a `segment` column is present (a no-op on the QA fixture, which has none),
  > with `test_non_all_segments_are_excluded_from_the_metric` pinning the bug directly; and stages
  > 04 and 07 now default `--segments` to `"all"` (matching what 06 and 08 already did), so future
  > runs don't repeat the over-scope. The already-computed `image`/`text` rows in the stored
  > `per_pair_long` were left in place — sunk CPU cost, now harmless at read time — rather than
  > deleted. `image`/`text` remain a named, intentionally out-of-scope VLM extension (§4), not
  > something Part C claims to test.
  >
  > **The prune is now unblocked at the decision level** — `rm diag/{image,text}.npz` per pair
  > is safe once M19 extraction finishes, since Part C is pinned to `all` and nothing reads the
  > other two. Not yet executed.

**M20 — the 7 NLI metrics** (`cli_extract/28_uq_metrics.py`).

Reuses `scripts/snne_baseline/snne_core.py` (`EntailmentDeberta`) unchanged, sharded per pair,
env `/root/miniconda3/envs/snne` (vLLM is broken in that env — irrelevant here, we generate
nothing). Settings frozen to the QA run: entailment similarity, `variant=only_denom`,
`temp=1.0`, `condition_on_question=True`.

Output merges into `uq_rows.parquet` as 7 new columns, so `csx_probe` reads them with no new
plumbing.

> **Built (2026-08-27).** `csx_extract/uq_metrics.py` reuses `run_baselines.per_question_scores`
> unchanged; the row source is the same `n_generations` list-of-10-strings column
> `sampled_manifest.py` already parses for M19, so this needed no new data-archaeology into the
> upstream `.pkl`/`.jsonl` generation dumps (those turned out not to retain the raw sample text
> anyway -- only aggregate measures and semantic-id integers). Checkpointed the same way as the
> M19 RAM fix, but row-structured (a periodically-flushed partial parquet, not a memmap) since the
> output here is 7 floats/row, not a big array. `csx_probe/uq.py`'s `read_pair` left-joins
> `paths.uq_metrics(pair)` onto `id` when present, NaN otherwise, so M20 landing per-pair rather
> than all-at-once never blocks L2 ingest for the others.
>
> One real bug caught during the smoke test: `per_question_scores`'s `lexical_sim` stays a 0-dim
> torch tensor (the only one of the 7 not explicitly `float()`-cast in the shared/vendored script)
> -- pyarrow rejects it outright on `to_parquet`. Fixed with a defensive `float()` in our own
> wrapper rather than patching the shared script other callers depend on.
>
> **Runs under `/root/miniconda3/envs/snne`, not `semantic_uncertainty`** -- `rouge_score` and
> `evaluate` are only installed there. Verified end to end: real `qwen25vl_advqa` rows, the real
> `microsoft/deberta-v2-xlarge-mnli` model, correct checkpoint/resume (crash-simulated mid-run,
> confirmed it skips already-scored ids and doesn't restart), correct NaN fallback for
> not-yet-scored pairs. **Not yet launched for the full cohort** -- ~12-15 GPU-hours, and both
> GPUs are currently occupied by M19's last 2 pairs. Small model (few GB), so it fits alongside
> either M19 job on VRAM; whether to run it concurrently (competes for SM cycles, may push out
> M19's critical path) or wait for a free GPU is a scheduling decision, not a blocker.

> **The caveat that must be in the paper, not just the code.** DeBERTa sees the question text and
> the answer strings — **never the image**. Two answers to a visual question are judged for
> entailment without the thing they are about. This is not a new compromise we are introducing:
> the incumbent DSE band for these pairs was computed the same way upstream, so the 8 metrics and
> the band they are compared against share the limitation. It still bounds the claim, and §19
> must say so. **Verify before running** that the VLM `cluster_assignment_entropy` did come from
> a text-only entailment clustering; if it did not, the metric comparison is not apples-to-apples
> and the section needs redesigning.

**M21 — `best_split` thresholds per metric per pair.** The SEP binarisation: scan 100 candidate
cuts, minimise within-group sum of squares (1-D 2-means / Jenks). **The generalisation that was
required in QA and will be required again:** the original grid runs `linspace(1e-10, max)`, which
assumes a non-negative score. `lexical_sim` and `snne` are negated quantities, so with a negative
maximum the grid runs *backwards* and puts ~100% of rows in one band. The grid must start at the
observed minimum — identical for any non-negative score. Gate: reproduce each pair's stored
`DSE_threshold` exactly from the `dse` column.

### Phase III — CPU, after the GPU passes land

**M22 — sampled router + the full routed grid.** With the manifest in place, `router.sampled`
stops raising and the routed grid gains `@sampled`, `spec1_z_cm`, `spec1_platt_cm`, and
`spec1_z_hier`. Two aggregations, exactly as published:

| variant | input | width (`hs_wide`) |
|---|---|---|
| `mean_std` | `concat(mean, std)` over the 10 sample vectors | 2d ≈ 14,338 |
| `cloud` | 10 geometry scalars from the samples' Gram matrix | 10 |

That `cloud` stays competitive at **10 dimensions** against `mean_std` at 14k is one of the
QA run's sharper findings — the band is a property of semantic *spread*, and those ten numbers
encode spread. Whether that replicates on VLM is a genuinely open question and a good result
either way.

**M23 — the metric-router grid** (`cli_probe/09_metric_router.py`), all 64 ordered pairs.

- **The frozen partition is what makes the grid legal.** Re-splitting per metric would give each
  metric its own train/test partition, so `natural_A.train` would intersect `natural_B.test` and
  every off-diagonal cell would leak. Instead the DSE train/test id sets are reused **verbatim**
  and only the `category` field is recomputed. Then `natural_A.train == natural_B.train` as row
  sets for all A, B, and every cross-metric cell is leak-free by construction. **Asserted at
  build time across all 64 pairs × 3 arms × 9 VLM pairs: zero overlapping rows.**
- Arms rebuilt per metric: `natural` (same rows, relabelled), `balanced2` (quota per metric),
  `matched2` (strata are *that metric's own* values — raw for naturally-discrete metrics, 40
  global quantile bins for the six with thousands of distinct values).
- Router fit **once per train-metric** and scored against all 8 test-metric label sets, so a grid
  costs 8 fits, not 64.
- Report the **honest denominator first**: raw band agreement and Cohen's κ between every metric
  pair, then `drop(A→B) = AUROC(train A, test B) − AUROC(train B, test B)` so the diagonal
  absorbs arm, prevalence and intrinsic predictability. The QA run had to concede that agreement
  genuinely predicts the drop (Spearman +0.62 to +0.70) and that the claim is *redundancy
  modulates the cost, it does not create the effect*. We concede the same or we report a
  different finding — we do not quietly drop the concession.
- `lexical_sim` is the check that matters: the only metric built without NLI. If VLM transfer
  survives to `lexical_sim` the mechanism is not an entailment artefact.

**M24 — `draft.py` §16–§21 + the three parity documents.** Render our analogues of all three
target docs side by side with the QA originals, so "does it replicate" is answerable by reading
two columns.

---

## 4. What replication would and would not establish

Worth fixing before the numbers arrive, so the reading is not chosen to fit them.

**A replication means:** on a modality with different features, different failure modes and
different answer distributions, pooled `IvC` still hides a confidence shortcut, SR-xAUC still
exposes it, and band-conditional routing still survives it. That turns a QA-specific result into
a claim about probe-based error detection generally, and it is a strong addition to the paper.

**A non-replication is also publishable**, and the plan must not be built to avoid it. The most
interesting negative outcome: VLM confidence bands may be *less* predictive of correctness than
in QA (VQA answer spaces are small and heavily duplicated — see the 35–80% dedup spread), which
would compress the `IHvCL`/`ILvCH` conflict and leave the generalist less to exploit. If the
shortcut is weaker on VLM, that is a finding about *when* the shortcut appears, and it sharpens
the QA result rather than undermining it.

**What neither outcome establishes:** the published §21 conjectures that a VLM has **two**
confidence channels — linguistic and visual-grounding — which would turn the §16 2×2 into a
2×2×2. Our bands are cut from **linguistic** entropy only, so we do not test that. We are the
first data adjacent to it, not on it. The `hs_mean_image` / `hs_mean_text` segments are already
in the store, which makes the segment-conditional version a one-flag follow-up — worth naming as
the next experiment, not worth claiming.

---

## 5. Frozen constants to carry verbatim

| thing | value | source |
|---|---|---|
| `ROUTE_T` | 0.5 | `51`/`54` |
| hier OOF folds | 5-fold `StratifiedKFold(y_hi, shuffle, 42)`, whole pipeline refit per fold | `55` |
| SR-xAUC boot | n=1000 over units, seed 42 | `58` |
| atomic cells | `IHvCH`, `IHvCL`, `ILvCH`, `ILvCL` | `58` |
| `asym` | `IHvCL − ILvCH` (the **report's** convention) | `LEVER_A_REPORT.md` |
| expert transform | refit **per band**, train rows only | `54.fit_expert` |
| sampled `n` | 10 throughout; no n-sweep | `36` |
| router variants | `mean_std`, `cloud` | `25`/`36` |
| metric strata | raw if ≤ `RAW_KEY_MAX_DISTINCT` distinct, else 40 global quantile bins | `35` |
| `best_split` grid | 100 cuts from **observed min** (not 1e-10) to max | `35`, generalised |
| SNNE settings | entailment sim, `only_denom`, `temp=1.0`, `condition_on_question=True` | `snne_core` |
| `C` | per-pair policy as everywhere else | `configs/c_policy.yaml` |
| evaluation cells | always the **true** band, never the router's predicted band | `54.emit`, guide §12.8 |

---

## 6. Timeline

| phase | work | wall clock | gated on |
|---|---|---|---|
| **I** | M13–M18, all CPU + tests + QA fixture parity | ~2–3 days of implementation | nothing |
| **II** | M19 sampled extraction (~51 GPU-h), M20 NLI metrics (~12–15 GPU-h), M21 thresholds | ~32–36 h on 2 GPUs | **GPUs are currently fully occupied by another user** |
| **III** | M22 routed grid, M23 metric grid, M24 reports | under 2 h at `--n-jobs 24` | II |

The compute is not the hard part — Phase III is under two hours. The critical path is Phase II's
GPU window, and the work that de-risks it is all in Phase I, which can start immediately.

---

## 7. Risks

- **GPU contention.** ~65 GPU-hours cannot be scheduled against another user's two-GPU job
  without coordinating. Raise before launching, not after.
- **`min`/`median` commutation** in SR-xAUC. Tested (M17).
- **`hier` collapsing to `spec1_z`** through in-fold router predictions. Tested against the
  router's held-out error rate (M13).
- **Layer statistics drift** between the greedy and sampled passes. Silent if wrong; recomputed
  in-process by the identical definition (M19).
- **Silent VLM processor corruption.** Each quirk yields plausible output rather than an error;
  mitigated by ordering `advqa` first.
- **Text-only NLI on visual answers.** Bounds the §19 claim; must be verified against how the
  incumbent VLM band was computed before M20 runs.
- **Two experts have no common origin**, so *some* rescaling is mechanically required before
  pooling. That is the cost of two heads, not a bonus to the routed arm, and `proba` is carried
  as the control that shows what happens when the rescaling is chosen badly.
- **Disk.** `/` is at 85% and must not be written to; sampled features go to
  `/data/csx_store/sampled/` and are **kept permanently** (**~204 GB measured**, against 321 G
  free — see the correction in §1, the 35–45 G estimate was wrong by 4–5×) — cheap insurance
  against a future idea needing the sampled internals again, since re-extraction would burn the
  same ~65 GPU-hours a second time. `/data` is shared and already 83% full, so if it tightens the
  reduction to reach for first is `rm diag/{image,text}.npz` per pair (~204 G → ~68 G), which
  costs no GPU time and only forfeits the VLM-only segment split that 8-QA parity never used.
  Joblib temp to `/dev/shm`.
