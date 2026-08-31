# Results contract v1

The only coupling between **component 2 (`csx_probe`)** and
**component 3 (`csx_report`)**. `csx_report` imports neither of the other
packages; it reads these tables.

## The one rule

**Every table is atomic per pair. Nothing here is aggregated, and no column is
named `cohort`.**

One pair's rows are complete and meaningful on their own. Medians, confidence
intervals and pass-rule counts are computed by component 3, over whichever pairs
are asked for at runtime. That is what makes re-grouping free: LLM vs VLM,
per-dataset, per-model-size, leave-one-out — none of it requires refitting
anything.

The published artifacts already split on exactly this seam, which is the evidence
the seam is real:

| published artifact | rows | side |
|---|---:|---|
| `transfer_grid_selected_c/per_pair_long.csv` | 34,944 | atomic |
| `transfer_grid_selected_c/derived_per_pair.csv` | 10,752 | atomic |
| `alpha_rotation_selected_c/rotation_long.csv` | 35,560 | atomic |
| `alpha_rotation_selected_c/verdict.csv` | 112 | atomic (per-pair `passes`) |
| `transfer_grid_selected_c/aggregate_median.csv` | 4,368 | **aggregated** |

`verdict.csv` carrying per-pair `passes` is why the pre-registered "≥6 of 8" rule
needs nothing special: it is a count over whichever pairs are in the group.

---

## Provenance columns, on every table

| column | why |
|---|---|
| `pair` | the grouping key; never a cohort |
| `family` | one of the 7 |
| `segment` | `all` for text; `all`/`image`/`text` for VLM |
| `C` | **the value this row was actually fitted at** |
| `c_mode` | `pinned` / `per_pair` / `group` |
| `prompt_template` | `ranking` / `chat` |

`C` and `prompt_template` are recorded per row because they are the two things
that can legitimately differ between pairs in one group. Component 3 reports the
spread within a group rather than hiding it — a group whose pairs were fit at
different `C`, or that mixes the ranking and chat templates, is a real caveat.

---

## Tables

### `per_pair_long`

One row per `(pair, family, segment, train_arm, test_arm, head, contrast)`.

```
pair model dataset modality family segment C c_mode prompt_template
train_arm test_arm diagonal head contrast
AUROC AUROC_lo AUROC_hi n n_pos
```

- `train_arm` / `test_arm` ∈ `dse_natural`, `dse_balanced2`, `dse_matched`,
  `dse_matched2`; plus `pl_*_dNN` (composition placebo) and `ns_*_dNN`
  (size-only) as train arms in the confound tables.
- `head` ∈ `g` (correctness), `sep` (confidence), `entropy_only` (unfit).
- `contrast` ∈ `IvC`; the six admissible cells `IHvCH ILvCL IHvC ILvC CHvI CLvI`;
  the two definitional cells `IHvCL ILvCH`; and the derived scalars
  `cell_min cell_mean cell_spread shortcut_IHvCL`.
- Orientation is always *larger ⇒ more incorrect*, with **no per-head sign
  flips**: `sep` scoring 0.000 on `IHvCL` is the informative observation, not a
  bug. `CHvI` / `CLvI` are computed as `(pos = I, neg = CH/CL)`, which is
  identically the 1−AUROC convention with one fewer sign flip to get wrong.
- `AUROC` is `NaN` when either side of a contrast has fewer than
  `min_per_class` (10) rows. It is **never** substituted with 0.5.

### `rotation_long`

One row per `(pair, family, segment, kind, alpha, draw)`.

```
pair model dataset family segment C c_mode kind alpha draw n_train
auroc_nat_IvC norm_w
cos_sigma_sep cos_euclid_sep theta_sigma_sep theta_euclid_sep
cos_sigma_entropy cos_euclid_entropy theta_sigma_entropy theta_euclid_entropy
```

`kind` ∈ `alpha` (a ladder rung, `draw = -1`), `boot` (bootstrap refit of that
rung), `placebo` (a size-matched draw), `refboot` (bootstrap refit of the frozen
reference axis, for the stability check).

Cosines are **signed** — never `abs` — and `theta = degrees(arccos(clip(c,-1,1)))`.

### `verdict`

One row per `(pair, family, metric)`, `metric` ∈ `Sigma`, `Euclid`.

```
pair family metric C c_mode
theta_a0 theta_a1 delta delta_lo delta_hi
null_med null_p95 passes
theta_a000 theta_a025 theta_a050 theta_a075 theta_a100
```

`passes` is the per-pair criterion `delta > null_p95`. The null is **all 400
outer differences** `placebo(alpha=1)_i - placebo(alpha=0)_j` (20 x 20), and the
bar is that distribution's 95th percentile. The "≥6 of 8" rollup is component 3's.

### `c_selection`

One row per `(space, family, arm, pair)` — the full CV curve, not just the argmax.

```
space family train_arm pair model dataset kind pca_dim n_train n_feat
best_C best_cv_auroc
cv_auroc_C=<v> ... conv_C=<v> ...        one pair of columns per grid value
```

Storing the whole curve is what makes the `C` policy re-resolvable **with no CV
refits**: switching a pair between `pinned` and `per_pair` reads these columns
again and only refits the grids whose `C` actually moved.

### `arm_stats`

One row per `(pair, arm, split)`: `n`, the four cell counts, `pct_incorrect`,
`tau`. Component 3 renders it; component 2 computes it.

---

## Storage

Parquet under `<store>/results/<table>.parquet`, with per-pair checkpoints at
`<store>/results/units/<table>/<pair>.parquet` so a killed run resumes instead of
restarting. The top-level table is a concatenation of the checkpoints — **read
from disk, not from the in-memory results of the current invocation**, so a
partial rerun cannot silently drop pairs it did not touch this time.

## Compatibility

`schema_version` lives in `<store>/results/_meta.json`. Component 3 refuses a
version it does not know rather than guessing. Adding a column is not a version
bump; changing what one means is.
