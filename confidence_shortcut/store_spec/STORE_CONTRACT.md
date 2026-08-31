# Store contract v1

The only coupling between **component 1 (`csx_extract`, GPU)** and
**component 2 (`csx_probe`, CPU)**. Neither package imports the other; they agree
on this document and nothing else.

A pair becomes usable by component 2 **the moment its entry validates**. There is
no "finish all extraction, then start experiments" gate — `csx-store verify
--pair X` is runnable by whoever ran the extraction, with no experiment code
involved, and its exit code is the handoff.

---

## Layout

```
<store>/
  uq/
    uq_rows.parquet             L2  every row of every in-scope run CSV
    generations/<pair>.parquet  L2  the 10 sampled answer strings (optional)
    summary.csv                 L2  per-pair cell counts + tau
  raw/<pair>/
    meta.json                   L0  provenance + geometry  (REQUIRED)
    rows.parquet                L0  one row per extracted example (REQUIRED)
    hs/<segment>.npy            L0  pooled hidden-state features
    hs/peaks.json               L0  the peak layers those pooled features used
    diag/<segment>.npz          L0  reduced attention/Laplacian diagonals
  features/<pair>/<family>/<segment>/
    X.npy                       L1  float32, C-contiguous, mmap-able
    ids.json                    L1  row order for X
    meta.json                   L1  kind, pca_dim, top_k, provenance
  manifest.parquet              L1  availability per (pair, family, segment)
  results/                      contract 2 — see results_spec/RESULTS_CONTRACT.md
```

`<segment>` is `all` for text pairs. VLM pairs additionally carry `image` and
`text`: a VLM has two confidence channels (linguistic and visual-grounding) that
need not agree, and separating them at extraction time is the only way to test
that later — the spans are not recoverable from a pooled vector afterwards.

---

## L0 — what extraction writes

### `meta.json` (required)

```json
{
  "schema_version": 1,
  "pair": "qwen25vl_vqav2",
  "model": {"key": "qwen25vl", "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "layers": 28, "n_q_heads": 28, "n_kv_heads": 4, "hidden_dim": 3584},
  "dataset": "vqav2",
  "modality": "vlm",
  "prompt_template": "chat",
  "segments": ["all", "image", "text"],
  "n_rows": 15000,
  "n_pool": 34991,
  "subsample": {"n_target": 15000, "seed": 42, "stratified_on": ["IH","CH","IL","CL"]},
  "phase1": {"done": true, "attn_impl": "sdpa", "dtype": "bfloat16",
             "written": "2026-08-23T14:02:11Z", "extractor_version": "1.0.0"},
  "phase2": {"done": false},
  "top_k": 50,
  "sink_k": 10,
  "notes": []
}
```

`model.layers` is **resolved from the HF config at extraction time**, not read
from `pairs.yaml` — the VLM layer counts were never in the repo's `MODEL_MAP`, so
a hardcoded value there would be a guess. Component 2 reads the resolved number
from here.

`phase1.done` / `phase2.done` are separate because they are separate passes with
very different costs. Phase 1 (sdpa) unlocks the three `hs_*` families; phase 2
(eager attention) unlocks the four spectral families. A pair with
`phase1.done && !phase2.done` is **valid and usable** — component 2 simply reports
the spectral families as unavailable for it.

`notes` records anything that changes the features and would otherwise be
invisible — most importantly an image-resolution cap, if one is ever used to make
Pixtral's long sequences affordable.

### `rows.parquet` (required)

One row per extracted example, in the order used by every `.npy` in the entry.

| column | type | notes |
|---|---|---|
| `id` | str | `train::N` / `validation::N`, joins to L2 |
| `row` | int32 | position in the `.npy` arrays; strictly `0..n-1` |
| `category` | str | `IH` / `CH` / `IL` / `CL`, copied from L2 |
| `entropy` | float64 | copied from L2 |
| `s_ext` | float32 | length-normalised greedy log-prob; the last component of every `hs_*` vector |
| `seq_len` | int32 | full prompt + answer token count |
| `answer_start`, `answer_end` | int32 | the teacher-forced answer span |
| `image_start`, `image_end` | int32 | image-token span; `-1, -1` for text pairs |
| `image_path` | str | VLM only; empty for text pairs |

### `hs/<segment>.npy` + `hs/peaks.json`

Hidden states are **reduced destructively**, following `02_reduce_hidden.py`:
raw `[N, L, D]` is ~10 GB per pair at gemma-3-27b's 62 layers, so extraction
computes peak layers (per-layer AUROC of an L2 LR on **train rows only**), writes
the pooled `concat[mean_z(mid), mean_z(late), s_ext]` matrices, and deletes the
raw shards. **Changing the pooling scheme later means re-extracting.**

`hs/<segment>.npy` is a dict-of-arrays `.npz` keyed by scheme
(`hs_wide`, `hs_narrow`, `hs_peak_only`), each `[N, 2*D+1]` float16.
`peaks.json` records `{"mid": int, "late": int, "buckets": {...},
"layer_auc": {...}}` per segment.

### `diag/<segment>.npz`

Spectral diagonals are **reduced non-destructively**. Full `[L,H,S]` would run to
~60 GB for one vqav2 pair, so reduction happens inside the extraction loop; top-50
retains everything all four spectral families need (`TOP_K_GRID` maxes at 50), so
this side can be re-derived without touching a GPU.

| key | shape | dtype | feeds |
|---|---|---|---|
| `attn_topk` | `[N, L, H, 50]` | float16 | `attn_eigvals` |
| `lap_topk` | `[N, L, H, 50]` | float16 | `lapeigvals` |
| `sink_topk` | `[N, L, H, 50]` | float16 | `sink` (score = `attn_diag + lap_diag`) |
| `sink_vnorm_topk` | `[N, L, H, 50]` | float16 | `sink x ||V||`, gathered at the sink top-k indices |
| `attn_logdet` | `[N, L, H]` | float32 | `attnlogdet` |

**`attn_logdet` is not optional and not derivable from the top-k.** `attnlogdet`
is `mean(log(clamp_min(attn_diag, 1e-12)))` over *all* positions, not a top-k
feature. Storing only top-k would leave that family silently wrong rather than
missing, which is why it is a separate required key.

---

## L1 — what component 2 derives

`features/<pair>/<family>/<segment>/`:

- `X.npy` — float32, C-contiguous, `[n, dim]`, memory-mappable.
- `ids.json` — `[str, ...]`, the row order of `X`.
- `meta.json` — `{"schema_version", "pair", "family", "segment", "kind",
  "pca_dim", "top_k", "dim", "n", "sha256", "source": {...}, "builder_version"}`.

`kind` is `hs` or `spectral`; it selects the LR hyperparameters and the transform.
`pca_dim` is the CV-selected spectral basis (`null` means StandardScaler).

`manifest.parquet` has one row per `(pair, family, segment)` with `status` in
`ready` / `missing_raw` / `stale` / `error`, plus `n`, `dim`, `kind`, `pca_dim`
and the source hash.

**`missing_raw` is a real state, not an error.** Notably, the legacy
`certain_mispredictions_results/.../gemma3_27b_*.pt` and `qwen3_14b_*.pt`
diagonals exist and look usable but are the 1,400-row sampled runs (980 train /
420 test) against 13.5k–50k-row CSVs, with no `hs_*` at all. The manifest must
classify them `missing_raw` rather than quietly build a 1,400-row pair.

### The legacy adapter

The 8 `qa8` pairs already have their internal states, in the older
`full_natural_data/` layout. `csx_probe/store/legacy_adapter.py` presents that
directory as a valid L0 entry, so legacy artifacts and newly extracted ones reach
component 2 through the identical interface. That is what keeps the 1e-4 parity
gate meaningful rather than a special case running down a separate code path.

---

## Validation

`csx-store verify --pair <key>` checks, in order:

1. `meta.json` parses, `schema_version` is supported, required keys present.
2. `rows.parquet` exists; `row` is exactly `0..n-1`; `id` is unique; `n_rows`
   in meta agrees.
3. Every declared segment has the artifacts its completed phases promise.
4. Array leading dimensions all equal `n_rows`.
5. `[L, H]` geometry is consistent across `diag` keys and matches `model.layers`.
6. `attn_logdet` is present whenever `phase2.done` — the one key whose absence
   would be silent rather than loud.
7. Spans are sane: `0 <= answer_start < answer_end <= seq_len`; for VLM pairs the
   image and text spans partition `[0, seq_len)`.
8. Every `id` is present in L2 (if L2 is built), so a pair cannot be extracted
   against rows the study does not know about.

Exit 0 = usable. Non-zero = the reason, named.

---

## Compatibility

`schema_version` is bumped when a required field changes meaning or disappears.
Component 2 refuses an entry whose version it does not know, rather than guessing
— an entry written by a newer extractor is a hard stop, not a best-effort read.
Adding an optional key does not bump the version.
