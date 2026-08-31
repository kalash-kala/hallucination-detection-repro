# The alpha-rotation experiment on `vlm` -- does matching move the correctness axis?

*9 pairs, 7 families, both inner products. Generated from the atomic `rotation_long` and `verdict` tables; every number here is a median or a per-pair value, never a re-fit.*

<a id='s1'></a>
## 1. How this works

**The question.** A probe fit on natural rows might be tracking correctness, or it might be tracking confidence, which correlates with correctness. If it is the latter, then removing the confidence signal from the training set should force the probe's weight vector to *rotate* -- it has to find a different direction to do the job.

**The knob: alpha.** `alpha` interpolates the training set from natural (`alpha=0`) to fully entropy-matched (`alpha=1`), where entropy carries no information about correctness by construction. The ladder is nested: each rung is a subset of the one below it, so the only thing changing is the confidence-correctness coupling.

**The measurement.** Fit the same probe at each rung, then measure the angle `theta` between the `alpha=0` weight vector and each later one. A large `theta(1)` means matching moved the axis.

| metric | definition | reads as |
|---|---|---|
| **Sigma-metric** | `cos = u'Sv / sqrt(u'Su * v'Sv)`, `S` = covariance of natural-**test** features | the correlation between the two probes' *scores* -- the behavioural angle |
| **Euclidean** | `cos = u.v / (norm(u) * norm(v))` | the literal angle between weight vectors, every coordinate weighted equally |

**Why the placebo decides everything.** Shrinking a training set rotates a probe on its own, through variance alone. The placebo draws subsets of the *same size* as each alpha rung but sampled at random, so it measures rotation-from-shrinkage with the confidence structure left intact. A family only counts if its real `delta` exceeds the placebo's 95th percentile.

**The bar.** `delta > null_p95` per pair, then `ceil(0.75 * n_pairs)` pairs must clear it: **7 of 9** here. (The published run's `6/8` is this same rule.)

If `null_p95` is the part you need to explain to someone, [Appendix A](#sA) does exactly that from first principles.

<a id='s2'></a>
## 2. The populations

Proof that alpha does what it claims: the ladder shrinks the training set monotonically, and discrimination on the *natural* test set decays only mildly -- so the probe is still working, just from a different direction.

| alpha | train rows (median/pair) | total train rows | AUROC natural test (IvC), median |
|---|---|---|---|
| 0.00 | 9,839 | 67,317 | 0.777 |
| 0.25 | 8,758 | 60,503 | 0.769 |
| 0.50 | 7,681 | 53,625 | 0.763 |
| 0.75 | 6,514 | 46,746 | 0.755 |
| 1.00 | 5,308 | 39,932 | 0.738 |

<a id='s3'></a>
## 3. The rotation

Median `theta` in degrees across the 9 pairs. `change` is `theta(1) - theta(0)`.

### `hs_wide`

| metric | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| Sigma-metric | 40.9 | 42.9 | 46.2 | 54.1 | 61.4 | **+20.5** |
| Euclidean | 67.9 | 69.2 | 71.2 | 75.7 | 81.9 | **+14.0** |
| *AUROC on natural test (IvC)* | 0.830 | 0.829 | 0.826 | 0.817 | 0.777 | *-0.053* |

### `lapeigvals`

| metric | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| Sigma-metric | 42.3 | 46.3 | 51.6 | 57.8 | 69.3 | **+27.0** |
| Euclidean | 43.0 | 45.6 | 48.9 | 54.3 | 67.9 | **+24.8** |
| *AUROC on natural test (IvC)* | 0.753 | 0.751 | 0.744 | 0.740 | 0.700 | *-0.054* |

### `hs_narrow`

| metric | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| Sigma-metric | 41.3 | 43.7 | 46.0 | 53.7 | 67.3 | **+26.0** |
| Euclidean | 64.6 | 67.0 | 70.3 | 75.1 | 82.2 | **+17.6** |
| *AUROC on natural test (IvC)* | 0.831 | 0.830 | 0.827 | 0.818 | 0.775 | *-0.056* |

### `hs_peak_only`

| metric | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| Sigma-metric | 41.5 | 44.0 | 46.2 | 54.1 | 67.1 | **+25.6** |
| Euclidean | 64.3 | 66.6 | 69.9 | 74.7 | 82.3 | **+18.0** |
| *AUROC on natural test (IvC)* | 0.831 | 0.830 | 0.827 | 0.819 | 0.774 | *-0.057* |

### `attn_eigvals`

| metric | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| Sigma-metric | 49.0 | 52.2 | 56.1 | 63.9 | 72.4 | **+23.5** |
| Euclidean | 61.7 | 64.3 | 68.5 | 74.2 | 81.8 | **+20.1** |
| *AUROC on natural test (IvC)* | 0.687 | 0.684 | 0.677 | 0.667 | 0.622 | *-0.066* |

### `attnlogdet`

| metric | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| Sigma-metric | 56.1 | 58.9 | 60.6 | 65.9 | 79.4 | **+23.3** |
| Euclidean | 64.6 | 68.3 | 72.2 | 76.9 | 84.2 | **+19.6** |
| *AUROC on natural test (IvC)* | 0.587 | 0.582 | 0.583 | 0.575 | 0.568 | *-0.019* |

### `sink`

| metric | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| Sigma-metric | 41.2 | 44.0 | 49.1 | 55.9 | 61.3 | **+20.2** |
| Euclidean | 42.8 | 45.8 | 49.3 | 54.1 | 63.7 | **+20.9** |
| *AUROC on natural test (IvC)* | 0.785 | 0.782 | 0.777 | 0.763 | 0.742 | *-0.043* |

<a id='s3b'></a>
## 3b. Per-pair angle ladders

The medians above hide the spread, which is where the interesting disagreements live. Every pair's own ladder, in degrees.

### `hs_wide` -- Sigma-metric

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 39.4 | 41.7 | 44.8 | 49.1 | 54.3 | **14.9** |
| gemma3_12b_okvqa | 48.3 | 50.3 | 52.7 | 56.1 | 61.0 | **12.7** |
| gemma3_12b_vqav2 | 34.2 | 35.7 | 37.7 | 40.7 | 45.8 | **11.5** |
| pixtral12b_advqa | 40.5 | 42.9 | 48.2 | 56.0 | 68.1 | **27.6** |
| pixtral12b_okvqa | 40.9 | 42.9 | 46.2 | 51.1 | 61.4 | **20.5** |
| pixtral12b_vqav2 | 44.1 | 44.9 | 45.7 | 46.8 | 48.0 | **3.8** |
| qwen25vl_advqa | 49.5 | 53.5 | 59.1 | 66.6 | 77.6 | **28.1** |
| qwen25vl_okvqa | 44.5 | 47.0 | 50.7 | 57.6 | 73.2 | **28.7** |
| qwen25vl_vqav2 | 39.5 | 42.5 | 46.2 | 54.1 | 73.6 | **34.1** |

### `hs_wide` -- Euclidean

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 79.5 | 81.5 | 84.2 | 87.7 | 90.8 | **11.4** |
| gemma3_12b_okvqa | 72.8 | 74.3 | 76.0 | 78.5 | 81.5 | **8.6** |
| gemma3_12b_vqav2 | 69.1 | 71.3 | 73.9 | 77.1 | 82.0 | **12.9** |
| pixtral12b_advqa | 59.1 | 62.4 | 67.0 | 73.5 | 81.9 | **22.7** |
| pixtral12b_okvqa | 65.4 | 67.9 | 71.0 | 75.7 | 83.4 | **18.0** |
| pixtral12b_vqav2 | 74.6 | 76.1 | 77.9 | 79.9 | 82.6 | **8.0** |
| qwen25vl_advqa | 58.1 | 61.0 | 65.4 | 70.8 | 78.3 | **20.2** |
| qwen25vl_okvqa | 67.9 | 69.2 | 71.2 | 74.7 | 81.9 | **14.0** |
| qwen25vl_vqav2 | 59.9 | 61.7 | 64.1 | 68.3 | 77.1 | **17.1** |

### `lapeigvals` -- Sigma-metric

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 51.6 | 55.8 | 60.7 | 68.3 | 75.1 | **23.4** |
| gemma3_12b_okvqa | 45.7 | 48.4 | 52.8 | 57.6 | 65.1 | **19.4** |
| gemma3_12b_vqav2 | 32.2 | 34.3 | 37.2 | 40.8 | 47.6 | **15.4** |
| pixtral12b_advqa | 41.9 | 44.8 | 50.0 | 57.8 | 69.3 | **27.3** |
| pixtral12b_okvqa | 37.1 | 39.3 | 43.3 | 48.8 | 60.1 | **23.0** |
| pixtral12b_vqav2 | 40.1 | 40.8 | 41.5 | 42.5 | 43.8 | **3.7** |
| qwen25vl_advqa | 50.1 | 55.0 | 60.6 | 69.6 | 82.1 | **32.0** |
| qwen25vl_okvqa | 45.2 | 48.8 | 54.4 | 64.3 | 86.2 | **41.1** |
| qwen25vl_vqav2 | 42.3 | 46.3 | 51.6 | 62.1 | 87.0 | **44.7** |

### `lapeigvals` -- Euclidean

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 46.4 | 51.6 | 55.3 | 60.0 | 67.9 | **21.5** |
| gemma3_12b_okvqa | 45.2 | 46.7 | 50.1 | 53.4 | 59.0 | **13.8** |
| gemma3_12b_vqav2 | 33.1 | 34.8 | 37.1 | 39.5 | 45.2 | **12.2** |
| pixtral12b_advqa | 40.0 | 43.5 | 48.9 | 56.4 | 68.2 | **28.2** |
| pixtral12b_okvqa | 35.8 | 37.9 | 41.5 | 46.4 | 56.0 | **20.2** |
| pixtral12b_vqav2 | 39.6 | 40.2 | 40.7 | 41.4 | 42.7 | **3.1** |
| qwen25vl_advqa | 51.5 | 55.0 | 59.1 | 68.1 | 79.0 | **27.5** |
| qwen25vl_okvqa | 57.5 | 59.7 | 62.5 | 67.3 | 77.3 | **19.9** |
| qwen25vl_vqav2 | 43.0 | 45.6 | 48.9 | 54.3 | 68.1 | **25.0** |

### `hs_narrow` -- Sigma-metric

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 39.8 | 42.3 | 45.3 | 49.8 | 54.9 | **15.0** |
| gemma3_12b_okvqa | 56.0 | 57.7 | 60.1 | 63.2 | 67.3 | **11.3** |
| gemma3_12b_vqav2 | 33.7 | 35.2 | 37.4 | 40.6 | 46.1 | **12.4** |
| pixtral12b_advqa | 41.3 | 43.7 | 48.9 | 56.7 | 68.9 | **27.7** |
| pixtral12b_okvqa | 40.3 | 42.4 | 45.8 | 51.0 | 61.4 | **21.2** |
| pixtral12b_vqav2 | 44.3 | 45.0 | 45.8 | 47.0 | 48.3 | **4.0** |
| qwen25vl_advqa | 49.5 | 53.5 | 59.1 | 66.7 | 77.7 | **28.2** |
| qwen25vl_okvqa | 44.2 | 46.6 | 50.3 | 56.9 | 71.8 | **27.6** |
| qwen25vl_vqav2 | 39.4 | 42.4 | 46.0 | 53.7 | 72.6 | **33.2** |

### `hs_narrow` -- Euclidean

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 78.6 | 80.7 | 83.4 | 86.7 | 89.9 | **11.4** |
| gemma3_12b_okvqa | 83.9 | 85.4 | 86.9 | 88.9 | 91.4 | **7.5** |
| gemma3_12b_vqav2 | 64.6 | 66.9 | 69.6 | 73.0 | 78.0 | **13.4** |
| pixtral12b_advqa | 58.7 | 62.0 | 66.5 | 73.1 | 81.4 | **22.8** |
| pixtral12b_okvqa | 64.5 | 67.0 | 70.3 | 75.1 | 83.2 | **18.7** |
| pixtral12b_vqav2 | 74.4 | 75.9 | 77.7 | 79.7 | 82.4 | **8.1** |
| qwen25vl_advqa | 58.7 | 61.8 | 66.2 | 71.6 | 78.8 | **20.0** |
| qwen25vl_okvqa | 68.7 | 70.0 | 71.9 | 75.3 | 82.2 | **13.5** |
| qwen25vl_vqav2 | 60.0 | 61.8 | 64.1 | 68.2 | 76.8 | **16.8** |

### `hs_peak_only` -- Sigma-metric

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 40.0 | 42.5 | 45.5 | 50.1 | 55.0 | **15.0** |
| gemma3_12b_okvqa | 55.6 | 57.3 | 59.8 | 62.9 | 67.1 | **11.5** |
| gemma3_12b_vqav2 | 33.4 | 34.9 | 37.1 | 40.3 | 45.7 | **12.4** |
| pixtral12b_advqa | 41.5 | 44.0 | 49.2 | 57.1 | 69.2 | **27.7** |
| pixtral12b_okvqa | 40.2 | 42.4 | 45.7 | 50.9 | 61.4 | **21.2** |
| pixtral12b_vqav2 | 44.2 | 44.9 | 45.7 | 46.8 | 48.2 | **4.0** |
| qwen25vl_advqa | 49.5 | 53.5 | 59.2 | 66.9 | 77.9 | **28.4** |
| qwen25vl_okvqa | 44.1 | 46.4 | 50.1 | 56.5 | 71.2 | **27.1** |
| qwen25vl_vqav2 | 39.6 | 42.6 | 46.2 | 54.1 | 73.3 | **33.7** |

### `hs_peak_only` -- Euclidean

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 78.9 | 81.1 | 83.8 | 87.1 | 90.4 | **11.5** |
| gemma3_12b_okvqa | 84.2 | 85.7 | 87.3 | 89.3 | 91.8 | **7.5** |
| gemma3_12b_vqav2 | 64.3 | 66.5 | 69.3 | 72.7 | 77.8 | **13.6** |
| pixtral12b_advqa | 58.6 | 61.9 | 66.5 | 73.0 | 81.4 | **22.8** |
| pixtral12b_okvqa | 64.1 | 66.6 | 69.9 | 74.7 | 82.9 | **18.8** |
| pixtral12b_vqav2 | 74.2 | 75.7 | 77.5 | 79.6 | 82.3 | **8.1** |
| qwen25vl_advqa | 59.1 | 62.1 | 66.6 | 72.0 | 79.4 | **20.3** |
| qwen25vl_okvqa | 69.4 | 70.7 | 72.6 | 76.0 | 82.8 | **13.4** |
| qwen25vl_vqav2 | 60.0 | 61.8 | 64.2 | 68.3 | 76.7 | **16.8** |

### `attn_eigvals` -- Sigma-metric

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 56.6 | 58.6 | 61.2 | 67.0 | 70.8 | **14.2** |
| gemma3_12b_okvqa | 49.0 | 52.2 | 56.1 | 60.4 | 67.7 | **18.7** |
| gemma3_12b_vqav2 | 43.0 | 45.0 | 48.0 | 52.2 | 58.7 | **15.7** |
| pixtral12b_advqa | 45.9 | 48.3 | 55.3 | 62.5 | 72.4 | **26.6** |
| pixtral12b_okvqa | 49.9 | 53.4 | 58.7 | 65.9 | 77.0 | **27.1** |
| pixtral12b_vqav2 | 57.5 | 59.3 | 61.5 | 63.9 | 67.0 | **9.5** |
| qwen25vl_advqa | 63.5 | 69.0 | 75.9 | 86.9 | 95.3 | **31.9** |
| qwen25vl_okvqa | 41.5 | 45.7 | 51.7 | 63.0 | 84.3 | **42.8** |
| qwen25vl_vqav2 | 41.5 | 46.8 | 54.2 | 68.9 | 97.5 | **56.0** |

### `attn_eigvals` -- Euclidean

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 55.3 | 58.8 | 63.4 | 70.1 | 74.8 | **19.5** |
| gemma3_12b_okvqa | 48.2 | 51.2 | 55.1 | 59.2 | 65.5 | **17.3** |
| gemma3_12b_vqav2 | 69.7 | 71.6 | 74.0 | 77.5 | 81.9 | **12.2** |
| pixtral12b_advqa | 50.5 | 55.9 | 63.7 | 72.6 | 80.9 | **30.3** |
| pixtral12b_okvqa | 61.1 | 64.3 | 68.5 | 74.2 | 82.4 | **21.3** |
| pixtral12b_vqav2 | 70.8 | 72.7 | 75.1 | 78.3 | 81.8 | **11.0** |
| qwen25vl_advqa | 61.8 | 66.9 | 71.5 | 81.7 | 89.1 | **27.3** |
| qwen25vl_okvqa | 61.7 | 63.4 | 66.7 | 70.9 | 78.5 | **16.8** |
| qwen25vl_vqav2 | 63.9 | 66.8 | 70.6 | 75.9 | 86.8 | **22.9** |

### `attnlogdet` -- Sigma-metric

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 61.1 | 61.3 | 60.3 | 65.9 | 70.7 | **9.6** |
| gemma3_12b_okvqa | 60.3 | 63.7 | 67.7 | 72.7 | 79.4 | **19.1** |
| gemma3_12b_vqav2 | 46.6 | 49.4 | 54.3 | 60.4 | 69.8 | **23.2** |
| pixtral12b_advqa | 47.7 | 51.6 | 56.8 | 62.6 | 72.7 | **24.9** |
| pixtral12b_okvqa | 64.0 | 67.4 | 72.6 | 77.6 | 86.6 | **22.5** |
| pixtral12b_vqav2 | 56.1 | 58.9 | 61.1 | 63.8 | 67.6 | **11.5** |
| qwen25vl_advqa | 64.1 | 69.2 | 74.7 | 83.6 | 91.1 | **26.9** |
| qwen25vl_okvqa | 42.2 | 46.0 | 50.9 | 61.6 | 82.8 | **40.6** |
| qwen25vl_vqav2 | 46.7 | 52.8 | 60.6 | 72.5 | 95.5 | **48.8** |

### `attnlogdet` -- Euclidean

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 63.6 | 68.3 | 69.5 | 76.5 | 81.8 | **18.2** |
| gemma3_12b_okvqa | 68.4 | 70.8 | 73.5 | 77.0 | 82.0 | **13.6** |
| gemma3_12b_vqav2 | 54.9 | 57.5 | 61.8 | 66.9 | 73.2 | **18.3** |
| pixtral12b_advqa | 61.7 | 65.3 | 72.0 | 78.4 | 86.6 | **24.9** |
| pixtral12b_okvqa | 63.1 | 67.0 | 71.1 | 76.1 | 83.8 | **20.7** |
| pixtral12b_vqav2 | 71.5 | 74.6 | 77.6 | 81.4 | 85.0 | **13.5** |
| qwen25vl_advqa | 64.7 | 70.5 | 75.0 | 85.3 | 93.6 | **28.9** |
| qwen25vl_okvqa | 66.8 | 68.9 | 72.2 | 76.0 | 84.2 | **17.5** |
| qwen25vl_vqav2 | 64.6 | 67.8 | 72.3 | 76.9 | 87.4 | **22.8** |

### `sink` -- Sigma-metric

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 43.0 | 45.8 | 49.9 | 55.9 | 61.3 | **18.4** |
| gemma3_12b_okvqa | 44.7 | 47.2 | 50.3 | 54.4 | 60.5 | **15.8** |
| gemma3_12b_vqav2 | 33.5 | 35.4 | 37.6 | 40.9 | 46.5 | **13.0** |
| pixtral12b_advqa | 41.2 | 44.0 | 49.6 | 57.4 | 69.2 | **28.1** |
| pixtral12b_okvqa | 37.3 | 39.8 | 43.4 | 48.7 | 59.7 | **22.5** |
| pixtral12b_vqav2 | 41.9 | 42.7 | 43.5 | 44.6 | 45.8 | **3.9** |
| qwen25vl_advqa | 50.1 | 54.2 | 60.1 | 68.3 | 78.9 | **28.8** |
| qwen25vl_okvqa | 40.7 | 44.0 | 49.1 | 58.0 | 79.0 | **38.3** |
| qwen25vl_vqav2 | 40.4 | 44.1 | 48.8 | 58.2 | 80.6 | **40.2** |

### `sink` -- Euclidean

| pair | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 | change |
|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 42.8 | 45.8 | 49.0 | 53.8 | 58.1 | **15.4** |
| gemma3_12b_okvqa | 52.0 | 53.9 | 56.0 | 59.1 | 63.7 | **11.7** |
| gemma3_12b_vqav2 | 40.4 | 42.2 | 43.5 | 46.3 | 51.0 | **10.6** |
| pixtral12b_advqa | 41.2 | 43.9 | 49.3 | 58.0 | 70.2 | **29.1** |
| pixtral12b_okvqa | 36.5 | 39.0 | 42.6 | 47.6 | 57.2 | **20.7** |
| pixtral12b_vqav2 | 47.7 | 48.8 | 49.9 | 51.4 | 53.6 | **5.9** |
| qwen25vl_advqa | 44.0 | 47.4 | 51.4 | 59.6 | 67.7 | **23.7** |
| qwen25vl_okvqa | 54.6 | 56.9 | 59.9 | 64.6 | 75.9 | **21.3** |
| qwen25vl_vqav2 | 42.8 | 45.2 | 48.5 | 54.1 | 68.3 | **25.5** |

<a id='s4'></a>
## 4. The placebo null and the verdict

`margin = delta - null_p95`. A positive margin clears the null. Note how many margins are within a degree or two of zero -- these verdicts are decided on fine differences, not comfortable ones.

### `hs_wide` -- Sigma-metric

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 39.4 | 54.3 | 14.9 | [9.4, 19.8] | 12.3 | +2.59 | **yes** |
| gemma3_12b_okvqa | 48.3 | 61.0 | 12.7 | [9.9, 15.0] | 9.1 | +3.63 | **yes** |
| gemma3_12b_vqav2 | 34.2 | 45.8 | 11.5 | [9.4, 13.7] | 9.6 | +1.96 | **yes** |
| pixtral12b_advqa | 40.5 | 68.1 | 27.6 | [18.9, 35.3] | 21.6 | +5.91 | **yes** |
| pixtral12b_okvqa | 40.9 | 61.4 | 20.5 | [15.9, 22.9] | 14.3 | +6.29 | **yes** |
| pixtral12b_vqav2 | 44.1 | 48.0 | 3.8 | [1.8, 6.2] | 5.2 | -1.40 | no |
| qwen25vl_advqa | 49.5 | 77.6 | 28.1 | [19.8, 37.2] | 24.7 | +3.39 | **yes** |
| qwen25vl_okvqa | 44.5 | 73.2 | 28.7 | [25.2, 32.1] | 20.6 | +8.09 | **yes** |
| qwen25vl_vqav2 | 39.5 | 73.6 | 34.1 | [30.3, 38.6] | 25.0 | +9.09 | **yes** |

**8/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `hs_wide` -- Euclidean

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 79.5 | 90.8 | 11.4 | [6.3, 12.5] | 13.0 | -1.63 | no |
| gemma3_12b_okvqa | 72.8 | 81.5 | 8.6 | [5.6, 9.0] | 8.4 | +0.19 | **yes** |
| gemma3_12b_vqav2 | 69.1 | 82.0 | 12.9 | [9.4, 13.1] | 12.1 | +0.85 | **yes** |
| pixtral12b_advqa | 59.1 | 81.9 | 22.7 | [15.6, 22.7] | 18.6 | +4.18 | **yes** |
| pixtral12b_okvqa | 65.4 | 83.4 | 18.0 | [13.6, 16.3] | 16.7 | +1.32 | **yes** |
| pixtral12b_vqav2 | 74.6 | 82.6 | 8.0 | [5.0, 8.3] | 9.5 | -1.56 | no |
| qwen25vl_advqa | 58.1 | 78.3 | 20.2 | [13.8, 20.8] | 16.3 | +3.89 | **yes** |
| qwen25vl_okvqa | 67.9 | 81.9 | 14.0 | [10.4, 14.0] | 13.2 | +0.84 | **yes** |
| qwen25vl_vqav2 | 59.9 | 77.1 | 17.1 | [12.7, 17.6] | 15.0 | +2.12 | **yes** |

**7/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `lapeigvals` -- Sigma-metric

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 51.6 | 75.1 | 23.4 | [15.1, 29.8] | 20.1 | +3.32 | **yes** |
| gemma3_12b_okvqa | 45.7 | 65.1 | 19.4 | [15.0, 22.8] | 13.7 | +5.77 | **yes** |
| gemma3_12b_vqav2 | 32.2 | 47.6 | 15.4 | [12.8, 19.0] | 12.6 | +2.77 | **yes** |
| pixtral12b_advqa | 41.9 | 69.3 | 27.3 | [18.3, 36.6] | 22.3 | +5.06 | **yes** |
| pixtral12b_okvqa | 37.1 | 60.1 | 23.0 | [18.5, 27.5] | 17.1 | +5.90 | **yes** |
| pixtral12b_vqav2 | 40.1 | 43.8 | 3.7 | [0.7, 6.9] | 5.7 | -2.03 | no |
| qwen25vl_advqa | 50.1 | 82.1 | 32.0 | [19.7, 38.3] | 24.7 | +7.32 | **yes** |
| qwen25vl_okvqa | 45.2 | 86.2 | 41.1 | [34.7, 44.5] | 29.8 | +11.25 | **yes** |
| qwen25vl_vqav2 | 42.3 | 87.0 | 44.7 | [38.9, 48.1] | 33.1 | +11.65 | **yes** |

**8/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `lapeigvals` -- Euclidean

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 46.4 | 67.9 | 21.5 | [12.1, 27.0] | 18.4 | +3.17 | **yes** |
| gemma3_12b_okvqa | 45.2 | 59.0 | 13.8 | [9.3, 17.1] | 10.1 | +3.68 | **yes** |
| gemma3_12b_vqav2 | 33.1 | 45.2 | 12.2 | [8.5, 16.6] | 10.6 | +1.55 | **yes** |
| pixtral12b_advqa | 40.0 | 68.2 | 28.2 | [19.3, 37.3] | 22.1 | +6.09 | **yes** |
| pixtral12b_okvqa | 35.8 | 56.0 | 20.2 | [16.8, 25.0] | 15.9 | +4.26 | **yes** |
| pixtral12b_vqav2 | 39.6 | 42.7 | 3.1 | [0.5, 7.5] | 5.8 | -2.68 | no |
| qwen25vl_advqa | 51.5 | 79.0 | 27.5 | [15.8, 30.4] | 18.9 | +8.59 | **yes** |
| qwen25vl_okvqa | 57.5 | 77.3 | 19.9 | [13.8, 23.9] | 17.8 | +2.03 | **yes** |
| qwen25vl_vqav2 | 43.0 | 68.1 | 25.0 | [19.7, 29.8] | 19.8 | +5.25 | **yes** |

**8/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `hs_narrow` -- Sigma-metric

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 39.8 | 54.9 | 15.0 | [9.6, 19.9] | 12.4 | +2.63 | **yes** |
| gemma3_12b_okvqa | 56.0 | 67.3 | 11.3 | [8.5, 13.1] | 9.0 | +2.32 | **yes** |
| gemma3_12b_vqav2 | 33.7 | 46.1 | 12.4 | [9.9, 14.8] | 10.1 | +2.27 | **yes** |
| pixtral12b_advqa | 41.3 | 68.9 | 27.7 | [18.6, 35.4] | 22.4 | +5.21 | **yes** |
| pixtral12b_okvqa | 40.3 | 61.4 | 21.2 | [16.7, 23.6] | 14.7 | +6.44 | **yes** |
| pixtral12b_vqav2 | 44.3 | 48.3 | 4.0 | [2.2, 6.3] | 5.4 | -1.45 | no |
| qwen25vl_advqa | 49.5 | 77.7 | 28.2 | [19.6, 38.0] | 24.8 | +3.43 | **yes** |
| qwen25vl_okvqa | 44.2 | 71.8 | 27.6 | [24.2, 30.9] | 19.7 | +7.93 | **yes** |
| qwen25vl_vqav2 | 39.4 | 72.6 | 33.2 | [29.2, 37.5] | 24.4 | +8.73 | **yes** |

**8/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `hs_narrow` -- Euclidean

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 78.6 | 89.9 | 11.4 | [6.1, 12.7] | 13.0 | -1.58 | no |
| gemma3_12b_okvqa | 83.9 | 91.4 | 7.5 | [4.3, 7.9] | 8.9 | -1.44 | no |
| gemma3_12b_vqav2 | 64.6 | 78.0 | 13.4 | [9.8, 14.0] | 12.1 | +1.36 | **yes** |
| pixtral12b_advqa | 58.7 | 81.4 | 22.8 | [15.6, 22.6] | 18.7 | +4.06 | **yes** |
| pixtral12b_okvqa | 64.5 | 83.2 | 18.7 | [14.1, 17.2] | 16.9 | +1.77 | **yes** |
| pixtral12b_vqav2 | 74.4 | 82.4 | 8.1 | [4.9, 8.3] | 9.5 | -1.44 | no |
| qwen25vl_advqa | 58.7 | 78.8 | 20.0 | [13.7, 20.7] | 16.5 | +3.54 | **yes** |
| qwen25vl_okvqa | 68.7 | 82.2 | 13.5 | [9.7, 13.5] | 12.7 | +0.75 | **yes** |
| qwen25vl_vqav2 | 60.0 | 76.8 | 16.8 | [12.2, 17.1] | 14.9 | +1.95 | **yes** |

**6/9 clear the placebo null** (bar for this cohort: 7) -- **fails the bar**

### `hs_peak_only` -- Sigma-metric

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 40.0 | 55.0 | 15.0 | [9.9, 19.5] | 12.1 | +2.91 | **yes** |
| gemma3_12b_okvqa | 55.6 | 67.1 | 11.5 | [8.8, 13.1] | 8.9 | +2.54 | **yes** |
| gemma3_12b_vqav2 | 33.4 | 45.7 | 12.4 | [9.9, 14.8] | 10.1 | +2.26 | **yes** |
| pixtral12b_advqa | 41.5 | 69.2 | 27.7 | [18.5, 35.7] | 22.7 | +4.97 | **yes** |
| pixtral12b_okvqa | 40.2 | 61.4 | 21.2 | [16.7, 23.7] | 14.8 | +6.38 | **yes** |
| pixtral12b_vqav2 | 44.2 | 48.2 | 4.0 | [2.1, 6.1] | 5.4 | -1.40 | no |
| qwen25vl_advqa | 49.5 | 77.9 | 28.4 | [19.9, 38.2] | 24.9 | +3.51 | **yes** |
| qwen25vl_okvqa | 44.1 | 71.2 | 27.1 | [23.8, 30.4] | 19.3 | +7.81 | **yes** |
| qwen25vl_vqav2 | 39.6 | 73.3 | 33.7 | [29.6, 38.1] | 24.8 | +8.96 | **yes** |

**8/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `hs_peak_only` -- Euclidean

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 78.9 | 90.4 | 11.5 | [6.4, 12.7] | 13.0 | -1.48 | no |
| gemma3_12b_okvqa | 84.2 | 91.8 | 7.5 | [4.3, 7.9] | 8.9 | -1.37 | no |
| gemma3_12b_vqav2 | 64.3 | 77.8 | 13.6 | [10.1, 14.2] | 12.1 | +1.48 | **yes** |
| pixtral12b_advqa | 58.6 | 81.4 | 22.8 | [15.7, 22.5] | 18.6 | +4.14 | **yes** |
| pixtral12b_okvqa | 64.1 | 82.9 | 18.8 | [14.2, 17.4] | 17.3 | +1.58 | **yes** |
| pixtral12b_vqav2 | 74.2 | 82.3 | 8.1 | [4.9, 8.4] | 9.5 | -1.38 | no |
| qwen25vl_advqa | 59.1 | 79.4 | 20.3 | [13.7, 20.9] | 16.7 | +3.61 | **yes** |
| qwen25vl_okvqa | 69.4 | 82.8 | 13.4 | [9.6, 13.5] | 12.7 | +0.75 | **yes** |
| qwen25vl_vqav2 | 60.0 | 76.7 | 16.8 | [12.1, 17.2] | 15.0 | +1.80 | **yes** |

**6/9 clear the placebo null** (bar for this cohort: 7) -- **fails the bar**

### `attn_eigvals` -- Sigma-metric

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 56.6 | 70.8 | 14.2 | [5.9, 22.5] | 17.7 | -3.48 | no |
| gemma3_12b_okvqa | 49.0 | 67.7 | 18.7 | [13.5, 22.5] | 14.0 | +4.70 | **yes** |
| gemma3_12b_vqav2 | 43.0 | 58.7 | 15.7 | [13.2, 18.4] | 13.6 | +2.15 | **yes** |
| pixtral12b_advqa | 45.9 | 72.4 | 26.6 | [14.8, 37.9] | 26.1 | +0.46 | **yes** |
| pixtral12b_okvqa | 49.9 | 77.0 | 27.1 | [20.8, 27.4] | 22.5 | +4.54 | **yes** |
| pixtral12b_vqav2 | 57.5 | 67.0 | 9.5 | [6.2, 11.1] | 9.8 | -0.29 | no |
| qwen25vl_advqa | 63.5 | 95.3 | 31.9 | [17.2, 37.2] | 26.6 | +5.24 | **yes** |
| qwen25vl_okvqa | 41.5 | 84.3 | 42.8 | [36.7, 45.9] | 30.4 | +12.43 | **yes** |
| qwen25vl_vqav2 | 41.5 | 97.5 | 56.0 | [46.3, 57.4] | 41.1 | +14.93 | **yes** |

**7/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `attn_eigvals` -- Euclidean

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 55.3 | 74.8 | 19.5 | [8.1, 24.6] | 20.0 | -0.50 | no |
| gemma3_12b_okvqa | 48.2 | 65.5 | 17.3 | [12.1, 21.2] | 13.0 | +4.31 | **yes** |
| gemma3_12b_vqav2 | 69.7 | 81.9 | 12.2 | [8.4, 11.7] | 13.4 | -1.19 | no |
| pixtral12b_advqa | 50.5 | 80.9 | 30.3 | [16.0, 33.4] | 27.6 | +2.72 | **yes** |
| pixtral12b_okvqa | 61.1 | 82.4 | 21.3 | [15.4, 18.7] | 19.9 | +1.38 | **yes** |
| pixtral12b_vqav2 | 70.8 | 81.8 | 11.0 | [7.4, 10.4] | 11.2 | -0.19 | no |
| qwen25vl_advqa | 61.8 | 89.1 | 27.3 | [15.2, 30.1] | 24.2 | +3.10 | **yes** |
| qwen25vl_okvqa | 61.7 | 78.5 | 16.8 | [11.3, 20.9] | 13.0 | +3.78 | **yes** |
| qwen25vl_vqav2 | 63.9 | 86.8 | 22.9 | [15.1, 26.0] | 19.6 | +3.29 | **yes** |

**6/9 clear the placebo null** (bar for this cohort: 7) -- **fails the bar**

### `attnlogdet` -- Sigma-metric

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 61.1 | 70.7 | 9.6 | [-10.0, 26.0] | 16.0 | -6.39 | no |
| gemma3_12b_okvqa | 60.3 | 79.4 | 19.1 | [13.0, 20.2] | 16.1 | +2.98 | **yes** |
| gemma3_12b_vqav2 | 46.6 | 69.8 | 23.2 | [17.8, 24.8] | 19.6 | +3.60 | **yes** |
| pixtral12b_advqa | 47.7 | 72.7 | 24.9 | [8.1, 37.4] | 26.9 | -2.00 | no |
| pixtral12b_okvqa | 64.0 | 86.6 | 22.5 | [11.8, 21.3] | 21.9 | +0.68 | **yes** |
| pixtral12b_vqav2 | 56.1 | 67.6 | 11.5 | [3.9, 16.3] | 11.2 | +0.29 | **yes** |
| qwen25vl_advqa | 64.1 | 91.1 | 26.9 | [12.7, 31.6] | 26.0 | +0.97 | **yes** |
| qwen25vl_okvqa | 42.2 | 82.8 | 40.6 | [32.2, 40.3] | 28.4 | +12.22 | **yes** |
| qwen25vl_vqav2 | 46.7 | 95.5 | 48.8 | [37.1, 47.9] | 36.9 | +11.92 | **yes** |

**7/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `attnlogdet` -- Euclidean

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 63.6 | 81.8 | 18.2 | [3.2, 25.1] | 21.7 | -3.48 | no |
| gemma3_12b_okvqa | 68.4 | 82.0 | 13.6 | [7.3, 14.6] | 13.4 | +0.14 | **yes** |
| gemma3_12b_vqav2 | 54.9 | 73.2 | 18.3 | [12.7, 19.3] | 17.8 | +0.54 | **yes** |
| pixtral12b_advqa | 61.7 | 86.6 | 24.9 | [8.2, 30.1] | 26.8 | -1.90 | no |
| pixtral12b_okvqa | 63.1 | 83.8 | 20.7 | [12.2, 18.5] | 21.9 | -1.16 | no |
| pixtral12b_vqav2 | 71.5 | 85.0 | 13.5 | [6.6, 13.4] | 13.2 | +0.29 | **yes** |
| qwen25vl_advqa | 64.7 | 93.6 | 28.9 | [13.5, 31.9] | 26.1 | +2.78 | **yes** |
| qwen25vl_okvqa | 66.8 | 84.2 | 17.5 | [11.7, 17.5] | 15.9 | +1.56 | **yes** |
| qwen25vl_vqav2 | 64.6 | 87.4 | 22.8 | [15.2, 22.4] | 21.6 | +1.18 | **yes** |

**6/9 clear the placebo null** (bar for this cohort: 7) -- **fails the bar**

### `sink` -- Sigma-metric

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 43.0 | 61.3 | 18.4 | [12.3, 24.5] | 15.8 | +2.51 | **yes** |
| gemma3_12b_okvqa | 44.7 | 60.5 | 15.8 | [11.8, 18.9] | 11.1 | +4.73 | **yes** |
| gemma3_12b_vqav2 | 33.5 | 46.5 | 13.0 | [10.6, 16.4] | 10.8 | +2.20 | **yes** |
| pixtral12b_advqa | 41.2 | 69.2 | 28.1 | [20.3, 36.2] | 22.4 | +5.69 | **yes** |
| pixtral12b_okvqa | 37.3 | 59.7 | 22.5 | [18.3, 25.9] | 16.1 | +6.34 | **yes** |
| pixtral12b_vqav2 | 41.9 | 45.8 | 3.9 | [0.8, 7.0] | 5.3 | -1.33 | no |
| qwen25vl_advqa | 50.1 | 78.9 | 28.8 | [18.1, 36.4] | 24.2 | +4.68 | **yes** |
| qwen25vl_okvqa | 40.7 | 79.0 | 38.3 | [32.6, 42.3] | 27.6 | +10.69 | **yes** |
| qwen25vl_vqav2 | 40.4 | 80.6 | 40.2 | [34.9, 43.2] | 29.4 | +10.80 | **yes** |

**8/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

### `sink` -- Euclidean

| pair | theta(0) | theta(1) | delta | boot 95% CI | null p95 | margin | clears? |
|---|---|---|---|---|---|---|---|
| gemma3_12b_advqa | 42.8 | 58.1 | 15.4 | [8.6, 21.4] | 15.4 | -0.03 | no |
| gemma3_12b_okvqa | 52.0 | 63.7 | 11.7 | [7.1, 15.5] | 8.4 | +3.37 | **yes** |
| gemma3_12b_vqav2 | 40.4 | 51.0 | 10.6 | [7.0, 14.3] | 9.8 | +0.76 | **yes** |
| pixtral12b_advqa | 41.2 | 70.2 | 29.1 | [19.9, 34.4] | 24.4 | +4.71 | **yes** |
| pixtral12b_okvqa | 36.5 | 57.2 | 20.7 | [17.0, 24.7] | 16.3 | +4.43 | **yes** |
| pixtral12b_vqav2 | 47.7 | 53.6 | 5.9 | [1.9, 11.1] | 6.6 | -0.79 | no |
| qwen25vl_advqa | 44.0 | 67.7 | 23.7 | [15.6, 28.4] | 19.1 | +4.54 | **yes** |
| qwen25vl_okvqa | 54.6 | 75.9 | 21.3 | [15.1, 24.9] | 18.5 | +2.85 | **yes** |
| qwen25vl_vqav2 | 42.8 | 68.3 | 25.5 | [19.9, 29.0] | 19.6 | +5.86 | **yes** |

**7/9 clear the placebo null** (bar for this cohort: 7) -- **PASS**

<a id='s5'></a>
## 5. Summary and stability

Pass counts out of 9 pairs (bar = 7):

| family | Sigma | verdict | Euclid | verdict |
|---|---|---|---|---|
| `hs_wide` | 8/9 | **PASS** | 7/9 | **PASS** |
| `lapeigvals` | 8/9 | **PASS** | 8/9 | **PASS** |
| `hs_narrow` | 8/9 | **PASS** | 6/9 | fail |
| `hs_peak_only` | 8/9 | **PASS** | 6/9 | fail |
| `attn_eigvals` | 7/9 | **PASS** | 6/9 | fail |
| `attnlogdet` | 7/9 | **PASS** | 6/9 | fail |
| `sink` | 8/9 | **PASS** | 7/9 | **PASS** |

**How close are these calls?** Of 126 (pair, family, metric) cells: 9 sit within 0.5 deg of the threshold, 18 within 1 deg, and 45 within 2 deg. Median margin is +3.52 deg for passing cells and -1.44 for failing ones.

Per-pair totals out of 14 cells (7 families x 2 metrics):

| pair | cells passed |
|---|---|
| pixtral12b_vqav2 | 2/14 |
| gemma3_12b_advqa | 6/14 |
| pixtral12b_advqa | 12/14 |
| gemma3_12b_okvqa | 12/14 |
| gemma3_12b_vqav2 | 13/14 |
| pixtral12b_okvqa | 13/14 |
| qwen25vl_advqa | 14/14 |
| qwen25vl_okvqa | 14/14 |
| qwen25vl_vqav2 | 14/14 |

<a id='s6'></a>
## 6. Cross-check: the entropy reference

The same angles measured against the *entropy* direction rather than the alpha=0 probe. Entropy is noise-free, so this is the cleaner reference -- but it saturates near 90 deg, which is why it is a cross-check and not the headline.

| family | metric | a=0.00 | a=0.25 | a=0.50 | a=0.75 | a=1.00 |
|---|---|---|---|---|---|---|
| `hs_wide` | Sigma-metric | 65.0 | 66.2 | 68.8 | 73.4 | 81.1 |
| `hs_wide` | Euclidean | 88.2 | 88.4 | 89.3 | 89.9 | 90.7 |
| `lapeigvals` | Sigma-metric | 40.4 | 44.3 | 49.7 | 60.1 | 71.1 |
| `lapeigvals` | Euclidean | 45.9 | 49.4 | 52.0 | 55.5 | 67.9 |
| `hs_narrow` | Sigma-metric | 62.4 | 64.8 | 68.4 | 73.1 | 79.4 |
| `hs_narrow` | Euclidean | 88.6 | 88.8 | 89.3 | 89.8 | 90.6 |
| `hs_peak_only` | Sigma-metric | 62.8 | 64.0 | 67.5 | 72.2 | 78.8 |
| `hs_peak_only` | Euclidean | 88.6 | 88.8 | 89.3 | 89.9 | 90.6 |
| `attn_eigvals` | Sigma-metric | 62.4 | 65.4 | 67.1 | 69.3 | 81.0 |
| `attn_eigvals` | Euclidean | 63.9 | 67.0 | 70.5 | 77.5 | 87.5 |
| `attnlogdet` | Sigma-metric | 58.8 | 62.4 | 66.9 | 72.2 | 83.1 |
| `attnlogdet` | Euclidean | 67.0 | 69.6 | 74.0 | 79.7 | 86.3 |
| `sink` | Sigma-metric | 39.7 | 42.8 | 48.1 | 57.0 | 68.1 |
| `sink` | Euclidean | 41.0 | 44.4 | 48.9 | 56.8 | 65.9 |

<a id='s7'></a>
## 7. Caveats

- **Many verdicts are knife-edges.** See the margin counts in section 5. A cell that clears the null by 0.2 deg should not be reported with the same confidence as one that clears by 10.

- **The Euclidean angle saturates.** As `theta(1)` approaches 90 deg the metric loses resolution: two genuinely different rotations both read as 'near-orthogonal'. Check `theta(1)` in section 4 before reading a large Euclidean `delta` as a large effect.

- **Angles above 90 deg appear in some cells.** These are obtuse weight-vector angles, not errors, but they mean 'the axis reversed past orthogonal' and should not be averaged naively with acute ones.

- **A pass is not an effect size.** The rule asks whether rotation exceeds what shrinkage alone produces. It does not say the residual rotation is large or that the probe was *only* tracking confidence.

- **`n_train` shrinks with alpha**, so the alpha=1 probe is fit on the least data. That is exactly what the placebo controls for, and it is why the placebo -- not the raw `delta` -- is the result.

<a id='sA'></a>
## Appendix A -- what `null_p95` is, and why it decides everything

**One sentence.** `null_p95` is how much the probe would rotate from having less training data *alone* -- so a real rotation has to beat it to count for anything.

### The problem it solves

We measure `delta = theta(1) - theta(0)`: how far the probe's direction moved once the confidence-correctness link was stripped out of its training set.

`delta` is almost guaranteed to come out positive even when nothing has been demonstrated. Entropy-matching works by **throwing rows away**, so the `alpha=1` probe is fit on far less data than the `alpha=0` one (see section 2). Fit any model on less data and its weight vector wobbles from noise alone. Some of `delta` is signal and some is just data loss, and the raw number cannot tell them apart.

### How the null is built

Build training sets at **the same sizes as the alpha ladder**, but choose the rows **at random** instead of by entropy-matching. The confidence structure is left completely intact -- the only thing that changed is the row count. So any rotation measured here is, by construction, pure shrinkage noise.

Take 20 such draws at `alpha=0`'s size and 20 at `alpha=1`'s size, then compare **every** a=1 draw against **every** a=0 draw: 20 x 20 = **400 differences**. That is the null distribution -- 400 samples of *how much does shrinkage alone rotate this probe?*

`null_p95` is the **95th percentile** of those 400. Shrinkage alone exceeds it only 5% of the time. The rule `delta > null_p95` is therefore an ordinary 5%-significance test whose critical value happens to be measured in degrees rather than expressed as a p-value.

### One design subtlety

The null uses all 400 **outer** differences, not the 20 paired ones. The placebo draws at the two rungs are independent, so pairing them by index would impose a correspondence that does not exist, understate the null's spread, and make the test *easier* to pass than it should be. (The bootstrap CI on `delta` **is** paired by draw index -- there the two refits genuinely share a resampled row set, so pairing removes variance common to both.)

### Why it matters, from this cohort

| pair | family | metric | delta | null p95 | verdict |
|---|---|---|---|---|---|
| qwen25vl_vqav2 | `attn_eigvals` | Sigma | 56.0 deg | 41.1 deg | **passes** by 14.9 deg |
| pixtral12b_advqa | `attnlogdet` | Sigma | 24.9 deg | 26.9 deg | **fails** by 2.0 deg |

`pixtral12b_advqa`'s probe on `attnlogdet` genuinely rotated **24.9 degrees** -- that is not a small movement. But for that pair and family, shrinking the training set that far routinely produces about 26.9 degrees of rotation on its own. So the whole effect is explained by data loss and says nothing about confidence.

**That is the entire point of the control.** Without it, 24.9 degrees would have been reported as a result.

