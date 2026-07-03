# hallucination-detection-repro

Reproduction package for the per-category hallucination-detection study —
**Part 9** (band-stratified correctness AUROC) and **Part 11** (band-specialist
LR) across 4 models × 3 datasets.

**Start here → [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)**

Quick version:

```bash
conda env create -f environment/semantic_uncertainty.yml
conda env create -f environment/snne.yml
./reproduce.sh all          # GPU extraction + SNNE + CPU analysis
```

Outputs: `results/per_category_analysis/per_category_pairwise_auroc.csv` (Part 9)
and `per_category_band_specialist_auroc.csv` (Part 11).

No GPU? Restore the precomputed-artifacts tarball (~36 GB) into the package root
and run only `./reproduce.sh prep && ./reproduce.sh analyze`.