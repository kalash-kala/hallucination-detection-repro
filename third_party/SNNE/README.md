# SNNE

This repository is the official implementation of our ACL Findings 2025 paper [Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity](https://arxiv.org/abs/2506.00245).

## 🔗 Quick Links
- [SNNE](#snne)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Data Preparation](#data-preparation)
  - [Demo](#demo)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Install Requirements
```bash
conda env create -f environment.yaml
conda activate snne
pip install flash-attn==2.6.1 --no-build-isolation
pip install -e .
```

## Data Preparation
For almost all tasks, the dataset is downloaded automatically from the Hugging Face Datasets library upon first execution.
The only exception is BioASQ (task b, BioASQ11, 2023), for which the data needs to be [downloaded](http://participants-area.bioasq.org/datasets) manually and stored at `./data/bioasq/training11b.json`.

## Demo
### Generate answers + Compute SE, NE, DSE, and pTrue
- QA
```bash
./scripts/generate/generate_qa.sh
```

- Summarization
```bash
./scripts/generate/generate_summarization.sh
```

- Translation
```bash
./scripts/generate/generate_translation.sh
```

### Compute other UQ methods
- SNNE and WSNNE
```bash
./scripts/compute/compute_snne.sh
```

- Graph baselines (SumEigv, Deg, Eccen) + NumSet + LexSim
```bash
./scripts/compute/compute_graph_baselines.sh
```

- KLE
```bash
./scripts/compute/compute_kle.sh
```

- LUQ
```bash
./scripts/compute/compute_luq.sh
```

- SAR
```bash
./scripts/compute/compute_sar.sh
```

- Eigenscore
```bash
./scripts/compute/compute_eigenscore.sh
```

### Evaluation
- **SE, NE, DSE, and pTrue**: Open the Jupyter notebook in `notebooks/evaluation.ipynb`, populate the `wandb_id` variable in the second cell with the id assigned to your run, and execute all cells of the notebook.
- **Other methods**: Open the csv files in the corresponding folder `*_results` and find the evaluation metrics.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Dang Nguyen (nguyentuanhaidang@gmail.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{nguyen2025beyond,
  title={Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity},
  author={Nguyen, Dang and Payani, Ali and Mirzasoleiman, Baharan},
  journal={In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025}
}
```

## Acknowledgements
The structure of this repo is largely based on [semantic_uncertainty](https://github.com/jlko/semantic_uncertainty). The graph baselines are adapted from [UQ-NLG](https://github.com/zlin7/UQ-NLG) while summarization and translation parts are adapted from [lm-polygraph](https://github.com/IINemo/lm-polygraph). We are very grateful for their open sources.