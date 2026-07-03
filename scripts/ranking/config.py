"""Shared configuration for the QA-recoverability ranking experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
EXPERIMENT_DIR = DATA_DIR / "ranking_experiment"
SPLITS_DIR = EXPERIMENT_DIR / "splits"
CACHE_DIR = EXPERIMENT_DIR / "cache"
PROBE_DIR = EXPERIMENT_DIR / "probe"
RESULTS_DIR = EXPERIMENT_DIR / "results"

DEFAULT_INPUT_CSV = DATA_DIR / "sampled_600_distractors.csv"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT_TEMPLATE = "Question: {q}\nAnswer: {a}"
PROMPT_PREFIX_TEMPLATE = "Question: {q}\nAnswer: "


@dataclass
class ExperimentConfig:
    input_csv: Path = DEFAULT_INPUT_CSV
    output_dir: Path = EXPERIMENT_DIR
    model_name: str = DEFAULT_MODEL
    layer_1idx: int = 16          # which transformer block output to probe (1-indexed)
    alpha: float = 1.0            # length-normalisation exponent for s_ext
    test_fraction: float = 0.30
    seed: int = 42
    device: str = "cuda"
    gpu_id: int = 0
    device_map: Optional[str] = None   # set to "auto" to shard across all visible GPUs
    dtype: str = "bfloat16"       # "bfloat16" | "float16" | "float32"
    probe_C: float = 1.0
    probe_max_iter: int = 1000
    answer_column: str = "low_t_generation"
    candidate_column: str = "candidate_list"
    verdict_column: str = "LLM_verdict"
    id_column: str = "id"
    question_column: str = "question"
    gt_column: str = "ground_truth"

    def paths(self) -> dict:
        return {
            "splits": self.output_dir / "splits",
            "cache": self.output_dir / "cache",
            "probe": self.output_dir / "probe",
            "results": self.output_dir / "results",
        }


def ensure_dirs(cfg: ExperimentConfig) -> None:
    for p in cfg.paths().values():
        p.mkdir(parents=True, exist_ok=True)
