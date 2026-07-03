"""
Experiment 3 — value-vector NORM extraction (the only GPU step in the sink-hole plan).

feature_3[l,h,rank] = s_i * ||V_i||. The sink score s_i (= attn_diag + lap_diag)
is already cached, so the ONLY missing ingredient is ||V_i|| — the L2 norm of the
per-head value vector at each token position. This script re-runs one teacher-forced
forward pass per example over (question + greedy answer), using the SAME
_answer_token_span tokenisation as extract_attention.py so token positions align
1:1 with the cached diagonals, and captures each layer's v_proj output via a forward
hook (no attention weights needed -> sdpa is fine and faster than eager).

GQA note: v_proj outputs n_kv_heads*head_dim, so ||V_i|| is per KV-head. The feature
builder maps query head h -> kv head h // (n_q_heads // n_kv_heads). We store the
per-KV-head norms [L, n_kv, S]; the query->kv broadcast happens at feature-build time.

Outputs (one file per pair+split):
  results/sinkhole/value_norms/{model}_{dataset}_{split}.pt
    {"value_norms": [Tensor[L, n_kv, S] fp16], "labels": LongTensor[N],
     "ids": [str], "categories": [str|None],
     "n_q_heads": int, "n_kv_heads": int, "head_dim": int,
     "model": str, "dataset": str, "split": str}

Usage:
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY extract_value_norms.py --model llama --dataset sciq
  CUDA_VISIBLE_DEVICES=0 $PY extract_value_norms.py --all
  # split across two GPUs by hand, e.g.:
  #   CUDA_VISIBLE_DEVICES=0 ... --model llama --dataset sciq &
  #   CUDA_VISIBLE_DEVICES=1 ... --model mistral --dataset sciq &
"""
from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

RANKING_DIR = Path(__file__).resolve().parents[1] / "ranking"
sys.path.insert(0, str(RANKING_DIR))
from config import ExperimentConfig          # noqa: E402
from data_loader import load_split, Sample   # noqa: E402
from model_utils import _answer_token_span   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"
OUT_DIR = REPO_ROOT / "results" / "sinkhole" / "value_norms"

MODEL_MAP = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "gemma":   "google/gemma-3-12b-it",
}
PAIRS = [(m, d) for m in MODEL_MAP for d in ("sciq", "triviaqa", "math")]
SPLITS = ("train", "test")

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def make_cfg(model: str, dataset: str, gpu_id: int) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=DATA_DIR / f"ranking_experiment_{model}_{dataset}",
        model_name=MODEL_MAP[model], gpu_id=gpu_id,
        dtype="bfloat16", alpha=1.0, layer_1idx=1,
    )


def head_geometry(model_name: str):
    c = AutoConfig.from_pretrained(model_name)
    tc = getattr(c, "text_config", c)  # gemma3 is multimodal -> text_config
    nq = int(tc.num_attention_heads)
    nkv = int(getattr(tc, "num_key_value_heads", nq))
    hd = getattr(tc, "head_dim", None)
    if hd is None:
        hd = int(tc.hidden_size) // nq
    return nq, nkv, int(hd)


def load_model(model_name: str, gpu_id: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def register_vproj_hooks(model, captured: dict, expect_layers: int | None = None):
    """Hook every language-model '*.self_attn.v_proj'. Excludes any vision-tower
    modules (gemma-3 is multimodal: its vision encoder also has self_attn.v_proj,
    which would otherwise inflate the layer count). Returns (handles, layer_order)."""
    layers = []
    for name, mod in model.named_modules():
        if name.endswith("self_attn.v_proj") and "vision" not in name.lower():
            m = _LAYER_RE.search(name)
            if m is None:
                raise RuntimeError(f"could not parse layer index from {name}")
            layers.append((int(m.group(1)), mod))
    layers.sort(key=lambda t: t[0])
    if expect_layers is not None and len(layers) != expect_layers:
        raise RuntimeError(
            f"hooked {len(layers)} v_proj layers but config expects {expect_layers} "
            f"text layers — check vision-module filtering")

    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            captured[idx] = out.detach()[0]  # [T, n_kv*head_dim]
        return hook

    for idx, mod in layers:
        handles.append(mod.register_forward_hook(make_hook(idx)))
    return handles, [idx for idx, _ in layers]


@torch.no_grad()
def value_norms_one(s: Sample, model, tokenizer, device, captured, layer_order,
                    n_kv: int, head_dim: int):
    greedy = (s.greedy_prediction or "").strip()
    if greedy == "":
        return None
    full_ids, _ans, _k = _answer_token_span(tokenizer, s.question, greedy)
    input_ids = torch.tensor([full_ids], device=device)

    captured.clear()
    model(input_ids=input_ids, use_cache=False)

    norms = []
    for idx in layer_order:
        v = captured[idx].float()                 # [T, n_kv*head_dim]
        T = v.shape[0]
        v = v.view(T, n_kv, head_dim)
        norms.append(v.norm(dim=-1).transpose(0, 1).half().cpu())  # [n_kv, T]
    vn = torch.stack(norms)                        # [L, n_kv, T]
    del input_ids
    return vn


def process_pair(model_key, dataset, gpu_id, overwrite, limit):
    cfg = make_cfg(model_key, dataset, gpu_id)
    targets = {sp: OUT_DIR / f"{model_key}_{dataset}_{sp}.pt" for sp in SPLITS}
    if all(p.exists() for p in targets.values()) and not overwrite and not limit:
        print(f"=== {model_key}/{dataset} === all splits cached — skipping", flush=True)
        return

    model_name = MODEL_MAP[model_key]
    nq, nkv, hd = head_geometry(model_name)
    cfg_full = AutoConfig.from_pretrained(model_name)
    n_text_layers = int(getattr(getattr(cfg_full, "text_config", cfg_full),
                                "num_hidden_layers"))
    print(f"\n=== {model_key}/{dataset} === loading {model_name} "
          f"(n_q={nq} n_kv={nkv} head_dim={hd} text_layers={n_text_layers})", flush=True)
    t0 = time.time()
    model, tokenizer, device = load_model(model_name, gpu_id)
    captured: dict = {}
    handles, layer_order = register_vproj_hooks(model, captured, expect_layers=n_text_layers)
    print(f"  model loaded in {time.time()-t0:.1f}s; hooked {len(layer_order)} v_proj layers", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out_path = targets[split]
        if out_path.exists() and not overwrite and not limit:
            print(f"  [{split}] cached — skipping", flush=True)
            continue
        samples = load_split(cfg, split)
        if limit:
            samples = samples[:limit]
        vns, labels, ids, cats = [], [], [], []
        n_skip = 0
        for s in tqdm(samples, desc=f"{model_key}/{dataset}/{split}"):
            vn = value_norms_one(s, model, tokenizer, device, captured,
                                 layer_order, nkv, hd)
            if vn is None:
                n_skip += 1
                continue
            vns.append(vn)
            labels.append(int(not s.open_text_label))
            ids.append(s.id)
            cats.append(s.category)
        payload = {
            "value_norms": vns,
            "labels": torch.tensor(labels, dtype=torch.long),
            "ids": ids, "categories": cats,
            "n_q_heads": nq, "n_kv_heads": nkv, "head_dim": hd,
            "model": model_key, "dataset": dataset, "split": split,
        }
        if limit:
            out_path = out_path.with_suffix(".smoke.pt")
        torch.save(payload, out_path)
        print(f"  [{split}] wrote {len(ids)} examples (skipped empty greedy={n_skip}) "
              f"-> {out_path}", flush=True)

    for h in handles:
        h.remove()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_MAP))
    p.add_argument("--dataset", choices=["sciq", "triviaqa", "math"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="cap examples per split (smoke test)")
    args = p.parse_args()

    if args.all:
        pairs = PAIRS
    elif args.model and args.dataset:
        pairs = [(args.model, args.dataset)]
    else:
        raise SystemExit("specify --all or both --model and --dataset")

    for model_key, dataset in pairs:
        process_pair(model_key, dataset, args.gpu_id, args.overwrite, args.limit)


if __name__ == "__main__":
    main()