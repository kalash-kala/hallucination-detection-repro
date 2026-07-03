"""Shared causal-LM helpers: answer-token log probabilities and layer-l hidden states.

The same model instance powers both the external scorer (length-normalised log P)
and the hidden-state source for the internal probe, so it is cached in a global.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ExperimentConfig, PROMPT_TEMPLATE, PROMPT_PREFIX_TEMPLATE


_MODEL: Optional[AutoModelForCausalLM] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_DEVICE: Optional[torch.device] = None


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def load_model(cfg: ExperimentConfig):
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    if cfg.device_map is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu_id))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = dict(torch_dtype=_dtype_from_str(cfg.dtype), low_cpu_mem_usage=True)
    if cfg.device_map:
        kwargs["device_map"] = cfg.device_map
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    model.eval()

    if not cfg.device_map:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = torch.device("cuda")

    _MODEL, _TOKENIZER, _DEVICE = model, tokenizer, device
    return model, tokenizer, device


def _answer_token_span(tokenizer, q: str, a: str) -> tuple[list[int], list[int], int]:
    """Return (full_ids, answer_ids, prefix_len) where answer tokens are the suffix."""
    prefix = PROMPT_PREFIX_TEMPLATE.format(q=q)
    full = PROMPT_TEMPLATE.format(q=q, a=a)
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=True)["input_ids"]

    # Tokenisation of "...Answer: " + "answer" is usually prefix-stable, but guard
    # against merged boundary tokens by trimming to the longest shared prefix.
    n = min(len(prefix_ids), len(full_ids))
    k = 0
    while k < n and prefix_ids[k] == full_ids[k]:
        k += 1
    if k == 0:
        raise RuntimeError("Unexpected tokenisation: no shared prefix between prompt and full sequence.")
    answer_ids = full_ids[k:]
    if len(answer_ids) == 0:
        # Degenerate: answer tokenises to zero new tokens; fall back to last token of full_ids.
        answer_ids = [full_ids[-1]]
        k = len(full_ids) - 1
    return full_ids, answer_ids, k


@torch.no_grad()
def answer_logprob_and_len(q: str, a: str, cfg: ExperimentConfig) -> tuple[float, int]:
    """Teacher-forced sum of log P over answer tokens, plus the answer-token count."""
    model, tokenizer, device = load_model(cfg)
    full_ids, answer_ids, prefix_len = _answer_token_span(tokenizer, q, a)
    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids=input_ids).logits[0]      # (T, V)
    # logits[i] predicts token at position i+1, so positions prefix_len..T-1 come from logits[prefix_len-1..T-2].
    target_positions = list(range(prefix_len, len(full_ids)))
    pred_positions = [p - 1 for p in target_positions]
    pred_logits = logits[pred_positions]               # (A, V)
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    tgt = torch.tensor(answer_ids, device=device)
    token_lps = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return float(token_lps.sum().item()), len(answer_ids)


@torch.no_grad()
def hidden_state_last_answer(q: str, a: str, layer_1idx: int, cfg: ExperimentConfig) -> np.ndarray:
    """Hidden state of the last answer token at the requested transformer layer.

    `layer_1idx=1` -> output of the first transformer block.
    `layer_1idx=L` (L = num_hidden_layers) -> output of the last block.
    `hidden_states[0]` (embeddings) is never selectable with this API.
    """
    model, tokenizer, device = load_model(cfg)
    full_ids, _answer_ids, _ = _answer_token_span(tokenizer, q, a)
    input_ids = torch.tensor([full_ids], device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states                # tuple length L+1, [0]=embeddings
    if layer_1idx < 1 or layer_1idx >= len(hidden_states):
        raise ValueError(
            f"layer_1idx={layer_1idx} out of range; model has {len(hidden_states) - 1} transformer blocks."
        )
    h = hidden_states[layer_1idx][0, -1].float().cpu().numpy()
    return h


@torch.no_grad()
def score_and_hidden(q: str, a: str, layer_1idx: int, cfg: ExperimentConfig) -> tuple[float, int, np.ndarray]:
    """Single forward pass returning log P, answer length, and the hidden state."""
    model, tokenizer, device = load_model(cfg)
    full_ids, answer_ids, prefix_len = _answer_token_span(tokenizer, q, a)
    input_ids = torch.tensor([full_ids], device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    logits = outputs.logits[0]
    target_positions = list(range(prefix_len, len(full_ids)))
    pred_positions = [p - 1 for p in target_positions]
    log_probs = torch.log_softmax(logits[pred_positions].float(), dim=-1)
    tgt = torch.tensor(answer_ids, device=device)
    sum_lp = float(log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item())

    hidden_states = outputs.hidden_states
    if layer_1idx < 1 or layer_1idx >= len(hidden_states):
        raise ValueError(
            f"layer_1idx={layer_1idx} out of range; model has {len(hidden_states) - 1} transformer blocks."
        )
    h = hidden_states[layer_1idx][0, -1].float().cpu().numpy()
    return sum_lp, len(answer_ids), h


@torch.no_grad()
def score_and_hidden_multilayer(
    q: str, a: str, layers: list, cfg: ExperimentConfig
) -> tuple[float, int, dict]:
    """Single forward pass; returns log P, answer length, and hidden states for every requested layer.

    Returns:
        (sum_logprob, n_answer_tokens, {layer_1idx: np.ndarray shape [hidden_size]})
    """
    model, tokenizer, device = load_model(cfg)
    full_ids, answer_ids, prefix_len = _answer_token_span(tokenizer, q, a)
    input_ids = torch.tensor([full_ids], device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    logits = outputs.logits[0]
    target_positions = list(range(prefix_len, len(full_ids)))
    pred_positions = [p - 1 for p in target_positions]
    log_probs = torch.log_softmax(logits[pred_positions].float(), dim=-1)
    tgt = torch.tensor(answer_ids, device=device)
    sum_lp = float(log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item())

    hidden_states = outputs.hidden_states   # tuple length num_layers+1; [0]=embeddings
    num_blocks = len(hidden_states) - 1
    hs: dict = {}
    for L in layers:
        if L < 1 or L > num_blocks:
            raise ValueError(f"layer_1idx={L} out of range; model has {num_blocks} transformer blocks.")
        hs[L] = hidden_states[L][0, -1].float().cpu().numpy()
    return sum_lp, len(answer_ids), hs
