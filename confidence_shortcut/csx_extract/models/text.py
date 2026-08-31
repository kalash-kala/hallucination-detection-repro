"""Loading a text backbone and building one teacher-forced forward pass.

The prompt is the published `ranking` template -- "Question: {q}\\nAnswer: " with
the greedy answer teacher-forced after it -- so pairs extracted here sit in the
same feature space as the eight qa8 pairs that come in through the legacy
adapter. That is what makes a median over a mixed group meaningful rather than a
comparison of two different prompt conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from csx_common import registry

PROMPT = "Question: {question}\nAnswer:"


@dataclass
class LoadedLM:
    model: torch.nn.Module
    tokenizer: object
    pair: registry.Pair
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    hidden_dim: int

    @property
    def device(self):
        return self.model.device


def load(pair_key: str, *, attn_implementation: str = "sdpa",
         dtype=torch.bfloat16) -> LoadedLM:
    pair = registry.get(pair_key)
    hf_id = pair.model.hf_id

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pinned to a single device -- see the matching note in models/vlm.py: "auto"
    # can split the model across every visible GPU regardless of need, which
    # breaks phase 2's cross-layer torch.stack over attention tensors.
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation=attn_implementation,
    ).eval()

    cfg = AutoConfig.from_pretrained(hf_id)
    tc = getattr(cfg, "text_config", cfg)
    n_layers = int(tc.num_hidden_layers)
    declared = pair.model.layers
    if declared is not None and declared != n_layers:
        raise RuntimeError(
            f"{pair_key}: pairs.yaml declares layers={declared} but {hf_id} has "
            f"{n_layers}; the layer buckets depend on this being right.")

    return LoadedLM(
        model=model, tokenizer=tokenizer, pair=pair, n_layers=n_layers,
        n_q_heads=int(tc.num_attention_heads),
        n_kv_heads=int(getattr(tc, "num_key_value_heads", tc.num_attention_heads)),
        hidden_dim=int(tc.hidden_size),
    )


def build_inputs(lm: LoadedLM, question: str, answer: str):
    """Teacher-forced (prompt + answer). Returns (inputs, n_prompt, n_answer)."""
    prompt = PROMPT.format(question=question)
    p_ids = lm.tokenizer(prompt, return_tensors="pt")["input_ids"]
    # Leading space matters: " Paris" and "Paris" tokenise differently, and the
    # published template scores the answer as a continuation of "Answer:".
    a_ids = lm.tokenizer(" " + answer.strip(), return_tensors="pt",
                         add_special_tokens=False)["input_ids"]
    n_prompt, n_answer = int(p_ids.shape[1]), int(a_ids.shape[1])
    if n_answer == 0:
        return None, n_prompt, 0
    input_ids = torch.cat([p_ids, a_ids], dim=1).to(lm.device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    return inputs, n_prompt, n_answer


def prompt_token_ids(lm: LoadedLM, question: str) -> list[int]:
    prompt = PROMPT.format(question=question)
    return lm.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
