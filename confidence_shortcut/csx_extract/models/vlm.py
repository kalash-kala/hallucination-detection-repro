"""Loading a VLM and building one teacher-forced forward pass.

Three families, three ways to get it silently wrong. Each is handled explicitly
below with the reason, because the failure mode in every case is *plausible
output*, not an exception -- see the module docstring in ../spans.py.

The governing rule: **pass the processor's output through wholesale**. Every
family puts its image geometry in a differently-named key (`token_type_ids`,
`image_grid_thw`, `image_sizes`), and dropping one gives wrong numbers rather
than an error. So nothing is filtered out; only `attention_mask` is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from csx_common import registry

from .. import prompts


@dataclass
class LoadedVLM:
    model: torch.nn.Module
    processor: object
    pair: registry.Pair
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    hidden_dim: int

    @property
    def device(self):
        return self.model.device


def _text_config(cfg):
    return getattr(cfg, "text_config", cfg)


def load(pair_key: str, *, attn_implementation: str = "sdpa",
         dtype=torch.bfloat16) -> LoadedVLM:
    """Load a VLM for extraction.

    `attn_implementation` is the phase switch: phase 1 uses "sdpa" (fast, no
    attention weights), phase 2 needs "eager" because sdpa never materialises the
    [S,S] matrix the spectral families are built from.
    """
    pair = registry.get(pair_key)
    hf_id = pair.model.hf_id

    processor = AutoProcessor.from_pretrained(hf_id)
    tokenizer = processor.tokenizer

    # Pixtral ships without a pad token, and any batched processor call raises
    # without one. Set before the first processor call, not after.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Single explicit device, not "auto": every backbone here fits on one 80GB
    # GPU (phase 1 proved it, under sdpa), and this pipeline forwards one row at
    # a time -- there is no batch to shard for. "auto" balances across every
    # CUDA-visible device by its own memory heuristic regardless of need, and
    # under eager attention (which materialises the full [S,S] matrix) that
    # heuristic split a 7-12B model across two GPUs, so a single row's per-layer
    # attention tensors landed on different devices and torch.stack in
    # phase2_attention.py raised. `{"": 0}` pins everything to CUDA device 0 of
    # whatever CUDA_VISIBLE_DEVICES exposes.
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation=attn_implementation,
    ).eval()

    cfg = AutoConfig.from_pretrained(hf_id)
    tc = _text_config(cfg)
    n_layers = int(tc.num_hidden_layers)
    hidden_dim = int(tc.hidden_size)
    n_kv = int(getattr(tc, "num_key_value_heads", 0) or 0)
    # Pixtral's text_config omits num_attention_heads (it inherits Mistral's
    # default while overriding head_dim), so it is derived rather than assumed.
    n_q = getattr(tc, "num_attention_heads", None)
    if n_q is None:
        head_dim = int(getattr(tc, "head_dim", 0) or 0)
        if not head_dim:
            raise RuntimeError(
                f"{hf_id}: config declares neither num_attention_heads nor "
                f"head_dim; cannot determine the query-head count")
        n_q = hidden_dim // head_dim
    n_q = int(n_q)

    declared = pair.model.layers
    if declared is not None and declared != n_layers:
        raise RuntimeError(
            f"{pair_key}: pairs.yaml declares layers={declared} but "
            f"{hf_id} has {n_layers}. Fix the registry rather than proceeding -- "
            f"the layer buckets and feature widths depend on it.")

    return LoadedVLM(model=model, processor=processor, pair=pair,
                     n_layers=n_layers, n_q_heads=n_q, n_kv_heads=n_kv,
                     hidden_dim=hidden_dim)


def build_inputs(lv: LoadedVLM, image_path: str, question: str, answer: str):
    """Teacher-forced (prompt + answer) inputs for one row.

    Returns `(inputs, n_prompt_tokens, n_answer_tokens)`. The prompt is built
    through the chat template so it matches what the generation run produced, and
    the answer tokens are appended without special tokens so the boundary is
    exact.
    """
    inputs = _processor_call(lv, image_path, question)
    # Wholesale: token_type_ids / image_grid_thw / image_sizes / pixel_values all
    # survive. Filtering here is exactly the bug this module exists to avoid.
    n_prompt = int(inputs["input_ids"].shape[1])

    answer_ids = lv.processor.tokenizer(
        answer, return_tensors="pt", add_special_tokens=False)["input_ids"]
    n_answer = int(answer_ids.shape[1])
    if n_answer == 0:
        # An empty greedy answer has no answer span to pool over. Caller skips
        # the row rather than emitting a degenerate one.
        return None, n_prompt, 0

    inputs["input_ids"] = torch.cat(
        [inputs["input_ids"], answer_ids.to(inputs["input_ids"].device)], dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    _extend_per_position_keys(inputs, n_prompt, n_answer)

    inputs = {k: (v.to(lv.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    return inputs, n_prompt, n_answer


# Per-position keys the processor emits alongside input_ids. Each is a [1, S]
# parallel array describing what kind of token sits at each position, and each is
# consumed by a different family:
#   token_type_ids     gemma-3   -- marks image positions
#   mm_token_type_ids  Qwen2.5-VL -- drives the 3-D rope index
_PER_POSITION = ("token_type_ids", "mm_token_type_ids")

# Value meaning "ordinary text" in those arrays. Answer tokens are text.
_TEXT_TYPE = 0


def _extend_per_position_keys(inputs: dict, n_prompt: int, n_answer: int) -> None:
    """Grow the per-position arrays to cover the teacher-forced answer.

    Passing the processor's keys through wholesale is necessary but NOT
    sufficient: these arrays are parallel to input_ids, so appending answer
    tokens without extending them leaves a [1, n_prompt] array against a
    [1, n_prompt+n_answer] sequence.

    Qwen2.5-VL fails loudly on that (get_rope_index indexes the array with the
    attention mask). gemma-3 does NOT -- its token_type_ids would simply be short,
    and the image-position marking would silently apply to the wrong tokens. So
    this is handled generically rather than per-model.
    """
    for key in _PER_POSITION:
        v = inputs.get(key)
        if v is None or not hasattr(v, "shape") or v.ndim != 2:
            continue
        if v.shape[1] != n_prompt:
            raise RuntimeError(
                f"{key} has length {v.shape[1]} against a {n_prompt}-token "
                f"prompt; it is not the per-position array this assumes")
        pad = torch.full((v.shape[0], n_answer), _TEXT_TYPE,
                         dtype=v.dtype, device=v.device)
        inputs[key] = torch.cat([v, pad], dim=1)


def _processor_call(lv: LoadedVLM, image_path: str, question: str):
    """One processor invocation, with the run's exact user-turn wording.

    `prompts.text_for` supplies the instruction the generation run used, not a
    reasonable-looking paraphrase: the answer being teacher-forced was produced
    under that exact prompt, so anything else scores it out of context.
    """
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image"},
                    {"type": "text", "text": prompts.text_for(lv.pair, question)}],
    }]
    prompt = lv.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return lv.processor(images=[image], text=[prompt], return_tensors="pt")


def prompt_input_ids(lv: LoadedVLM, image_path: str, question: str):
    """Just the processor's prompt token ids, for the prompt/span cross-check."""
    return _processor_call(lv, image_path, question)["input_ids"][0].tolist()
