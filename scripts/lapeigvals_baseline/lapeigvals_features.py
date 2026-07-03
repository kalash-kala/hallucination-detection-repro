"""
Standalone copies of the LapEigvals feature math (pure torch, no deps).

Lifted verbatim (with light docstring edits) from the official EMNLP 2025
implementation of "Hallucination Detection in LLMs Using Spectral Features of
Attention Maps" (Binkowski et al., 2025), arXiv:2502.17598:

  - attention_diagonal / laplacian_diagonal_from_attn
        <- hallucinations/features/attention_weights.py   (vertical_edges=False
           branch, the variant used by the repro stage train_attn_vs_laplacian)
  - get_attn_log_det / get_attn_eigvals_per_head_topk /
    get_laplacian_eigvals_per_head_topk
        <- hallucinations/features/attn_feats.py

We copy rather than import to avoid pulling the upstream package's config/loguru/
DVC dependencies (mirrors how scripts/snne_baseline/snne_core.py made wandb-free
copies). The three trainable variants these features back are:

  AttnLogDet   = get_attn_log_det(attn_diags)            (the LLMCheck baseline)
  AttnEigvals  = get_attn_eigvals_per_head_topk(attn_diags, top_k)
  LapEigvals   = get_laplacian_eigvals_per_head_topk(lap_diags, top_k)   (headline)

All diagonals have shape [#layers, #heads, #seq] per example.
"""
from __future__ import annotations

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Diagonal extraction from raw attention                                       #
# (from hallucinations/features/attention_weights.py)                          #
# --------------------------------------------------------------------------- #
def attention_diagonal(item_attn: list[Tensor]) -> Tensor:
    """Self-attention diagonal for a single example.
    Input  item_attn: list over #layers of [#heads, seq, seq]
    Output: [#layers, #heads, seq]
    """
    return torch.stack(
        [torch.diagonal(layer_attn, dim1=1, dim2=2) for layer_attn in item_attn]
    )


def laplacian_diagonal_from_attn(
    item_attn: list[Tensor],
    vertical_edges: bool = False,
    vertical_edge_weight: float | None = None,
) -> Tensor:
    """Laplacian diagonal (L = D - A, weighted out-degree minus self-loop) per example.
    Input  item_attn: list over #layers of [#heads, seq, seq]
    Output: [#layers, #heads, seq]
    """
    device = item_attn[0].device
    if vertical_edges:
        assert vertical_edge_weight is not None

    fst_layer_attn = item_attn[0]
    fst_nom = fst_layer_attn.sum(dim=1)
    fst_denom = torch.arange(1, fst_layer_attn.size(1) + 1, device=device).flip(dims=[0])
    fst_weighted_degree = fst_nom / fst_denom
    fst_lap = fst_weighted_degree - torch.diagonal(fst_layer_attn, offset=0, dim1=1, dim2=2)

    per_layer_laplacian_diags = [fst_lap]
    for layer_attn in item_attn[1:]:
        if vertical_edges:
            assert vertical_edge_weight is not None
            nom = layer_attn.sum(dim=1) + vertical_edge_weight
            denom = torch.arange(1, layer_attn.size(1) + 1, device=device).flip(dims=[0]) + 1
        else:
            nom = layer_attn.sum(dim=1)
            denom = torch.arange(1, layer_attn.size(1) + 1, device=device).flip(dims=[0])

        layer_weighted_degree = nom / denom
        layer_lap_diag = layer_weighted_degree - torch.diagonal(
            layer_attn, offset=0, dim1=1, dim2=2
        )
        per_layer_laplacian_diags.append(layer_lap_diag)

    return torch.stack(per_layer_laplacian_diags)


# --------------------------------------------------------------------------- #
# Feature builders over a list of per-example diagonals                        #
# (from hallucinations/features/attn_feats.py)                                 #
# --------------------------------------------------------------------------- #
def get_attn_log_det(attn_diags: list[Tensor], layer_idx: int | None = None) -> Tensor:
    """AttnLogDet / LLMCheck:  mean(log(diag(A))) per head.
    Returns [#examples, (#layers*#heads)] if layer_idx is None else [#examples, #heads].
    """
    # clamp_min guards against fp16-underflowed zeros producing log(0)=-inf;
    # genuine softmax self-attention is always > 0, so this is numeric hygiene
    # only, not a change to the method.
    if layer_idx is None:
        return torch.stack([a.clamp_min(1e-12).log().mean(dim=-1).flatten() for a in attn_diags])
    return torch.stack([a[layer_idx].clamp_min(1e-12).log().mean(dim=-1) for a in attn_diags])


def get_attn_eigvals_per_head_topk(
    attn_diags: list[Tensor], top_k: int, layer_idx: int | None = None
) -> Tensor:
    """AttnEigvals: top-k attention-diagonal values per head.
    Returns [#examples, (#layers*#heads*top_k)] if layer_idx is None
            else [#examples, (#heads*top_k)].
    """
    if layer_idx is None:
        return torch.stack(
            [e.sort(dim=-1, descending=True).values[:, :, :top_k].flatten() for e in attn_diags]
        )
    return torch.stack(
        [e.sort(dim=-1, descending=True).values[layer_idx, :, :top_k].flatten() for e in attn_diags]
    )


def get_laplacian_eigvals_per_head_topk(
    laplacian_diags: list[Tensor], top_k: int, layer_idx: int | None = None
) -> Tensor:
    """LapEigvals (headline): top-k Laplacian-diagonal values per head.
    Returns [#examples, (#layers*#heads*top_k)] if layer_idx is None
            else [#examples, (#heads*top_k)].
    """
    if layer_idx is None:
        return torch.stack(
            [e.sort(dim=-1, descending=True).values[:, :, :top_k].flatten() for e in laplacian_diags]
        )
    return torch.stack(
        [e.sort(dim=-1, descending=True).values[layer_idx, :, :top_k].flatten() for e in laplacian_diags]
    )