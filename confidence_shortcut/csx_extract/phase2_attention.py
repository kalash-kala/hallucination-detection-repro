"""Phase 2: attention/Laplacian diagonals, the sink feature, and attn_logdet.

Runs under eager attention, because sdpa never materialises the [S,S] matrix
these features are built from. That makes this the expensive pass: cost grows
with S^2, and Pixtral's median prompt is 1,143 tokens against gemma-3's ~293.

**Reduction happens inside the loop.** A full [L,H,S] diagonal set for one vqav2
pair is ~60 GB; the top-50 per head is ~313 KB per row per segment. Nothing is
ever written at full width.

Two things here are load-bearing and easy to get quietly wrong:

**Diagonals are computed on the FULL sequence, then masked per segment.** The
Laplacian diagonal divides by a position-dependent denominator (`arange(1, S+1)`
flipped -- lapeigvals_features.py), so it is not translation-invariant: computing
it on a sliced sub-sequence gives different numbers from computing it whole and
then selecting. Segment selection is therefore a mask over the full-length
diagonal, never a re-run on a slice.

**`attn_logdet` is stored separately and always.** It is a mean of log(diag) over
ALL positions in the segment, not a top-k statistic, so it cannot be recovered
from the top-k arrays. Storing only top-k would leave the `attnlogdet` family
silently wrong rather than visibly missing -- which is why the store contract
makes it a required key.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from csx_common import registry
from csx_common.store_schema import DIAG_TOPK_KEYS

from . import config, models, spans, writer


# ── the vendored feature math (lapeigvals_features.py, exp3_feature.py) ──────

def attention_diagonal(attns) -> torch.Tensor:
    """[L, H, S] self-attention diagonal for one example."""
    return torch.stack([torch.diagonal(a[0], dim1=1, dim2=2) for a in attns])


def laplacian_diagonal(attns) -> torch.Tensor:
    """[L, H, S] Laplacian diagonal, L = D - A (weighted out-degree minus loop).

    `vertical_edges=False`, the variant the published baseline uses. The
    denominator counts how many query positions can attend to each key under the
    causal mask, which is why this depends on the full sequence length.
    """
    out = []
    for a in attns:
        layer = a[0]                                  # [H, S, S]
        S = layer.size(1)
        denom = torch.arange(1, S + 1, device=layer.device).flip(dims=[0])
        deg = layer.sum(dim=1) / denom
        out.append(deg - torch.diagonal(layer, offset=0, dim1=1, dim2=2))
    return torch.stack(out)


class ValueNormCapture:
    """Captures per-layer v_proj outputs during a forward pass.

    Deliberately a context manager around the SAME forward that produces the
    attention weights, not a second pass. Phase 2 is the expensive stage and its
    cost is quadratic in sequence length; running the model twice per row would
    double the most expensive thing in the programme to collect something the
    first pass already computes.

    GQA: v_proj emits n_kv_heads*head_dim, so norms are per KV head. Query head h
    reads KV head h // (n_q // n_kv); that broadcast happens at feature-build
    time, so what is stored stays [L, n_kv, S].
    """

    def __init__(self, model, n_layers: int):
        self.model = model
        self.n_layers = n_layers
        self.n_kv = _n_kv_from(model)
        self._grabbed: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self):
        def hook(idx):
            def fn(_m, _i, out):
                self._grabbed[idx] = out.detach()
            return fn

        for i, layer in enumerate(_decoder_layers(self.model)):
            self._handles.append(
                layer.self_attn.v_proj.register_forward_hook(hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def norms(self) -> torch.Tensor:
        """[L, n_kv, S]."""
        missing = [i for i in range(self.n_layers) if i not in self._grabbed]
        if missing:
            raise RuntimeError(
                f"v_proj hook fired for {len(self._grabbed)} of {self.n_layers} "
                f"layers (missing {missing[:3]}); the decoder layer path is wrong "
                f"for {type(self.model).__name__}")
        per_layer = []
        for i in range(self.n_layers):
            v = self._grabbed[i][0]                     # [S, n_kv*head_dim]
            v = v.view(v.shape[0], self.n_kv, -1)
            per_layer.append(v.float().norm(dim=-1).T)  # [n_kv, S]
        self._grabbed.clear()
        return torch.stack(per_layer)


def _decoder_layers(model):
    for attr in ("model.language_model.layers", "model.model.layers",
                 "model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj):
                return obj
        except AttributeError:
            continue
    raise RuntimeError(
        f"could not locate the decoder layer list on {type(model).__name__}; "
        f"add its attribute path to _decoder_layers")


def _n_kv_from(model) -> int:
    cfg = model.config
    tc = getattr(cfg, "text_config", cfg)
    return int(getattr(tc, "num_key_value_heads"))


# ── the pass ─────────────────────────────────────────────────────────────────

def _reduce(attn_d: np.ndarray, lap_d: np.ndarray, vnorm: np.ndarray,
            mask: np.ndarray, kv_of_head: np.ndarray, top_k: int,
            sink_k: int) -> dict[str, np.ndarray]:
    """Full-length diagonals + a segment mask -> the stored top-k arrays.

    `attn_logdet` is a mean over every masked position; the rest are top-k.
    """
    sel = np.flatnonzero(mask)
    A = attn_d[:, :, sel]                              # [L,H,s]
    P = lap_d[:, :, sel]
    D = A + P                                          # sink score
    s = A.shape[-1]

    def topk(X: np.ndarray, k: int) -> np.ndarray:
        kk = min(k, s)
        part = np.sort(X, axis=-1)[:, :, ::-1][:, :, :kk]
        if kk < k:  # pad short segments by edge, as the published builder does
            part = np.pad(part, ((0, 0), (0, 0), (0, k - kk)), mode="edge")
        return part.astype(np.float16)

    # The sink feature ranks by sink score and multiplies by ||V|| at those same
    # ranks, so the two must be gathered with ONE index array, not sorted apart.
    kk = min(sink_k, s)
    idx = np.argsort(-D, axis=-1)[:, :, :kk]
    d_top = np.take_along_axis(D, idx, axis=-1)
    vn = vnorm[:, kv_of_head, :][:, :, sel]            # [L,H,s] kv -> q broadcast
    vn_top = np.take_along_axis(vn, idx, axis=-1)
    sink = d_top
    sink_vn = d_top * vn_top
    if kk < sink_k:
        pad = ((0, 0), (0, 0), (0, sink_k - kk))
        sink = np.pad(sink, pad, mode="edge")
        sink_vn = np.pad(sink_vn, pad, mode="edge")

    # attn/lap/sink are attention diagonals, bounded near 1, and top-k keeps the
    # LARGEST values, so neither fp16 overflow nor underflow can reach them.
    # sink x ||V|| is the exception: ||V|| carries the model's raw activation
    # scale, and that is exactly the quantity that overflowed fp16 in phase 1 on
    # gemma-3. So it is stored fp32 like attn_logdet -- unconditionally, because
    # a dtype that depends on the row would have to be decided before the buffer
    # in run() is allocated, and truncating into an fp16 buffer afterwards would
    # restore the very inf this avoids.
    return {
        "attn_topk": topk(A, top_k),
        "lap_topk": topk(P, top_k),
        "sink_topk": sink.astype(np.float16),
        "sink_vnorm_topk": sink_vn.astype(np.float32),
        # Mean over ALL masked positions -- not derivable from the arrays above.
        "attn_logdet": np.log(np.clip(A, 1e-12, None)).mean(axis=-1
                                                            ).astype(np.float32),
    }


def run(pair_key: str, *, limit: int | None = None, verbose: bool = True) -> dict:
    pair = registry.get(pair_key)
    rows = writer.read_rows(pair_key)
    if limit:
        rows = rows.head(limit).copy()
    if (rows["seq_len"] <= 0).any():
        raise RuntimeError(
            f"{pair_key}: rows have no spans; phase 1 must run before phase 2")

    loaded = models.load(pair_key, attn_implementation="eager")
    n_layers = loaded.n_layers
    n_q, n_kv = loaded.n_q_heads, loaded.n_kv_heads
    group = max(n_q // max(n_kv, 1), 1)
    kv_of_head = np.arange(n_q) // group

    segs = pair.segments
    n = len(rows)
    top_k, sink_k = config.EXTRACT_TOP_K, config.SINK_K
    # sink_vnorm_topk is fp32 for the reason given in _reduce; the other three
    # top-k arrays are bounded attention diagonals and stay fp16.
    buf = {s: {k: np.zeros((n, n_layers, n_q,
                            sink_k if "sink" in k else top_k),
                           dtype=(np.float32 if k == "sink_vnorm_topk"
                                  else np.float16))
               for k in DIAG_TOPK_KEYS} for s in segs}
    for s in segs:
        buf[s]["attn_logdet"] = np.zeros((n, n_layers, n_q), dtype=np.float32)

    capture = ValueNormCapture(loaded.model, n_layers)
    t0 = time.time()
    for r in range(n):
        row = rows.iloc[r]
        inputs, n_prompt, n_answer = models.build_inputs(loaded, row)
        if inputs is None:
            raise RuntimeError(
                f"{pair_key}: row {row['id']} lost its answer between phases")

        sp = spans.Spans(
            seq_len=int(row["seq_len"]), answer_start=int(row["answer_start"]),
            answer_end=int(row["answer_end"]), image_start=int(row["image_start"]),
            image_end=int(row["image_end"]),
        )
        # One forward pass yields both the attention weights and the value norms.
        with capture, torch.no_grad():
            out = loaded.model(**inputs, output_attentions=True, use_cache=False)
        attn_d = attention_diagonal(out.attentions).float().cpu().numpy()
        lap_d = laplacian_diagonal(out.attentions).float().cpu().numpy()
        vnorm = capture.norms().cpu().numpy()
        del out

        S = attn_d.shape[-1]
        if S != sp.seq_len:
            raise RuntimeError(
                f"{pair_key}: row {row['id']} produced {S} positions but phase 1 "
                f"recorded seq_len={sp.seq_len}; the two passes disagree about "
                f"the sequence, so their features would not align")

        for s in segs:
            red = _reduce(attn_d, lap_d, vnorm, sp.mask(s), kv_of_head,
                          top_k, sink_k)
            for k, v in red.items():
                buf[s][k][r] = v

        if verbose and (r + 1) % 100 == 0:
            rate = (r + 1) / (time.time() - t0)
            print(f"  [{pair_key}] phase2 {r + 1}/{n} rows  {rate:.2f}/s",
                  flush=True)

    for s in segs:
        writer.save_diag(pair_key, s, buf[s])

    return {
        "n_rows": int(n),
        "minutes": round((time.time() - t0) / 60, 1),
        "attn_implementation": "eager",
        "top_k": top_k,
        "sink_k": sink_k,
        "n_q_heads": int(n_q),
        "n_kv_heads": int(n_kv),
    }
