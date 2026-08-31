"""Band-routed scoring: two confidence-band experts, a router, and a pooler.

The architecture the write-up argues for, against a single generalist probe:

    router(x) -> band b        which expert should score this row
    expert_b(x) -> f_b(x)      a probe fit only on band b's rows
    pool_b(f_b(x)) -> s(x)     the two experts' outputs put on one scale

Split three ways because the paper's three claims are separable and each has its
own failure mode:

  `experts.py`  specialisation -- does a band-local probe beat a global one?
  `router.py`   deployability -- can the band be predicted without the answer?
  `pooling.py`  commensurability -- are the two experts' outputs comparable?

`pooling.py` is the one that carries the argument: `z`, `platt`, `platt_prior`
and `proba` are the SAME 2-parameter affine-per-band family, differing only in
how the two numbers are chosen. Keeping them in one module is what makes that
claim checkable instead of rhetorical.
"""

from __future__ import annotations

from csx_probe.routing import best_split, experts, pooling, router

__all__ = ["best_split", "experts", "pooling", "router"]
