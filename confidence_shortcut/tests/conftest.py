"""Shared fixtures.

`make_entry` lives here rather than in one test module because two modules need
it and `tests/` is not a package -- importing across test files only works by
accident of `sys.path`, and breaks the moment pytest is invoked from elsewhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from csx_probe.store.read import Entry


class _Meta:
    """The slice of EntryMeta the arm and rotation code actually reads."""
    raw = {"model": {"key": "qwen25vl", "n_q_heads": 28, "layers": 28},
           "dataset": "advqa", "prompt_template": "chat"}
    modality = "vlm"
    segments = ("all",)
    layers = 28

    def phase_done(self, _phase):
        return True

    def available_families(self):
        from csx_probe import config
        return config.FAMILIES


def make_entry(n_per_cat=(400, 400, 400, 400), *, seed=0, n_strata=25,
               pair="qwen25vl_advqa") -> Entry:
    """Synthetic rows with the real band geometry.

    H rows sit strictly BELOW tau in entropy and L rows strictly above, because
    that is what the DSE threshold guarantees and several arm properties depend
    on it. Entropies come from a small discrete set so strata actually have
    multiple members -- with all-distinct floats every stratum is a singleton and
    matched2 would have nothing to match.
    """
    rng = np.random.default_rng(seed)
    cats: list[str] = []
    ent: list[float] = []
    for cat, n in zip(("IH", "CH", "IL", "CL"), n_per_cat):
        lo, hi = (0.0, 0.4) if cat in ("IH", "CH") else (0.6, 1.0)
        grid = np.linspace(lo, hi, n_strata)
        cats += [cat] * n
        ent += list(rng.choice(grid, n))
    rows = pd.DataFrame({
        "id": [f"train::{i}" for i in range(len(cats))],
        "row": np.arange(len(cats), dtype="int32"),
        "category": cats,
        "entropy": np.asarray(ent, dtype=float),
    })
    return Entry(pair=pair, meta=_Meta(), rows=rows)
