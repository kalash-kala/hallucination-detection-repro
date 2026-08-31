"""Backbone loading, dispatched on modality.

torch and transformers are imported lazily inside `load`, so the surrounding
package (rows.py, spans.py, subsample.py, verify.py) stays importable on a CPU
box with no torch installed -- which is what lets `csx-store verify` and row
building run anywhere.
"""

from __future__ import annotations

from csx_common import registry


def load(pair_key: str, **kw):
    """Load the backbone for a pair. Returns LoadedVLM or LoadedLM."""
    pair = registry.get(pair_key)
    if pair.is_vlm:
        from . import vlm
        return vlm.load(pair_key, **kw)
    from . import text
    return text.load(pair_key, **kw)


def build_inputs(loaded, row):
    """Teacher-forced inputs for one row of the row table.

    Returns `(inputs, n_prompt, n_answer)`; `inputs` is None when the greedy
    answer tokenises to nothing, which the callers skip.
    """
    if loaded.pair.is_vlm:
        from . import vlm
        return vlm.build_inputs(loaded, row["image_path"], row["question"],
                                row["answer"])
    from . import text
    return text.build_inputs(loaded, row["question"], row["answer"])
