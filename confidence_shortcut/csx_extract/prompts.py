"""Rebuilding the exact prompt each generation run used, and proving it matched.

`s_ext` is the log-probability of the stored greedy answer under the prompt we
build here, and the hidden states are read from that same sequence. So if the
prompt differs from the one the generation run used -- even in a way that reads
as equivalent -- every feature is measured against a context the model never saw.

Nothing about that failure is visible downstream. The pass completes, the arrays
have the right shapes, the AUROCs are finite and unremarkable. It was caught only
because a one-token greedy answer scored log P = -19 (P ~ 5e-9), which is absurd
for a greedy decode.

The fix is not to be careful about the wording. The generations folder stores
`prompt_token_ids` for every row, so the reconstruction can be checked against
ground truth exactly, per row, and that is what `check` does.
"""

from __future__ import annotations

from csx_common import registry


class PromptMismatch(Exception):
    """A rebuilt prompt does not match the one the generation run recorded."""


def text_for(pair: registry.Pair, question: str) -> str:
    """The user-turn text, before any chat template is applied."""
    tmpl = pair.dataset.prompt_text
    if not tmpl:
        raise PromptMismatch(
            f"{pair.key}: dataset {pair.dataset.key!r} declares no prompt_text; "
            f"extraction cannot guess the wording the run used")
    return tmpl.format(question=question)


def check(pair: registry.Pair, built: list[int], stored: list[int],
          *, row_id: str = "") -> None:
    """Demand an exact token-for-token match against the recorded prompt."""
    if list(built) == list(stored):
        return
    where = f" (row {row_id})" if row_id else ""
    if len(built) != len(stored):
        detail = f"length {len(built)} vs recorded {len(stored)}"
    else:
        i = next(i for i, (a, b) in enumerate(zip(built, stored)) if a != b)
        detail = f"first difference at position {i}: {built[i]} vs {stored[i]}"
    raise PromptMismatch(
        f"{pair.key}: rebuilt prompt does not match the generation run's "
        f"prompt_token_ids{where} -- {detail}. s_ext and every hidden state "
        f"would be computed against a prompt the model never saw. Fix "
        f"`prompt_text` for dataset {pair.dataset.key!r} in pairs.yaml.")
