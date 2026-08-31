"""Where the image is, where the answer is, and which positions each segment pools.

Spans decide which token positions every feature is averaged or ranked over, so
an error here corrupts all seven families at once while producing perfectly
plausible numbers. That is the failure mode this module exists to make
impossible, and it does it by never trusting a single source:

  * the image span is computed from the stored `prompt_token_ids` by masking the
    model's declared image token ids -- pure data, no processor involved;
  * when a live processor is available, `cross_check` recomputes the same span
    from its output and demands the two agree exactly.

Two independent derivations that must match is a much stronger guarantee than
either one alone, and it is what catches the three known silent corruptions:

  gemma-3   `token_type_ids` is the image marker, but the repo-wide
            `del inputs['token_type_ids']` branch (huggingface_models.py:424)
            keys on the model NAME, so on the VLM path it deletes exactly the
            array that says where the image is. Generation still works.
  Qwen2.5-VL `pixel_values` is a flattened concatenated patch sequence rather
            than batch-major, so `pixel_values[i]` is not row i's image.
  Pixtral   `[IMG_BREAK]` (id 12) is emitted at every patch-row boundary, so an
            `[IMG]`-only mask splits into 10-40 blocks and is not a span at all.
            Both ids together are exactly contiguous.

Segment definitions:
  all    every position in the teacher-forced sequence  (the text-pair convention)
  image  the image-token block
  text   everything else -- prompt text before and after the image, plus the answer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csx_common import registry


class SpanError(Exception):
    """A span could not be determined, or two derivations disagreed."""


@dataclass(frozen=True)
class Spans:
    seq_len: int
    answer_start: int
    answer_end: int
    image_start: int   # -1 for text pairs
    image_end: int     # -1 for text pairs

    def validate(self) -> None:
        if not 0 <= self.answer_start < self.answer_end <= self.seq_len:
            raise SpanError(
                f"answer span [{self.answer_start},{self.answer_end}) invalid for "
                f"seq_len={self.seq_len}")
        if self.image_start == -1 or self.image_end == -1:
            if (self.image_start, self.image_end) != (-1, -1):
                raise SpanError("image span half-set; use -1 for both or neither")
            return
        if not 0 <= self.image_start < self.image_end <= self.seq_len:
            raise SpanError(
                f"image span [{self.image_start},{self.image_end}) invalid for "
                f"seq_len={self.seq_len}")
        if self.image_end > self.answer_start:
            raise SpanError(
                f"image span [{self.image_start},{self.image_end}) overlaps the "
                f"answer at {self.answer_start}; image tokens must precede the "
                f"answer")

    def mask(self, segment: str) -> np.ndarray:
        """Boolean [seq_len] mask of the positions this segment pools over."""
        if segment == "all":
            return np.ones(self.seq_len, dtype=bool)
        if self.image_start == -1:
            raise SpanError(f"segment {segment!r} requested on a text pair")
        img = np.zeros(self.seq_len, dtype=bool)
        img[self.image_start:self.image_end] = True
        if segment == "image":
            return img
        if segment == "text":
            return ~img
        raise SpanError(f"unknown segment {segment!r}")


def image_span_from_tokens(token_ids, image_token_ids) -> tuple[int, int]:
    """[start, end) of the image block, from token ids alone.

    Requires the block to be contiguous. Pixtral only satisfies that when
    [IMG_BREAK] is included alongside [IMG], which is why the id set is declared
    per model in pairs.yaml rather than assumed to be a single token.
    """
    want = set(int(t) for t in image_token_ids)
    if not want:
        raise SpanError("no image token ids declared for this model")
    idx = [i for i, t in enumerate(token_ids) if int(t) in want]
    if not idx:
        raise SpanError(
            f"no image tokens {sorted(want)} found in a {len(token_ids)}-token "
            f"prompt; the wrong image token ids are declared, or this row has no "
            f"image")
    start, end = idx[0], idx[-1] + 1
    if end - start != len(idx):
        holes = end - start - len(idx)
        raise SpanError(
            f"image tokens are not contiguous: {len(idx)} tokens spread over "
            f"[{start},{end}) with {holes} foreign position(s) inside. Add the "
            f"missing structural token id (e.g. Pixtral's [IMG_BREAK]=12) to "
            f"image_token_ids in pairs.yaml.")
    return start, end


def build(pair: registry.Pair, prompt_token_ids, n_answer_tokens: int) -> Spans:
    """Spans for one row.

    The prompt is teacher-forced ahead of the answer, so the answer occupies the
    final `n_answer_tokens` positions and the boundary is exact rather than
    recovered by re-tokenising.
    """
    n_prompt = len(prompt_token_ids)
    if n_answer_tokens < 1:
        raise SpanError(f"{pair.key}: answer has no tokens")
    seq_len = n_prompt + n_answer_tokens
    spans = Spans(
        seq_len=seq_len,
        answer_start=n_prompt,
        answer_end=seq_len,
        image_start=-1,
        image_end=-1,
    )
    if pair.is_vlm:
        s, e = image_span_from_tokens(prompt_token_ids, pair.model.image_token_ids)
        spans = Spans(seq_len=seq_len, answer_start=n_prompt, answer_end=seq_len,
                      image_start=s, image_end=e)
    spans.validate()
    return spans


def cross_check(pair: registry.Pair, spans: Spans, processor_input_ids) -> None:
    """Recompute the image span from a live processor's output and demand a match.

    This is the guard that turns all three silent corruptions into a loud
    failure: if gemma-3's token_type_ids were dropped, or a Qwen batch were
    mis-packed, the processor's input_ids would not carry the same image block as
    the stored prompt, and that shows up here rather than as a slightly
    disappointing AUROC three days later.
    """
    if not pair.is_vlm:
        return
    got = image_span_from_tokens(processor_input_ids, pair.model.image_token_ids)
    want = (spans.image_start, spans.image_end)
    if got != want:
        raise SpanError(
            f"{pair.key}: image span disagreement -- stored prompt_token_ids say "
            f"{want}, the live processor says {got}. The processor output has "
            f"been altered (a dropped token_type_ids, a mis-packed batch, or a "
            f"changed image resolution); the features would be pooled over the "
            f"wrong positions.")
