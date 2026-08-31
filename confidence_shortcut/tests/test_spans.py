"""Spans: the module where a mistake is silent.

Image tokens are 87-95% of every VLM prompt, and the whole VLM design rests on
splitting `image` from `text` as two confidence channels. If the boundary is
wrong the two channels contaminate each other and the result looks like a
finding -- "the visual and linguistic channels agree closely" -- rather than a
bug. So the span logic is tested against real stored prompts, not just synthetic
ones.
"""

from __future__ import annotations

import json

import pytest

from csx_common import paths, registry
from csx_extract import spans

QWEN_IMG = 151655
GEMMA_IMG = 262144
PIX_IMG, PIX_BREAK = 10, 12


def _pair(key):
    return registry.get(key)


# ── the contiguity rule, on synthetic prompts ────────────────────────────────

def test_finds_a_simple_span():
    toks = [1, 2, QWEN_IMG, QWEN_IMG, QWEN_IMG, 3, 4]
    assert spans.image_span_from_tokens(toks, [QWEN_IMG]) == (2, 5)


def test_no_image_tokens_is_an_error():
    with pytest.raises(spans.SpanError, match="no image tokens"):
        spans.image_span_from_tokens([1, 2, 3], [QWEN_IMG])


def test_non_contiguous_span_is_refused_with_the_fix_named():
    """This is exactly the Pixtral failure: [IMG] alone leaves IMG_BREAK holes.
    Silently taking first..last would swallow foreign positions into the image
    segment, so it must fail and say what to add."""
    toks = [1, PIX_IMG, PIX_IMG, PIX_BREAK, PIX_IMG, PIX_IMG, 9]
    with pytest.raises(spans.SpanError, match="IMG_BREAK"):
        spans.image_span_from_tokens(toks, [PIX_IMG])


def test_including_img_break_makes_it_contiguous():
    toks = [1, PIX_IMG, PIX_IMG, PIX_BREAK, PIX_IMG, PIX_IMG, 9]
    assert spans.image_span_from_tokens(toks, [PIX_IMG, PIX_BREAK]) == (1, 6)


# ── the Spans invariants ─────────────────────────────────────────────────────

def test_masks_partition_the_sequence():
    s = spans.Spans(seq_len=10, answer_start=8, answer_end=10,
                    image_start=1, image_end=6)
    img, txt, alls = s.mask("image"), s.mask("text"), s.mask("all")
    assert img.sum() == 5 and txt.sum() == 5
    assert (img | txt).all() and not (img & txt).any()
    assert alls.all()


def test_text_segment_includes_the_answer():
    """`text` is the complement of the image block, so it spans the prompt text
    on both sides AND the answer -- the answer is text the model produced."""
    s = spans.Spans(seq_len=10, answer_start=8, answer_end=10,
                    image_start=1, image_end=6)
    assert s.mask("text")[8:10].all()


def test_image_overlapping_the_answer_is_refused():
    with pytest.raises(spans.SpanError, match="overlaps the answer"):
        spans.Spans(seq_len=10, answer_start=4, answer_end=10,
                    image_start=1, image_end=6).validate()


def test_text_pair_cannot_ask_for_an_image_segment():
    s = spans.Spans(seq_len=5, answer_start=3, answer_end=5,
                    image_start=-1, image_end=-1)
    with pytest.raises(spans.SpanError, match="text pair"):
        s.mask("image")


def test_half_set_image_span_is_refused():
    with pytest.raises(spans.SpanError, match="half-set"):
        spans.Spans(seq_len=5, answer_start=3, answer_end=5,
                    image_start=1, image_end=-1).validate()


def test_build_places_the_answer_at_the_end():
    p = _pair("llama_sciq")
    s = spans.build(p, [1, 2, 3, 4], n_answer_tokens=3)
    assert (s.answer_start, s.answer_end, s.seq_len) == (4, 7, 7)
    assert (s.image_start, s.image_end) == (-1, -1)


def test_build_on_a_vlm_pair_finds_the_image():
    p = _pair("qwen25vl_advqa")
    toks = [1, QWEN_IMG, QWEN_IMG, QWEN_IMG, 5, 6]
    s = spans.build(p, toks, n_answer_tokens=2)
    assert (s.image_start, s.image_end) == (1, 4)
    assert (s.answer_start, s.answer_end) == (6, 8)


def test_cross_check_catches_a_shifted_processor_span():
    """The guard against gemma-3's dropped token_type_ids and Qwen's mis-packed
    batch: two derivations of the same span that must agree."""
    p = _pair("qwen25vl_advqa")
    stored = [1, QWEN_IMG, QWEN_IMG, QWEN_IMG, 5, 6]
    s = spans.build(p, stored, n_answer_tokens=2)
    live = [1, 1, QWEN_IMG, QWEN_IMG, QWEN_IMG, 6]      # shifted by one
    with pytest.raises(spans.SpanError, match="image span disagreement"):
        spans.cross_check(p, s, live)


def test_cross_check_passes_when_they_agree():
    p = _pair("qwen25vl_advqa")
    toks = [1, QWEN_IMG, QWEN_IMG, QWEN_IMG, 5, 6]
    spans.cross_check(p, spans.build(p, toks, 2), toks)


# ── against the real stored prompts ──────────────────────────────────────────

def _generations(pair_key, limit):
    p = registry.get(pair_key)
    path = p.generations_path
    if path is None or not path.exists():
        pytest.skip(f"{pair_key}: generations folder not present")
    out = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            rec = json.loads(line)
            out.append(next(iter(rec.values())))
    return p, out


@pytest.mark.parametrize("pair_key", [
    "qwen25vl_advqa", "gemma3_12b_advqa", "pixtral12b_advqa",
])
def test_declared_image_tokens_form_a_contiguous_span_on_real_prompts(pair_key):
    """The registry's image_token_ids must actually work on stored data. This is
    what caught Pixtral's IMG_BREAK: with [IMG] alone the span is 10-40 blocks."""
    p, recs = _generations(pair_key, 200)
    for rec in recs:
        toks = rec["prompt_token_ids"]
        start, end = spans.image_span_from_tokens(toks, p.model.image_token_ids)
        assert 0 <= start < end <= len(toks)


@pytest.mark.parametrize("pair_key,expected", [
    ("gemma3_12b_advqa", 256),   # gemma-3 always emits exactly 256 soft tokens
])
def test_fixed_width_image_blocks(pair_key, expected):
    p, recs = _generations(pair_key, 50)
    for rec in recs:
        s, e = spans.image_span_from_tokens(rec["prompt_token_ids"],
                                            p.model.image_token_ids)
        assert e - s == expected


def test_image_precedes_the_answer_on_real_prompts():
    """The image block sits inside the prompt, and the answer is appended after
    it, so the ordering invariant holds by construction -- asserted on real data
    because `validate()` relies on it."""
    p, recs = _generations("pixtral12b_advqa", 100)
    for rec in recs:
        toks = rec["prompt_token_ids"]
        s = spans.build(p, toks, n_answer_tokens=3)
        assert s.image_end <= s.answer_start
