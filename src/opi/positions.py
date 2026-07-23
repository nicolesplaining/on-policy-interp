"""Token-position finding for the couplet planning task.

The mechanistic analyses reference positions *relative to the second newline*
(the ``i=0`` planning site), following the look-ahead convention:

    i < 0 : tokens inside the first line (i=-1 is one token before the newline;
            for Gemma the rhyme word is typically at i=-2 because of the comma)
    i = 0 : the newline token ending the first line
    i > 0 : generated tokens of the second line

All helpers use ``return_offsets_mapping`` so they work identically across the
Gemma / Qwen / Llama tokenizers.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def second_newline_token(tokenizer, prompt: str) -> int:
    """Index of the token covering the second ``\\n`` in ``prompt`` (the i=0 site).

    The canonical prompt ``"A rhyming couplet:\\n<line>\\n"`` has its first
    newline after the header and its second newline after the first line.
    """
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    newline_chars = [i for i, ch in enumerate(prompt) if ch == "\n"]
    if len(newline_chars) < 2:
        raise ValueError(f"Need >=2 newlines in prompt, found {len(newline_chars)}.")
    second_nl_char = newline_chars[1]
    tok = next((i for i, (s, e) in enumerate(offsets) if s <= second_nl_char < e), None)
    if tok is None:
        raise ValueError("Could not find token covering the second newline.")
    return tok


def position_from_offset(tokenizer, prompt: str, offset: int) -> Tuple[int, str]:
    """Absolute token index and decoded string for ``i = offset`` (rel. to i=0).

    Mirrors ``patch_all_layers_unified.py:find_patch_pos``.
    """
    enc = tokenizer(prompt, add_special_tokens=True)
    token_ids = enc["input_ids"]
    base = second_newline_token(tokenizer, prompt)
    pos = base + offset
    if pos < 0 or pos >= len(token_ids):
        raise ValueError(
            f"offset={offset} -> out-of-bounds pos={pos} (len={len(token_ids)})"
        )
    return pos, tokenizer.decode([token_ids[pos]])


def rhyme_word_position(tokenizer, prompt: str) -> Tuple[int, str]:
    """Token index of the first-line rhyme word (last alphabetic token before i=0).

    Scans backward from the newline over the offset mapping, skipping the comma
    / punctuation tokens, and returns the first token whose decoded text
    contains a letter. This is more robust than a fixed ``i=-2`` offset across
    tokenizers.
    """
    enc = tokenizer(prompt, add_special_tokens=True)
    token_ids = enc["input_ids"]
    base = second_newline_token(tokenizer, prompt)
    for pos in range(base - 1, -1, -1):
        text = tokenizer.decode([token_ids[pos]])
        if any(ch.isalpha() for ch in text):
            return pos, text
    raise ValueError("No alphabetic token found before the second newline.")


def standard_probe_offsets() -> List[int]:
    """Relative positions probed / patched by default.

    Covers the rhyme word region (negative), the newline (0), and the first few
    generated tokens (positive), matching Section 13.2 / 14.2 of the outline.
    """
    return [-2, -1, 0, 1, 2, 3]


def label_for_offset(offset: int) -> str:
    """Stable string id for an offset, used in result keys/dirs."""
    if offset == 0:
        return "i_0"
    sign = "plus" if offset > 0 else "minus"
    return f"i_{sign}{abs(offset)}"
