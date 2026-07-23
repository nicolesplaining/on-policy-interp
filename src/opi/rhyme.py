"""Rhyme checking, rhyme-family assignment, and word extraction.

Phonology comes from the CMU Pronouncing Dictionary via the `pronouncing`
library, matching the conventions used in the look-ahead planning-site study so
that mechanistic results are directly comparable.

A "rhyme family" is the CMU *rhyming part*: the stressed vowel and everything
after it (e.g. ``fright`` -> ``AY1 T``). Two words rhyme iff any of their
pronunciations share a rhyming part.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set

try:
    import pronouncing
    _HAS_PRONOUNCING = True
except ImportError:  # pragma: no cover - dependency guard
    _HAS_PRONOUNCING = False


# ---------------------------------------------------------------------------
# Rhyme families (CMU rhyming part)
# ---------------------------------------------------------------------------

def rhyming_parts(word: str) -> Set[str]:
    """All rhyming parts (rhyme families) for ``word``; empty if unknown."""
    if not _HAS_PRONOUNCING:
        return set()
    phones = pronouncing.phones_for_word(word.lower().strip())
    return {pronouncing.rhyming_part(p) for p in phones} - {""}


def rhyme_family(word: str) -> Optional[str]:
    """Canonical (first-pronunciation) rhyme family for ``word``.

    Used for balancing and bucketing prompts. Returns ``None`` when the word is
    absent from the CMU dictionary.
    """
    if not _HAS_PRONOUNCING:
        return None
    phones = pronouncing.phones_for_word(word.lower().strip())
    if not phones:
        return None
    rp = pronouncing.rhyming_part(phones[0])
    return rp or None


def do_rhyme(word1: str, word2: str) -> Optional[bool]:
    """True if the words rhyme, False if not, None if either is out-of-vocab.

    Mirrors ``poem/src/ablation/evaluate_rhyming.py:do_rhyme`` from look-ahead:
    identical words count as rhyming, and a word is "unknown" only when it has
    no CMU pronunciation.
    """
    w1, w2 = word1.lower().strip(), word2.lower().strip()
    if w1 == w2:
        return True
    if not _HAS_PRONOUNCING:
        return None
    rp1, rp2 = rhyming_parts(w1), rhyming_parts(w2)
    if not rp1 or not rp2:
        return None
    return bool(rp1 & rp2)


def is_known(word: str) -> bool:
    """Whether ``word`` has at least one CMU pronunciation."""
    return bool(rhyming_parts(word))


# ---------------------------------------------------------------------------
# Word extraction from prompts / generations
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[.!?]")
# Matches special tokens like <end_of_turn> (Gemma) or <|im_end|> (Qwen).
_SPECIAL_TOKEN_RE = re.compile(r"<[^>]+>")


def last_alpha_word(text: str) -> Optional[str]:
    """Return the last run of letters in ``text``, lower-cased, else None."""
    words = re.findall(r"[a-zA-Z]+", text)
    return words[-1].lower() if words else None


def extract_first_line_word(prompt_text: str) -> Optional[str]:
    """Last word of the first couplet line.

    Prompt format: ``"A rhyming couplet:\\n<First Line>\\n"``.
    """
    parts = prompt_text.rstrip("\n").split("\n")
    if len(parts) < 2:
        return None
    return last_alpha_word(parts[-1])


def extract_second_line_word_with_newline(continuation: str) -> Optional[str]:
    """Rhyme word from a continuation that ends at ``\\n`` or an end-of-turn token."""
    end = len(continuation)
    nl_pos = continuation.find("\n")
    if nl_pos >= 0:
        end = min(end, nl_pos)
    st = _SPECIAL_TOKEN_RE.search(continuation)
    if st:
        end = min(end, st.start())
    return last_alpha_word(continuation[:end])


def extract_second_line_word_without_newline(continuation: str) -> Optional[str]:
    """Rhyme word when ``\\n`` is suppressed during generation.

    Terminates at the first sentence punctuation or special token.
    """
    end = len(continuation)
    m = _PUNCT_RE.search(continuation)
    if m:
        end = min(end, m.start())
    st = _SPECIAL_TOKEN_RE.search(continuation)
    if st:
        end = min(end, st.start())
    return last_alpha_word(continuation[:end])


def _last_word_stripped(text: str) -> str:
    """Last alphabetic token, stripping surrounding punctuation (patching style)."""
    for w in reversed(text.split()):
        cleaned = w.strip(".,!?\"'—;: ")
        if cleaned.isalpha():
            return cleaned.lower()
    return ""


def word_before_nth_newline(text: str, n: int) -> str:
    """Last word on the line terminated by the ``n``-th newline (1-indexed)."""
    if n <= 0:
        return ""
    nls = [i for i, ch in enumerate(text) if ch == "\n"]
    if len(nls) < n:
        return ""
    end = nls[n - 1]
    start = nls[n - 2] + 1 if n >= 2 else 0
    return _last_word_stripped(text[start:end])


def extract_rhyme_word(full_text: str, prompt: str) -> str:
    """Rhyme word of the generated second line, given the full (prompt+gen) text.

    Mirrors ``patch_all_layers_unified.py:extract_rhyme_word``: the second line is
    the one ending at the ``(prompt newlines + 1)``-th newline.
    """
    target_newline_index = prompt.count("\n") + 1
    rhyme_word = word_before_nth_newline(full_text, target_newline_index)
    if rhyme_word:
        return rhyme_word
    if full_text.startswith(prompt):
        return _last_word_stripped(full_text[len(prompt):])
    return _last_word_stripped(full_text)


# ---------------------------------------------------------------------------
# Aggregate rhyme scoring over completions
# ---------------------------------------------------------------------------

def rhyme_rate(completions: List[str], prompt: str, rhyme_word: str) -> float:
    """Fraction of completions whose second-line word rhymes with ``rhyme_word``.

    Completions are full-text (prompt + generation), as returned by
    ``tokenizer.decode`` on ``model.generate`` output.
    """
    if not completions:
        return 0.0
    hits = sum(
        1 for c in completions
        if do_rhyme(extract_rhyme_word(c, prompt), rhyme_word) is True
    )
    return hits / len(completions)
