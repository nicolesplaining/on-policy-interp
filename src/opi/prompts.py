"""Shared couplet prompt format and prompt-pool construction.

Every training regime and evaluation uses the identical input structure so that
mechanistic token positions line up across conditions:

    A rhyming couplet:
    <first line>,\n

The trailing newline is the ``i=0`` planning-site position. Only the *targets*
and *training prefixes* differ between conditions; the first-line prompt
distribution is shared.

This module builds first lines by slotting a chosen rhyme word into neutral
carrier templates. Rhyme words are bucketed by CMU rhyme family so the pool can
be balanced and split into seen / held-out families.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from .rhyme import is_known, rhyme_family

PROMPT_PREFIX = "A rhyming couplet:\n"


def format_prompt(first_line: str) -> str:
    """Wrap a first line into the canonical prompt, ending in ``\\n`` (i=0)."""
    first_line = first_line.rstrip()
    if not first_line.endswith(","):
        first_line = first_line + ","
    return f"{PROMPT_PREFIX}{first_line}\n"


# First-line carriers. ``{w}`` is the rhyme word (last word before the comma).
# Kept deliberately generic so the same carrier works across many rhyme words;
# grammaticality is checked at build time by a light filter, not guaranteed.
FIRST_LINE_TEMPLATES: List[str] = [
    "She felt a sudden sense of {w}",
    "He walked alone toward the {w}",
    "They whispered softly of the {w}",
    "The morning broke upon the {w}",
    "I never understood the {w}",
    "We wandered far to find the {w}",
    "A quiet voice recalled the {w}",
    "The old man spoke about the {w}",
    "Beneath the trees she saw the {w}",
    "Through winter nights they kept the {w}",
    "Nobody warned him of the {w}",
    "The letter told a tale of {w}",
    "At dawn the city woke to {w}",
    "Her steady hands still held the {w}",
    "And so the story turned to {w}",
]

# Seed rhyme words spanning many families; each is expanded via CMU rhymes.
# Chosen for unambiguous, common pronunciations.
SEED_RHYME_WORDS: List[str] = [
    "fright", "rest", "doom", "bliss", "grief", "dark", "night", "day",
    "song", "rain", "gold", "sea", "fire", "stone", "wind", "star",
    "dream", "heart", "time", "snow", "flight", "cold", "green", "shore",
    "bell", "moon", "road", "tree", "sky", "hand", "door", "wave",
    "flame", "peace", "sound", "hill", "gate", "field", "cloud", "spring",
]


def _make_frequency_gate(min_zipf: float):
    """Return ``is_common(word) -> bool`` using wordfreq if available.

    Zipf frequency is a log10 scale: ~5+ very common, ~3.5 fairly common, ~2.5
    rare. When wordfreq is not installed the gate keeps every word (no-op) so the
    pipeline still runs, just with a noisier vocabulary.
    """
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return lambda w: True
    return lambda w: zipf_frequency(w, "en") >= min_zipf


def build_rhyme_word_bank(
    max_words_per_family: int = 12,
    seed_words: Optional[Sequence[str]] = None,
    min_zipf: float = 3.3,
) -> Dict[str, List[str]]:
    """Group common CMU rhyme words into ``{family: [words...]}`` buckets.

    Each seed word contributes itself plus its CMU rhymes (same rhyming part).
    Only single-word, alphabetic, in-dictionary words that clear the ``min_zipf``
    frequency gate are kept, so first lines read naturally.
    """
    import pronouncing

    is_common = _make_frequency_gate(min_zipf)
    seed_words = list(seed_words or SEED_RHYME_WORDS)
    bank: Dict[str, List[str]] = {}
    for seed in seed_words:
        fam = rhyme_family(seed)
        if fam is None:
            continue
        candidates = [seed] + pronouncing.rhymes(seed)
        bucket: List[str] = bank.setdefault(fam, [])
        seen = set(bucket)
        for w in candidates:
            w = w.lower()
            if not w.isalpha() or w in seen or not is_known(w):
                continue
            if not is_common(w):
                continue
            # keep the bucket to a single canonical family
            if rhyme_family(w) != fam:
                continue
            bucket.append(w)
            seen.add(w)
            if len(bucket) >= max_words_per_family:
                break
    # drop degenerate families with <2 words (cannot form varied prompts)
    return {f: ws for f, ws in bank.items() if len(ws) >= 2}


def _plausible_first_line(template: str, word: str) -> bool:
    """Light grammatical filter to drop obviously broken slottings."""
    # Avoid double article ("the the"), and require the word to differ from the
    # template's trailing determiner context in an obvious way.
    line = template.format(w=word)
    lowered = " " + line.lower() + " "
    return " the the " not in lowered and " a a " not in lowered


def build_prompt_pool(
    n_prompts: int,
    families: Sequence[str],
    bank: Dict[str, List[str]],
    seed: int = 0,
    balance: bool = True,
) -> List[Dict]:
    """Build ``n_prompts`` couplet prompts drawn from ``families``.

    Returns a list of records::

        {"id", "prompt", "first_line", "rhyme_word", "rhyme_family", "template"}

    When ``balance`` is set, families are sampled round-robin so common families
    do not dominate; otherwise families are sampled uniformly at random.
    """
    rng = random.Random(seed)
    families = [f for f in families if f in bank]
    if not families:
        raise ValueError("No usable rhyme families provided.")

    records: List[Dict] = []
    fam_cycle: List[str] = []

    def next_family() -> str:
        nonlocal fam_cycle
        if balance:
            if not fam_cycle:
                fam_cycle = list(families)
                rng.shuffle(fam_cycle)
            return fam_cycle.pop()
        return rng.choice(families)

    used: set = set()
    attempts = 0
    max_attempts = n_prompts * 50
    while len(records) < n_prompts and attempts < max_attempts:
        attempts += 1
        fam = next_family()
        word = rng.choice(bank[fam])
        template = rng.choice(FIRST_LINE_TEMPLATES)
        if not _plausible_first_line(template, word):
            continue
        first_line = template.format(w=word)
        prompt = format_prompt(first_line)
        key = (template, word)
        if key in used:
            continue
        used.add(key)
        records.append({
            "id": len(records),
            "prompt": prompt,
            "first_line": first_line + ",",
            "rhyme_word": word,
            "rhyme_family": fam,
            "template": template,
        })
    return records


def split_families(
    bank: Dict[str, List[str]],
    heldout_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Partition families into (train/seen, held-out) sets."""
    fams = sorted(bank.keys())
    rng = random.Random(seed)
    rng.shuffle(fams)
    n_held = max(1, int(round(len(fams) * heldout_fraction)))
    heldout = sorted(fams[:n_held])
    seen = sorted(fams[n_held:])
    return seen, heldout
