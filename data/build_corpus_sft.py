#!/usr/bin/env python3
"""Phase 2: build the fixed external poem corpus (``corpus_sft`` condition).

Per outline Section 7.1 the *first-line distribution is shared* across all
conditions; only the second-line targets differ. This builder reads the shared
train prompts and attaches a valid rhyming second line that is **not** generated
by the teacher (outline Section 7.4), so the corpus condition measures generic
static SFT rather than teacher imitation.

Two sources of second lines:

  * ``--source templated`` (default): a controlled, reproducible generator that
    slots a same-family rhyme word into varied carrier templates. Guarantees a
    valid rhyme and unambiguous pronunciation, and is distribution-matched to
    the shared first lines by construction.
  * ``--source hf --hf_dataset <name>``: mine rhyming couplets from a
    public-domain / existing poetry dataset on the Hugging Face Hub and reformat
    them. Requires that the dataset's couplets rhyme (checked with CMU) and are
    reformatted to the shared prompt structure.

Output: ``data/corpus/corpus_sft.jsonl`` with records
``{id, prompt, target, text, rhyme_word, second_word, rhyme_family}``.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opi import rhyme as R  # noqa: E402
from opi.prompts import format_prompt  # noqa: E402
from opi.io_utils import read_jsonl, write_jsonl  # noqa: E402


# Grammatical second-line carriers ending in the rhyme word ``{w}``. Kept varied
# in syntax so the corpus is not trivially uniform.
SECOND_LINE_TEMPLATES = [
    "and vanished slowly into {w}",
    "as though the world had turned to {w}",
    "she could not bear the endless {w}",
    "he dreamed of nothing but the {w}",
    "and left behind a quiet {w}",
    "beneath the pale and distant {w}",
    "that carried all her hopes of {w}",
    "yet still they longed to see the {w}",
    "until the dawn dissolved the {w}",
    "and no one spoke about the {w}",
    "while shadows gathered near the {w}",
    "a promise softer than the {w}",
    "and there they found a fragile {w}",
    "that no one else would ever {w}",
    "and so the night gave way to {w}",
]


def _pick_second_word(family: str, bank: dict, first_word: str, rng: random.Random):
    """A same-family rhyme word distinct from the first-line word."""
    choices = [w for w in bank.get(family, []) if w != first_word]
    if not choices:
        return None
    return rng.choice(choices)


def build_templated(train, bank, seed: int):
    rng = random.Random(seed)
    out = []
    for rec in train:
        fam = rec["rhyme_family"]
        second_word = _pick_second_word(fam, bank, rec["rhyme_word"], rng)
        if second_word is None:
            continue
        template = rng.choice(SECOND_LINE_TEMPLATES)
        second_line = template.format(w=second_word)
        # sanity: template's own trailing word must actually be the rhyme word
        if R.last_alpha_word(second_line) != second_word:
            continue
        if R.do_rhyme(rec["rhyme_word"], second_word) is not True:
            continue
        prompt = format_prompt(rec["first_line"])
        target = second_line + "\n"
        out.append({
            "id": len(out),
            "prompt": prompt,
            "target": target,
            "text": prompt + target,
            "rhyme_word": rec["rhyme_word"],
            "second_word": second_word,
            "rhyme_family": fam,
            "source": "templated",
        })
    return out


def build_from_hf(hf_dataset: str, split: str, text_field: str, limit: int):
    """Mine rhyming couplets from an existing poetry dataset (best-effort)."""
    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    out = []
    for row in ds:
        text = row.get(text_field) or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # scan consecutive line pairs for a clean rhyme
        for a, b in zip(lines, lines[1:]):
            wa, wb = R.last_alpha_word(a), R.last_alpha_word(b)
            if not wa or not wb:
                continue
            if R.do_rhyme(wa, wb) is not True:
                continue
            fam = R.rhyme_family(wa)
            if fam is None:
                continue
            first_line = a if a.endswith(",") else a.rstrip(".;:!?") + ","
            prompt = format_prompt(first_line)
            target = b.rstrip() + "\n"
            out.append({
                "id": len(out),
                "prompt": prompt,
                "target": target,
                "text": prompt + target,
                "rhyme_word": wa,
                "second_word": wb,
                "rhyme_family": fam,
                "source": f"hf:{hf_dataset}",
            })
            if len(out) >= limit:
                return out
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/prompt_pool/train.jsonl")
    ap.add_argument("--families", default="data/prompt_pool/families.json")
    ap.add_argument("--out", default="data/corpus/corpus_sft.jsonl")
    ap.add_argument("--source", choices=["templated", "hf"], default="templated")
    ap.add_argument("--hf_dataset", default=None)
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--hf_text_field", default="text")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "templated":
        import json
        train = read_jsonl(args.train)
        bank = json.load(open(args.families))["bank"]
        out = build_templated(train, bank, args.seed)
    else:
        if not args.hf_dataset:
            ap.error("--source hf requires --hf_dataset")
        out = build_from_hf(args.hf_dataset, args.hf_split, args.hf_text_field, args.limit)

    out = out[: args.limit]
    n = write_jsonl(args.out, out)
    print(f"Wrote {n} corpus couplets -> {args.out}")
    if out:
        print("Example:", repr(out[0]["text"]))


if __name__ == "__main__":
    main()
