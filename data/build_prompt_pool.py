#!/usr/bin/env python3
"""Phase 2: build the shared couplet prompt pool.

Produces the splits from outline Section 7.2, all drawn from the same first-line
distribution so every training regime and evaluation sees identical prompts:

    train                (default 10,000)  seen rhyme families
    val                  (default  1,000)  seen rhyme families
    test_id              (default  1,000)  seen families, unseen prompts
    test_heldout_family  (default  1,000)  entirely held-out families
    recovery             (default    500)  first line + malformed partial 2nd line

Rhyme families are split into seen / held-out so the held-out set measures
generalization to novel phonology (Section 7.3). Output is JSONL under
``data/prompt_pool/``.

Pure Python (no GPU / model). Deterministic given ``--seed``.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opi import prompts as P  # noqa: E402
from opi.io_utils import write_jsonl, write_json  # noqa: E402


# Fragments used to build imperfect partial second lines for recovery prompts:
# repetition, dangling dashes, semantic drift, awkward syntax (Section 7.6).
RECOVERY_FRAGMENTS = [
    "but every door she tried was—",
    "and then the the room began to",
    "yet nothing nothing seemed to",
    "so he turned around and around and",
    "while the cold wind kept on keeping on and",
    "then suddenly, without warning, the",
    "and all at once it felt so very",
    "but he could not, would not, ever",
]


def _make_recovery(record: dict, rng: random.Random) -> dict:
    """Attach a malformed partial second line to a base prompt record."""
    frag = rng.choice(RECOVERY_FRAGMENTS)
    prompt = record["prompt"] + frag
    return {
        "id": record["id"],
        "prompt": prompt,
        "first_line": record["first_line"],
        "rhyme_word": record["rhyme_word"],
        "rhyme_family": record["rhyme_family"],
        "partial_second_line": frag,
        "kind": "recovery",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="data/prompt_pool")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test_id", type=int, default=1000)
    ap.add_argument("--n_test_heldout", type=int, default=1000)
    ap.add_argument("--n_recovery", type=int, default=500)
    ap.add_argument("--heldout_fraction", type=float, default=0.2)
    ap.add_argument("--max_words_per_family", type=int, default=16)
    ap.add_argument("--min_zipf", type=float, default=3.3,
                    help="Frequency gate for rhyme words (higher = more common).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("Building rhyme-word bank...")
    bank = P.build_rhyme_word_bank(
        max_words_per_family=args.max_words_per_family, min_zipf=args.min_zipf
    )
    seen_fams, heldout_fams = P.split_families(bank, args.heldout_fraction, seed=args.seed)
    print(f"  {len(bank)} families | {len(seen_fams)} seen | {len(heldout_fams)} held-out")

    # Build one large seen-family pool, then carve disjoint train/val/test_id.
    n_seen_needed = args.n_train + args.n_val + args.n_test_id
    seen_pool = P.build_prompt_pool(n_seen_needed, seen_fams, bank, seed=args.seed, balance=True)
    if len(seen_pool) < n_seen_needed:
        print(f"  WARNING: only {len(seen_pool)} unique seen prompts "
              f"(< requested {n_seen_needed}); increase --max_words_per_family.")
    rng.shuffle(seen_pool)

    train = seen_pool[: args.n_train]
    val = seen_pool[args.n_train: args.n_train + args.n_val]
    test_id = seen_pool[args.n_train + args.n_val:]
    test_id = test_id[: args.n_test_id]

    test_heldout = P.build_prompt_pool(
        args.n_test_heldout, heldout_fams, bank, seed=args.seed + 1, balance=True
    )

    # Recovery prompts: reuse a fresh sample of seen-family base prompts.
    recovery_bases = P.build_prompt_pool(
        args.n_recovery, seen_fams, bank, seed=args.seed + 2, balance=True
    )
    recovery = [_make_recovery(r, rng) for r in recovery_bases]

    for name, split in [("train", train), ("val", val), ("test_id", test_id),
                        ("test_heldout_family", test_heldout), ("recovery", recovery)]:
        # re-key ids within each split
        for i, r in enumerate(split):
            r["id"] = i
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        n = write_jsonl(path, split)
        print(f"  wrote {n:>6}  {path}")

    write_json(os.path.join(args.out_dir, "families.json"), {
        "seed": args.seed,
        "n_families": len(bank),
        "seen_families": seen_fams,
        "heldout_families": heldout_fams,
        "bank": bank,
    })
    print("Done.")


if __name__ == "__main__":
    main()
