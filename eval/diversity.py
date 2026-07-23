#!/usr/bin/env python3
"""Diversity evaluation (outline Section 11.3, H5).

Guards against declaring a mode-collapsed on-policy model "better" just because
it repeats a few reliable rhymes. Over sampled completions per prompt, reports:

  * unique final rhyme words + final-word entropy
  * distinct-1 / distinct-2
  * self-BLEU (higher = less diverse)
  * repeated-template frequency (share of the modal 4-gram opening)
  * mean per-prompt rhyme accuracy across samples

Run with sampling (temperature>0, n_samples>1).
"""
import argparse
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi import rhyme as R  # noqa: E402
from opi.io_utils import read_jsonl, write_json  # noqa: E402
from eval.common import load_for_eval, generate_completions  # noqa: E402


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(all_tokens, n):
    grams, total = set(), 0
    for toks in all_tokens:
        gs = _ngrams(toks, n)
        grams.update(gs)
        total += len(gs)
    return len(grams) / max(1, total)


def self_bleu(all_tokens, sample=200):
    """Cheap self-BLEU proxy: mean bigram overlap of each line vs the others."""
    import random
    rng = random.Random(0)
    idx = list(range(len(all_tokens)))
    if len(idx) > sample:
        idx = rng.sample(idx, sample)
    scores = []
    ref_sets = [set(_ngrams(all_tokens[j], 2)) for j in idx]
    for a, ia in enumerate(idx):
        hyp = set(_ngrams(all_tokens[ia], 2))
        if not hyp:
            continue
        overlaps = []
        for b, _ in enumerate(idx):
            if b == a or not ref_sets[b]:
                continue
            overlaps.append(len(hyp & ref_sets[b]) / len(hyp))
        if overlaps:
            scores.append(sum(overlaps) / len(overlaps))
    return sum(scores) / max(1, len(scores))


def entropy(counter: Counter):
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counter.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompts", default="data/prompt_pool/test_id.jsonl")
    ap.add_argument("--max_prompts", type=int, default=300)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tokenizer, _ = load_for_eval(args.model, args.device)
    tag = args.tag or os.path.basename(args.model.rstrip("/"))
    recs = read_jsonl(args.prompts)[: args.max_prompts]
    prompts = [r["prompt"] for r in recs]
    comps = generate_completions(model, tokenizer, prompts, args.device,
                                 n_samples=args.n_samples, temperature=args.temperature,
                                 max_new_tokens=args.max_new_tokens)

    final_words = Counter()
    all_tokens, per_prompt_rhyme = [], []
    opening_4grams = Counter()
    for r, cs in zip(recs, comps):
        hits = 0
        for c in cs:
            line = c.split("\n")[0]
            toks = line.split()
            all_tokens.append(toks)
            if len(toks) >= 4:
                opening_4grams[tuple(toks[:4])] += 1
            sw = R.extract_second_line_word_with_newline(c)
            if sw:
                final_words[sw] += 1
            if R.do_rhyme(r["rhyme_word"], sw or "") is True:
                hits += 1
        per_prompt_rhyme.append(hits / max(1, len(cs)))

    total_final = sum(final_words.values())
    modal_template = opening_4grams.most_common(1)[0][1] if opening_4grams else 0
    out = {
        "model": args.model, "tag": tag,
        "n_prompts": len(recs), "n_samples": args.n_samples,
        "temperature": args.temperature,
        "unique_final_words": len(final_words),
        "final_word_entropy_bits": entropy(final_words),
        "final_word_type_token_ratio": len(final_words) / max(1, total_final),
        "distinct_1": distinct_n(all_tokens, 1),
        "distinct_2": distinct_n(all_tokens, 2),
        "self_bleu": self_bleu(all_tokens),
        "repeated_template_frac": modal_template / max(1, sum(opening_4grams.values())),
        "mean_rhyme_accuracy": sum(per_prompt_rhyme) / max(1, len(per_prompt_rhyme)),
    }
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k:28s} {v:.4f}")
    out_path = args.out or f"results/diversity/{tag}.json"
    write_json(out_path, out)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
