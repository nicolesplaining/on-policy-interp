#!/usr/bin/env python3
"""Behavioral evaluation (outline Section 11.1-11.2).

Metrics per checkpoint:
  * rhyme accuracy (in-distribution test)
  * valid-couplet rate (rhymes AND ends cleanly at a newline)
  * held-out-rhyme-family accuracy
  * recovery-prefix rhyme accuracy (continue from malformed partial 2nd lines)

Usage:
  python -m eval.behavioral --model runs/corpus_sft_seed0/ckpt_100 --device cuda:0
  python -m eval.behavioral --model google/gemma-3-4b-it --tag base --device cuda:0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi import rhyme as R  # noqa: E402
from opi.io_utils import read_jsonl, write_json  # noqa: E402
from eval.common import load_for_eval, generate_completions  # noqa: E402


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def score_split(model, tokenizer, device, records, temperature, n_samples, max_new_tokens):
    prompts = [r["prompt"] for r in records]
    comps = generate_completions(model, tokenizer, prompts, device,
                                 n_samples=n_samples, temperature=temperature,
                                 max_new_tokens=max_new_tokens)
    n_rhyme = n_valid = n_attempt = 0
    for r, cs in zip(records, comps):
        for c in cs:
            sw = R.extract_second_line_word_with_newline(c)
            n_attempt += 1
            res = R.do_rhyme(r["rhyme_word"], sw or "")
            rhymes = res is True
            n_rhyme += 1 if rhymes else 0
            # valid couplet: rhymes and the generation actually terminated a line
            ends_line = ("\n" in c) or ("<end_of_turn>" in c)
            n_valid += 1 if (rhymes and ends_line) else 0
    lo, hi = _wilson(n_rhyme, n_attempt)
    return {
        "n": len(records), "n_attempt": n_attempt,
        "rhyme_accuracy": n_rhyme / max(1, n_attempt),
        "valid_couplet_rate": n_valid / max(1, n_attempt),
        "ci_low": lo, "ci_high": hi,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool_dir", default="data/prompt_pool")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_prompts", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    args = ap.parse_args()

    model, tokenizer, _ = load_for_eval(args.model, args.device)
    tag = args.tag or os.path.basename(args.model.rstrip("/"))

    out = {"model": args.model, "tag": tag,
           "temperature": args.temperature, "n_samples": args.n_samples}
    for split in ["test_id", "test_heldout_family", "recovery"]:
        path = os.path.join(args.pool_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        recs = read_jsonl(path)[: args.max_prompts]
        out[split] = score_split(model, tokenizer, args.device, recs,
                                  args.temperature, args.n_samples, args.max_new_tokens)
        print(f"  {split:22s} rhyme={out[split]['rhyme_accuracy']:.3f} "
              f"valid={out[split]['valid_couplet_rate']:.3f} "
              f"CI=[{out[split]['ci_low']:.3f},{out[split]['ci_high']:.3f}]", flush=True)

    out_path = args.out or f"results/behavioral/{tag}.json"
    write_json(out_path, out)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
