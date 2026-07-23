#!/usr/bin/env python3
"""Phase 2: measure and report distributional differences between the fixed
corpus and the teacher traces (outline Section 7.5).

The two static datasets share first lines by construction; this checks the
*second-line* observables that could confound a mechanism comparison:

  * number of examples
  * second-line length (tokens by whitespace)
  * rhyme-family frequency distribution
  * final (rhyme) word frequency
  * type/token lexical diversity

Emits a JSON report and a short console summary. Any residual differences are
reported, per the outline's requirement to measure and disclose them.
"""
import argparse
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi import rhyme as R  # noqa: E402
from opi.io_utils import read_jsonl, write_json  # noqa: E402


def _second_line(rec):
    """Return the second-line text for a corpus or teacher record."""
    if "target" in rec:
        return rec["target"]
    return rec.get("text", "").split("\n")[-1]


def profile(records):
    lengths, fams, finals = [], Counter(), Counter()
    for r in records:
        sl = _second_line(r)
        toks = sl.split()
        lengths.append(len(toks))
        fam = r.get("rhyme_family") or R.rhyme_family(R.last_alpha_word(sl) or "")
        if fam:
            fams[fam] += 1
        fw = r.get("second_word") or R.last_alpha_word(sl)
        if fw:
            finals[fw] += 1
    n = len(records)
    mean_len = sum(lengths) / n if n else 0.0
    var_len = sum((x - mean_len) ** 2 for x in lengths) / n if n else 0.0
    return {
        "n": n,
        "mean_second_line_len": mean_len,
        "std_second_line_len": math.sqrt(var_len),
        "n_rhyme_families": len(fams),
        "n_unique_final_words": len(finals),
        "type_token_ratio": (len(finals) / sum(finals.values())) if finals else 0.0,
        "family_freq": dict(fams),
        "top_final_words": dict(finals.most_common(20)),
    }


def _tv_distance(a: Counter, b: Counter) -> float:
    """Total-variation distance between two count distributions."""
    ta, tb = sum(a.values()), sum(b.values())
    if ta == 0 or tb == 0:
        return 1.0
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus/corpus_sft.jsonl")
    ap.add_argument("--teacher", default="data/teacher_traces/teacher_sft.jsonl")
    ap.add_argument("--out", default="data/match_report.json")
    args = ap.parse_args()

    corpus = read_jsonl(args.corpus)
    teacher = read_jsonl(args.teacher)
    pc, pt = profile(corpus), profile(teacher)

    fam_tv = _tv_distance(Counter(pc["family_freq"]), Counter(pt["family_freq"]))
    report = {
        "corpus": pc,
        "teacher": pt,
        "differences": {
            "family_freq_tv_distance": fam_tv,
            "mean_len_gap": pc["mean_second_line_len"] - pt["mean_second_line_len"],
            "ttr_gap": pc["type_token_ratio"] - pt["type_token_ratio"],
        },
    }
    write_json(args.out, report)

    print(f"{'metric':<28}{'corpus':>12}{'teacher':>12}")
    print("-" * 52)
    for k in ["n", "mean_second_line_len", "std_second_line_len",
              "n_rhyme_families", "n_unique_final_words", "type_token_ratio"]:
        print(f"{k:<28}{pc[k]:>12.3f}{pt[k]:>12.3f}" if isinstance(pc[k], float)
              else f"{k:<28}{pc[k]:>12}{pt[k]:>12}")
    print("-" * 52)
    print(f"rhyme-family freq TV distance: {fam_tv:.3f}")
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
