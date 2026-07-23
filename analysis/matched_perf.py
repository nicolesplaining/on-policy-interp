#!/usr/bin/env python3
"""Matched-performance checkpoint selection (outline Section 10.2).

Instead of comparing every condition at ckpt_100 (which reach *different* rhyme
accuracies, confounding "how much did it train" with "how it trained"), select
for each (condition, seed) the **earliest longitudinal checkpoint** whose
validation rhyme accuracy reaches a common threshold, and compare forgetting at
those matched checkpoints.

    python -m analysis.matched_perf --threshold 0.85 --out results/matched/selection.json

Writes a manifest ``{condition: {seed: {tag, val_rhyme, step}}}`` consumed by
``scripts/run_matched.sh`` to run forgetting/behavioral on the selected ckpts.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi.io_utils import write_json  # noqa: E402

CONDS = ["corpus_sft", "teacher_sft", "teacher_kd", "onpolicy_kd"]


def select(runs_dir, conds, seeds, threshold):
    manifest, reachable = {}, {}
    for c in conds:
        manifest[c] = {}
        for s in seeds:
            h = json.load(open(os.path.join(runs_dir, f"{c}_seed{s}", "history.json")))
            cks = h["checkpoints"]
            chosen = None
            for ck in cks:  # in training order -> earliest first
                if ck["tag"] == "ckpt_000":
                    continue
                if ck["val_rhyme"] >= threshold:
                    chosen = ck
                    break
            if chosen is None:  # never reached threshold: use best
                chosen = max(cks, key=lambda x: x["val_rhyme"])
            manifest[c][str(s)] = {
                "tag": chosen["tag"], "val_rhyme": chosen["val_rhyme"],
                "step": chosen["step"],
                "reached": chosen["val_rhyme"] >= threshold,
            }
        reachable[c] = min(m["val_rhyme"] for m in manifest[c].values())
    return manifest, reachable


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/matched/selection.json")
    args = ap.parse_args()

    manifest, reachable = select(args.runs_dir, CONDS, args.seeds, args.threshold)
    write_json(args.out, {"threshold": args.threshold, "seeds": args.seeds,
                          "selection": manifest})
    print(f"Matched-performance selection @ rhyme >= {args.threshold}:")
    for c in CONDS:
        picks = "  ".join(f"s{s}:{manifest[c][str(s)]['tag']}"
                          f"({manifest[c][str(s)]['val_rhyme']:.3f}"
                          f"{'' if manifest[c][str(s)]['reached'] else '!'})"
                          for s in args.seeds)
        print(f"  {c:12s} {picks}")
    print("(! = never reached threshold; using best checkpoint)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
