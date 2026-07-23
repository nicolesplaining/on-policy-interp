#!/usr/bin/env python3
"""Refined handoff detection from stored patching C-arrays (no GPU needed).

The original ``mech.activation_patching`` flag used an early-third vs late-third
mean split, which misses a handoff that peaks mid-network and decays in the final
layers (as the 27B teacher's does). This recomputes a faithful criterion from the
committed ``C_newline_by_layer`` / ``C_rhyme_word_by_layer`` arrays:

    handoff  <=>  exists a layer L with
                    C_newline[L] - C_rhyme_word[L] >  margin   AND
                    C_newline[L]                    >  min_newline

i.e. a band where the newline is *both* causally effective and more effective
than the rhyme word. Reports max H, its normalized depth, and the count of
qualifying layers, per model (student conditions averaged over seeds).
"""
import argparse
import glob
import json
import os

import numpy as np


def load_C(results_dir, tag):
    paths = sorted(glob.glob(os.path.join(results_dir, "patching", f"{tag}_seed*_ckpt_100.json")))
    if not paths:
        p = os.path.join(results_dir, "patching", f"{tag}.json")
        if not os.path.exists(p):
            return None
        paths = [p]
    cn, cr = [], []
    for p in paths:
        d = json.load(open(p))
        cn.append(d["C_newline_by_layer"]); cr.append(d["C_rhyme_word_by_layer"])
    return np.array(cn).mean(0), np.array(cr).mean(0)


def detect(cn, cr, margin=0.1, min_newline=0.2):
    H = cn - cr
    qualifying = [L for L in range(len(H)) if H[L] > margin and cn[L] > min_newline]
    amax = int(np.argmax(H))
    return {
        "max_H": float(H.max()),
        "max_H_norm_depth": amax / max(1, len(H) - 1),
        "peak_newline_C": float(cn.max()),
        "peak_newline_C_norm_depth": int(cn.argmax()) / max(1, len(cn) - 1),
        "n_handoff_layers": len(qualifying),
        "handoff": len(qualifying) > 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--min_newline", type=float, default=0.2)
    ap.add_argument("--out", default="results/handoff.json")
    args = ap.parse_args()

    tags = ["base", "corpus_sft", "teacher_sft", "teacher_kd", "onpolicy_kd", "base_teacher"]
    out = {}
    print(f"{'model':14s} {'handoff':>8} {'max_H':>7} {'@depth':>7} {'peakNL_C':>9} {'@depth':>7} {'#layers':>8}")
    for t in tags:
        C = load_C(args.results_dir, t)
        if C is None:
            continue
        r = detect(*C, margin=args.margin, min_newline=args.min_newline)
        out[t] = r
        print(f"{t:14s} {str(r['handoff']):>8} {r['max_H']:>7.3f} "
              f"{r['max_H_norm_depth']:>7.2f} {r['peak_newline_C']:>9.3f} "
              f"{r['peak_newline_C_norm_depth']:>7.2f} {r['n_handoff_layers']:>8}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
