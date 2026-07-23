#!/usr/bin/env python3
"""Phase 5: linear probing of rhyme-family planning (outline Section 13).

Trains a linear probe on residual activations to predict the second-line rhyme
*family* at each (layer, position), following the look-ahead decodable-vs-causal
distinction. Reports, per position/layer, the probe accuracy, plus the key
statistic

    Delta_newline = max_layer [ acc(newline, i=0) - acc(first_generated, i=+1) ],

which measures whether rhyme information is unusually concentrated at the newline
(a decodable planning site). A shuffled-label control bounds chance.

Takes disjoint train / val activation files from ``mech.extract_activations`` so
probe accuracy reflects generalization, not memorization.

Usage:
  python -m mech.probe --train_acts mech/acts/base_train.pt \
      --val_acts mech/acts/base_val.pt --out results/probe/base.json
"""
import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi import rhyme as R  # noqa: E402
from opi.io_utils import write_json  # noqa: E402


def build_family_vocab(words_list):
    fams = {}
    for words in words_list:
        for w in words:
            f = R.rhyme_family(w)
            if f is not None and f not in fams:
                fams[f] = len(fams)
    return fams


def words_to_labels(words, fam_vocab):
    labels, keep = [], []
    for i, w in enumerate(words):
        f = R.rhyme_family(w)
        if f in fam_vocab:
            labels.append(fam_vocab[f])
            keep.append(i)
    return torch.tensor(labels), torch.tensor(keep)


def train_linear_probe(X, y, n_classes, device, epochs=60, lr=1e-2, wd=1e-3):
    probe = torch.nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    X = X.to(device).float()
    y = y.to(device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(X), y)
        loss.backward()
        opt.step()
    return probe


@torch.no_grad()
def eval_probe(probe, X, y, device):
    logits = probe(X.to(device).float())
    pred = logits.argmax(-1)
    return (pred == y.to(device)).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_acts", required=True)
    ap.add_argument("--val_acts", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle_control", action="store_true", default=True)
    args = ap.parse_args()

    tr = torch.load(args.train_acts, weights_only=False)
    va = torch.load(args.val_acts, weights_only=False)
    offsets = tr["offsets"]
    n_layers = tr["meta"]["n_layers"]

    fam_vocab = build_family_vocab([tr["target_words"], va["target_words"]])
    n_classes = len(fam_vocab)
    y_tr_all, keep_tr = words_to_labels(tr["target_words"], fam_vocab)
    y_va_all, keep_va = words_to_labels(va["target_words"], fam_vocab)
    print(f"families={n_classes} train={len(keep_tr)} val={len(keep_va)}", flush=True)

    results = {}       # f"{offset}_{layer}" -> acc
    shuffled = {}
    rng = random.Random(0)
    for off in offsets:
        for L in range(n_layers):
            Xtr = tr["acts"][off][L][keep_tr]
            Xva = va["acts"][off][L][keep_va]
            probe = train_linear_probe(Xtr, y_tr_all, n_classes, args.device)
            acc = eval_probe(probe, Xva, y_va_all, args.device)
            results[f"{off}_{L}"] = acc
            if args.shuffle_control and L == n_layers // 2:
                yp = y_tr_all[torch.randperm(len(y_tr_all))]
                probe_s = train_linear_probe(Xtr, yp, n_classes, args.device)
                shuffled[f"{off}_{L}"] = eval_probe(probe_s, Xva, y_va_all, args.device)
        peak = max(results[f"{off}_{L}"] for L in range(n_layers))
        print(f"  offset {off:+d}: peak_acc={peak:.3f}", flush=True)

    # Delta_newline over layers: acc(i=0) - acc(i=+1)
    if 0 in offsets and 1 in offsets:
        deltas = [results[f"0_{L}"] - results[f"1_{L}"] for L in range(n_layers)]
        delta_newline = max(deltas)
        argmax_layer = max(range(n_layers), key=lambda L: deltas[L])
    else:
        delta_newline, argmax_layer = None, None

    tag = args.tag or os.path.basename(args.out).replace(".json", "")
    out = {
        "tag": tag, "n_families": n_classes,
        "n_train": len(keep_tr), "n_val": len(keep_va),
        "n_layers": n_layers, "offsets": offsets,
        "acc": results,
        "shuffle_control": shuffled,
        "peak_acc_by_offset": {str(off): max(results[f"{off}_{L}"] for L in range(n_layers))
                               for off in offsets},
        "delta_newline": delta_newline,
        "delta_newline_layer": argmax_layer,
    }
    write_json(args.out, out)
    print(f"Delta_newline={delta_newline} at layer {argmax_layer} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
