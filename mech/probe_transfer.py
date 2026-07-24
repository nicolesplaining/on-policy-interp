#!/usr/bin/env python3
"""Cross-condition probe transfer (outline Section 13.5).

Train a rhyme-family probe on a *source* model's activations and evaluate it on a
*target* model's activations. High transfer accuracy = the two models encode the
rhyme family in aligned linear geometry. Used to test representational
preservation: a probe trained on the corpus_sft *start* should transfer well to a
continuation that preserved the representation (on-policy) and poorly to one that
reorganized it (off-policy).

Both source and target must have activation files from ``mech.extract_activations``
extracted on the same prompt pool. Compares transfer accuracy at the newline
(offset 0), best layer chosen on the source's own val split.

Usage:
  python -m mech.probe_transfer --source mech/acts/corpus_sft_seed0_ckpt_100 \
      --targets mech/acts/relax_onpolicy mech/acts/relax_offpolicy mech/acts/relax_moresft \
      --out results/probe_transfer.json
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi import rhyme as R  # noqa: E402
from opi.io_utils import write_json  # noqa: E402


def load_split(prefix):
    tr = torch.load(f"{prefix}_train.pt", weights_only=False)
    va = torch.load(f"{prefix}_val.pt", weights_only=False)
    return tr, va


def build_vocab(*word_lists):
    fams = {}
    for words in word_lists:
        for w in words:
            f = R.rhyme_family(w)
            if f is not None and f not in fams:
                fams[f] = len(fams)
    return fams


def labels(words, vocab):
    lab, keep = [], []
    for i, w in enumerate(words):
        f = R.rhyme_family(w)
        if f in vocab:
            lab.append(vocab[f]); keep.append(i)
    return torch.tensor(lab), torch.tensor(keep)


def train_probe(X, y, ncls, device, epochs=80, lr=1e-2, wd=1e-3):
    p = torch.nn.Linear(X.shape[1], ncls).to(device)
    opt = torch.optim.AdamW(p.parameters(), lr=lr, weight_decay=wd)
    X = X.to(device).float(); y = y.to(device)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(p(X), y).backward(); opt.step()
    return p


@torch.no_grad()
def acc(p, X, y, device):
    return (p(X.to(device).float()).argmax(-1) == y.to(device)).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--offset", type=int, default=0)  # newline
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s_tr, s_va = load_split(args.source)
    off = args.offset
    n_layers = s_tr["meta"]["n_layers"]

    # shared family vocab from source + all targets
    tgt_data = {t: load_split(t) for t in args.targets}
    vocab = build_vocab(s_tr["target_words"], s_va["target_words"],
                        *[d[0]["target_words"] for d in tgt_data.values()],
                        *[d[1]["target_words"] for d in tgt_data.values()])
    ncls = len(vocab)
    ys_tr, k_tr = labels(s_tr["target_words"], vocab)
    ys_va, k_va = labels(s_va["target_words"], vocab)

    # choose best source layer on source val, train probe there, then transfer
    best = {"layer": None, "self": -1, "probe": None}
    for L in range(n_layers):
        p = train_probe(s_tr["acts"][off][L][k_tr], ys_tr, ncls, args.device)
        a = acc(p, s_va["acts"][off][L][k_va], ys_va, args.device)
        if a > best["self"]:
            best = {"layer": L, "self": a, "probe": p}
    L = best["layer"]; p = best["probe"]
    print(f"source best layer={L} self-acc={best['self']:.3f} (families={ncls})", flush=True)

    out = {"source": args.source, "offset": off, "layer": L,
           "source_self_acc": best["self"], "n_families": ncls, "transfer": {}}
    for t, (t_tr, t_va) in tgt_data.items():
        yt, kt = labels(t_va["target_words"], vocab)
        transfer = acc(p, t_va["acts"][off][L][kt], yt, args.device)
        out["transfer"][os.path.basename(t)] = transfer
        print(f"  transfer to {os.path.basename(t):18s} = {transfer:.3f} "
              f"({transfer / best['self']:.2f} of self)", flush=True)
    write_json(args.out, out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
