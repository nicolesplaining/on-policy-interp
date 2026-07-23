#!/usr/bin/env python3
"""Forgetting evaluation (outline Section 12).

Compares a trained checkpoint against the frozen base student on non-poetry text:

  * output-distribution drift:  mean KL(base || trained) over next-token dists
  * representation drift:        per-layer linear CKA(base, trained)
  * capability retention:        accuracy on compact probes, base vs trained

The central prediction (H1) is that on-policy KD produces *task-selective*
changes: low general-text KL and high CKA relative to the static-SFT conditions.

Usage:
  python -m eval.forgetting --trained runs/onpolicy_kd_seed0/ckpt_100 \
      --base google/gemma-3-4b-it --device cuda:0
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi.io_utils import write_json  # noqa: E402
from eval.common import load_for_eval  # noqa: E402
from eval.general_text import GENERAL_TEXT, CAPABILITY_PROBES  # noqa: E402


@torch.no_grad()
def kl_and_hidden(model, tokenizer, texts, device, max_len=64):
    """Return next-token logprobs and per-layer hidden states for each text."""
    per_text = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        logprobs = F.log_softmax(out.logits[0].float(), dim=-1)  # [T, V]
        hs = [h[0].float().cpu() for h in out.hidden_states]      # list[L+1] of [T, d]
        per_text.append((logprobs.cpu(), hs))
    return per_text


def mean_kl(base_lp, trained_lp):
    """Mean KL(base || trained) over positions (base as reference)."""
    p = base_lp.exp()
    kl = (p * (base_lp - trained_lp)).sum(-1)  # [T]
    return kl.mean().item()


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between activation matrices X, Y of shape [N, d]."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.t() @ Y).norm() ** 2
    xx = (X.t() @ X).norm()
    yy = (Y.t() @ Y).norm()
    denom = (xx * yy).clamp_min(1e-12)
    return (xy / denom).item()


@torch.no_grad()
def capability(model, tokenizer, device, max_new_tokens=8):
    hits = 0
    for prompt, expect in CAPABILITY_PROBES:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        cont = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if expect.lower() in cont.lower():
            hits += 1
    return hits / len(CAPABILITY_PROBES)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    texts = GENERAL_TEXT
    print("Loading base...", flush=True)
    base, tok, _ = load_for_eval(args.base, args.device)
    base_out = kl_and_hidden(base, tok, texts, args.device)
    base_cap = capability(base, tok, args.device)
    del base
    torch.cuda.empty_cache()

    print("Loading trained...", flush=True)
    trained, tok2, _ = load_for_eval(args.trained, args.device)
    trained_out = kl_and_hidden(trained, tok2, texts, args.device)
    trained_cap = capability(trained, tok2, args.device)

    # output KL
    kls = [mean_kl(b[0], t[0]) for b, t in zip(base_out, trained_out)]
    mean_output_kl = sum(kls) / len(kls)

    # per-layer CKA: concatenate positions across texts per layer
    n_layers = len(base_out[0][1])
    cka_per_layer = []
    for L in range(n_layers):
        Xb = torch.cat([b[1][L] for b in base_out], dim=0)
        Xt = torch.cat([t[1][L] for t in trained_out], dim=0)
        cka_per_layer.append(linear_cka(Xb, Xt))

    tag = args.tag or os.path.basename(args.trained.rstrip("/"))
    out = {
        "trained": args.trained, "base": args.base, "tag": tag,
        "mean_output_kl": mean_output_kl,
        "cka_per_layer": cka_per_layer,
        "mean_cka": sum(cka_per_layer) / len(cka_per_layer),
        "min_cka": min(cka_per_layer),
        "capability_base": base_cap,
        "capability_trained": trained_cap,
        "capability_delta": trained_cap - base_cap,
    }
    print(f"  mean_output_KL={mean_output_kl:.4f}  mean_CKA={out['mean_cka']:.4f}  "
          f"min_CKA={out['min_cka']:.4f}  cap {base_cap:.3f}->{trained_cap:.3f}",
          flush=True)
    out_path = args.out or f"results/forgetting/{tag}.json"
    write_json(out_path, out)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
