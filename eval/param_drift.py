#!/usr/bin/env python3
"""Parameter-update distribution (outline Section 12.4).

Descriptive complement to the activation analysis: where did each regime move
weights? Loads the base and trained models on CPU and reports, for the update
``Delta = trained - base``:

  * per-layer update L2 norm (relative depth)
  * attention vs MLP update magnitude
  * embedding / final-norm update magnitude
  * concentration of update energy (Gini + top-25% layer share)

Usage:
  python -m eval.param_drift --trained runs/teacher_sft_seed0/ckpt_100 \
      --base google/gemma-3-4b-it --out results/param_drift/teacher_sft.json
"""
import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi.models import load_model  # noqa: E402
from opi.io_utils import write_json  # noqa: E402


def _state_dict(path):
    model, _, _ = load_model(path, device_map=None)
    sd = {k: v.float() for k, v in model.state_dict().items()}
    del model
    return sd


def _gini(values):
    xs = sorted(v for v in values if v >= 0)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("Loading base state dict...", flush=True)
    base = _state_dict(args.base)
    print("Loading trained state dict...", flush=True)
    trained = _state_dict(args.trained)

    layer_norm2 = {}     # layer idx -> summed squared update
    by_type = {"attn": 0.0, "mlp": 0.0, "embed": 0.0, "norm": 0.0, "other": 0.0}
    total2 = 0.0
    layer_re = re.compile(r"layers\.(\d+)\.")

    for k, w in trained.items():
        if k not in base or base[k].shape != w.shape:
            continue
        d2 = float((w - base[k]).pow(2).sum().item())
        total2 += d2
        m = layer_re.search(k)
        if m:
            layer_norm2.setdefault(int(m.group(1)), 0.0)
            layer_norm2[int(m.group(1))] += d2
        if "self_attn" in k:
            by_type["attn"] += d2
        elif "mlp" in k:
            by_type["mlp"] += d2
        elif "embed" in k or "lm_head" in k:
            by_type["embed"] += d2
        elif "norm" in k:
            by_type["norm"] += d2
        else:
            by_type["other"] += d2

    layers_sorted = sorted(layer_norm2)
    per_layer_norm = [layer_norm2[i] ** 0.5 for i in layers_sorted]
    n = len(layers_sorted)
    # top-25% layer share of update energy
    energies = sorted(layer_norm2.values(), reverse=True)
    top_k = max(1, n // 4)
    top_share = sum(energies[:top_k]) / max(1e-12, sum(energies))

    tag = args.tag or os.path.basename(args.trained.rstrip("/"))
    out = {
        "trained": args.trained, "base": args.base, "tag": tag,
        "total_update_norm": total2 ** 0.5,
        "n_layers": n,
        "per_layer_update_norm": {str(i): layer_norm2[i] ** 0.5 for i in layers_sorted},
        "update_by_type": {k: v ** 0.5 for k, v in by_type.items()},
        "attn_mlp_ratio": (by_type["attn"] / by_type["mlp"]) if by_type["mlp"] else None,
        "gini_layer_energy": _gini(list(layer_norm2.values())),
        "top25pct_layer_energy_share": top_share,
    }
    print(f"  total_update_norm={out['total_update_norm']:.3f} "
          f"gini={out['gini_layer_energy']:.3f} "
          f"top25%_share={top_share:.3f} attn/mlp={out['attn_mlp_ratio']}", flush=True)
    out_path = args.out or f"results/param_drift/{tag}.json"
    write_json(out_path, out)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
