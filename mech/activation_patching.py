#!/usr/bin/env python3
"""Phase 5: activation patching + rhyme-word -> newline handoff (Section 14).

Corrupt->clean residual patching, per layer, at the rhyme-word and newline
positions. For each layer L we measure

    C[L, pos] = P(second line adopts the *corrupt* rhyme | patch resid_pre at L,pos)

and the handoff statistic

    H[L] = C[L, newline] - C[L, rhyme_word].

A causal newline handoff requires (outline Section 14.5): H[L] negative in early
layers, positive in later layers, bootstrap CIs excluding zero in the relevant
regions, replicated in >=2/3 seeds. A high newline *probe* score alone is not
sufficient — this is the causal complement to ``mech.probe``.

Reuses the look-ahead ``patch_all_layers_unified`` hook idioms.

Usage:
  python -m mech.activation_patching --model runs/teacher_sft_seed0/ckpt_100 \
      --device cuda:0 --n_samples 20 --out results/patching/teacher_sft.json
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi import rhyme as R  # noqa: E402
from opi.positions import second_newline_token, rhyme_word_position  # noqa: E402
from opi.io_utils import write_json  # noqa: E402
from eval.common import load_for_eval  # noqa: E402

# Matched clean/corrupt pairs (from look-ahead), first-line rhyme word swapped.
PROMPT_PAIRS = [
    {"pair_id": "doom_dread",
     "clean_prompt": "A rhyming couplet:\nThe empty house was filled with silent doom,\n",
     "corrupt_prompt": "A rhyming couplet:\nThe empty house was filled with silent dread,\n",
     "clean_rhyme_word": "doom", "corrupt_rhyme_word": "dread"},
    {"pair_id": "bliss_joy",
     "clean_prompt": "A rhyming couplet:\nThe children laughed in bliss,\n",
     "corrupt_prompt": "A rhyming couplet:\nThe children laughed in joy,\n",
     "clean_rhyme_word": "bliss", "corrupt_rhyme_word": "joy"},
    {"pair_id": "dark_night",
     "clean_prompt": "A rhyming couplet:\nShe wandered home alone into the dark,\n",
     "corrupt_prompt": "A rhyming couplet:\nShe wandered home alone into the night,\n",
     "clean_rhyme_word": "dark", "corrupt_rhyme_word": "night"},
    {"pair_id": "grief_pain",
     "clean_prompt": "A rhyming couplet:\nI never knew the depth of such grief,\n",
     "corrupt_prompt": "A rhyming couplet:\nI never knew the depth of such pain,\n",
     "clean_rhyme_word": "grief", "corrupt_rhyme_word": "pain"},
    {"pair_id": "fright_fear",
     "clean_prompt": "A rhyming couplet:\nShe felt a sudden sense of fright,\n",
     "corrupt_prompt": "A rhyming couplet:\nShe felt a sudden sense of fear,\n",
     "clean_rhyme_word": "fright", "corrupt_rhyme_word": "fear"},
]


@torch.no_grad()
def sample_completions(model, tokenizer, prompt, device, n, temperature, max_new_tokens):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, num_return_sequences=n,
                         pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(row, skip_special_tokens=True) for row in out]


@torch.no_grad()
def cache_resid_all_layers(model, adapter, tokenizer, prompt, pos, device):
    layers = adapter.get_layers()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cached = [None] * len(layers)
    handles = []

    def mk(i):
        def hook(module, args):
            h = args[0]
            if h.shape[1] > pos:
                cached[i] = h[:, pos, :].detach().clone()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(mk(i)))
    model(**enc, use_cache=False)
    for h in handles:
        h.remove()
    return cached


def make_patch_hook(vec, pos):
    def hook(module, args):
        h = args[0]
        if h.shape[1] <= pos:
            return args
        out = h.clone()
        out[:, pos, :] = vec.to(out.device, dtype=out.dtype)
        return (out,) + args[1:]
    return hook


def run_pair(model, tokenizer, adapter, pair, device, n_samples, temperature, max_new_tokens):
    n_layers = adapter.get_n_layers()
    layers = adapter.get_layers()
    clean, corrupt = pair["clean_prompt"], pair["corrupt_prompt"]
    cw, xw = pair["clean_rhyme_word"], pair["corrupt_rhyme_word"]

    pos_map = {}  # name -> (clean_pos, corrupt_pos)
    pos_map["newline"] = (second_newline_token(tokenizer, clean),
                          second_newline_token(tokenizer, corrupt))
    pos_map["rhyme_word"] = (rhyme_word_position(tokenizer, clean)[0],
                             rhyme_word_position(tokenizer, corrupt)[0])

    # baseline corrupt-rhyme rate on the unpatched clean run
    base_comps = sample_completions(model, tokenizer, clean, device, n_samples,
                                    temperature, max_new_tokens)
    baseline = R.rhyme_rate(base_comps, clean, xw)

    out = {"pair_id": pair["pair_id"], "baseline_corrupt_rate": baseline,
           "C": {"newline": [], "rhyme_word": []}}
    for name in ["newline", "rhyme_word"]:
        clean_pos, corrupt_pos = pos_map[name]
        cache = cache_resid_all_layers(model, adapter, tokenizer, corrupt, corrupt_pos, device)
        for L in range(n_layers):
            handle = layers[L].register_forward_pre_hook(make_patch_hook(cache[L], clean_pos))
            comps = sample_completions(model, tokenizer, clean, device, n_samples,
                                       temperature, max_new_tokens)
            handle.remove()
            out["C"][name].append(R.rhyme_rate(comps, clean, xw))
    out["H"] = [out["C"]["newline"][L] - out["C"]["rhyme_word"][L] for L in range(n_layers)]
    return out


def detect_handoff(H_by_layer):
    """H negative in an early band, positive in a late band."""
    n = len(H_by_layer)
    third = max(1, n // 3)
    early = sum(H_by_layer[:third]) / third
    late = sum(H_by_layer[-third:]) / third
    return {
        "early_mean_H": early, "late_mean_H": late,
        "handoff": (early < 0) and (late > 0),
        "peak_newline_layer": max(range(n), key=lambda L: H_by_layer[L]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tokenizer, adapter = load_for_eval(args.model, args.device)
    n_layers = adapter.get_n_layers()
    pairs = []
    for pair in PROMPT_PAIRS:
        res = run_pair(model, tokenizer, adapter, pair, args.device,
                       args.n_samples, args.temperature, args.max_new_tokens)
        pairs.append(res)
        print(f"  {pair['pair_id']}: baseline={res['baseline_corrupt_rate']:.3f} "
              f"peak_newline_C={max(res['C']['newline']):.3f} "
              f"peak_rhymeword_C={max(res['C']['rhyme_word']):.3f}", flush=True)

    # aggregate H across pairs
    H_mean = [sum(p["H"][L] for p in pairs) / len(pairs) for L in range(n_layers)]
    C_nl = [sum(p["C"]["newline"][L] for p in pairs) / len(pairs) for L in range(n_layers)]
    C_rw = [sum(p["C"]["rhyme_word"][L] for p in pairs) / len(pairs) for L in range(n_layers)]
    handoff = detect_handoff(H_mean)

    tag = args.tag or os.path.basename(args.model.rstrip("/"))
    out = {
        "tag": tag, "model": args.model, "n_layers": n_layers,
        "n_samples": args.n_samples, "pairs": pairs,
        "H_mean_by_layer": H_mean,
        "C_newline_by_layer": C_nl, "C_rhyme_word_by_layer": C_rw,
        "peak_newline_C": max(C_nl), "peak_rhyme_word_C": max(C_rw),
        "handoff": handoff,
    }
    write_json(args.out, out)
    print(f"handoff={handoff['handoff']} early_H={handoff['early_mean_H']:.3f} "
          f"late_H={handoff['late_mean_H']:.3f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
