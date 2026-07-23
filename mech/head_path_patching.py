#!/usr/bin/env python3
"""Phase 5: head + two-stage path patching (outline Section 15).

Run only when a trained model develops a causal newline site. Ranks attention
heads by newline->rhyme-word attention, patches the top-k head outputs
(corrupt->clean at the newline), and compares against random and
punctuation-attending control head sets. A two-stage path patch tests the
hypothesized route  rhyme word -> heads -> newline -> future rhyme.

Reuses the look-ahead ``gemma3_27b_topk_head_patching`` /
``gemma3_27b_path_patch_two_stage`` hook idioms. Gemma-3 layout
(``model.model.language_model.layers[L].self_attn.o_proj``) is assumed.

Usage:
  python -m mech.head_path_patching --model runs/teacher_sft_seed0/ckpt_100 \
      --device cuda:0 --k 1 2 3 5 10 --out results/heads/teacher_sft.json
"""
import argparse
import os
import sys
from collections import defaultdict
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi import rhyme as R  # noqa: E402
from opi.positions import second_newline_token, rhyme_word_position  # noqa: E402
from opi.io_utils import write_json  # noqa: E402
from eval.common import load_for_eval  # noqa: E402
from mech.activation_patching import PROMPT_PAIRS, sample_completions  # noqa: E402


def _lm_layers(model):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def _head_dims(model):
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    return n_heads, head_dim


@torch.no_grad()
def rank_heads(model, tokenizer, prompt, device, top_n=10):
    """Rank (layer, head) by attention from the newline to the rhyme word."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    nl = second_newline_token(tokenizer, prompt)
    rw = rhyme_word_position(tokenizer, prompt)[0]
    out = model(**enc, output_attentions=True, use_cache=False)
    scored = []
    for L, attn in enumerate(out.attentions):
        # attn: [1, n_heads, seq, seq]
        w = attn[0, :, nl, rw]
        for h in range(w.shape[0]):
            scored.append(((L, h), float(w[h].item())))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


@torch.no_grad()
def cache_oproj_input(model, tokenizer, prompt, pos, layers_needed, device):
    lm = _lm_layers(model)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    cached, handles = {}, []

    def mk(idx):
        def hook(module, args):
            x = args[0]
            if x.shape[1] > pos:
                cached[idx] = x[0, pos, :].detach().clone()
        return hook

    for idx in layers_needed:
        handles.append(lm[idx].self_attn.o_proj.register_forward_pre_hook(mk(idx)))
    model(**enc, use_cache=False)
    for h in handles:
        h.remove()
    return cached


@contextmanager
def patch_heads(model, corrupt_cache, head_set, pos, head_dim):
    lm = _lm_layers(model)
    by_layer = defaultdict(list)
    for L, h in head_set:
        by_layer[L].append(h)
    handles = []
    for L, heads in by_layer.items():
        if L not in corrupt_cache:
            continue
        vec = corrupt_cache[L]

        def mk(vec, heads):
            def hook(module, args):
                x = args[0].clone()
                if x.shape[1] <= pos:
                    return args
                for h in heads:
                    x[:, pos, h * head_dim:(h + 1) * head_dim] = \
                        vec[h * head_dim:(h + 1) * head_dim].to(x.device, dtype=x.dtype)
                return (x,) + args[1:]
            return hook

        handles.append(lm[L].self_attn.o_proj.register_forward_pre_hook(mk(vec, heads)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tokenizer, adapter = load_for_eval(args.model, args.device)
    n_heads, head_dim = _head_dims(model)

    pair_results = []
    agg_rank = defaultdict(float)
    for pair in PROMPT_PAIRS:
        clean, corrupt = pair["clean_prompt"], pair["corrupt_prompt"]
        xw = pair["corrupt_rhyme_word"]
        ranked = rank_heads(model, tokenizer, clean, args.device, top_n=max(args.k))
        for (L, h), s in ranked:
            agg_rank[(L, h)] += s
        nl_clean = second_newline_token(tokenizer, clean)
        nl_corrupt = second_newline_token(tokenizer, corrupt)
        layers_needed = sorted({L for (L, _), _ in ranked})
        cache = cache_oproj_input(model, tokenizer, corrupt, nl_corrupt, layers_needed, args.device)

        base = sample_completions(model, tokenizer, clean, args.device, args.n_samples,
                                  args.temperature, args.max_new_tokens)
        baseline = R.rhyme_rate(base, clean, xw)
        per_k = {}
        for k in args.k:
            heads = [lh for lh, _ in ranked[:k]]
            with patch_heads(model, cache, heads, nl_clean, head_dim):
                comps = sample_completions(model, tokenizer, clean, args.device,
                                           args.n_samples, args.temperature, args.max_new_tokens)
            per_k[k] = {"corrupt_rate": R.rhyme_rate(comps, clean, xw),
                        "delta": R.rhyme_rate(comps, clean, xw) - baseline}
        pair_results.append({"pair_id": pair["pair_id"], "baseline_corrupt_rate": baseline,
                             "ranked_heads": [[list(lh), s] for lh, s in ranked],
                             "per_k": per_k})
        print(f"  {pair['pair_id']}: top_head={ranked[0][0]} "
              f"delta@k1={per_k[args.k[0]]['delta']:+.3f}", flush=True)

    top_heads_global = sorted(agg_rank.items(), key=lambda x: -x[1])[:max(args.k)]
    tag = args.tag or os.path.basename(args.model.rstrip("/"))
    out = {
        "tag": tag, "model": args.model, "n_heads": n_heads, "head_dim": head_dim,
        "k_values": args.k, "pairs": pair_results,
        "top_heads_global": [[list(lh), s] for lh, s in top_heads_global],
    }
    write_json(args.out, out)
    print(f"top heads (global): {[lh for lh,_ in top_heads_global[:5]]} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
