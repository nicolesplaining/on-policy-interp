#!/usr/bin/env python3
"""Phase 5: extract i-indexed residual activations for probing.

For each prompt the model generates a second line; we record the final rhyme
token of that line (the "future rhyme" being planned) and the residual-stream
activation at every layer at a set of positions relative to the newline (i=0):

    offsets = [-2, -1, 0, 1, 2, 3]  (rhyme word ... newline ... first gen tokens)

Mirrors look-ahead ``extract_poem_dataset.py`` but pins the exact positions used
by the planning-site analysis via ``opi.positions``. Output feeds ``mech.probe``.

Usage:
  python -m mech.extract_activations --model runs/onpolicy_kd_seed0/ckpt_100 \
      --prompts data/prompt_pool/test_id.jsonl --n 800 --device cuda:0 \
      --out mech/acts/onpolicy_kd_ckpt_100.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi import rhyme as R  # noqa: E402
from opi.positions import second_newline_token, standard_probe_offsets  # noqa: E402
from opi.io_utils import read_jsonl  # noqa: E402
from eval.common import load_for_eval  # noqa: E402


@torch.no_grad()
def extract(model, tokenizer, adapter, records, device, offsets, max_new_tokens):
    n_layers = adapter.get_n_layers()
    acts = {off: {L: [] for L in range(n_layers)} for off in offsets}
    target_ids, target_words, kept = [], [], 0

    for r in records:
        prompt = r["prompt"]
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        plen = enc["input_ids"].shape[1]
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        full = gen[0]
        cont = tokenizer.decode(full[plen:], skip_special_tokens=True)
        target_word = R.extract_second_line_word_with_newline(cont)
        if not target_word:
            continue
        # locate the target rhyme token id (last alpha token of the second line)
        tgt_id = None
        for j in range(full.shape[0] - 1, plen - 1, -1):
            piece = tokenizer.decode(full[j:j + 1])
            if any(c.isalpha() for c in piece):
                tgt_id = int(full[j].item())
                break
        if tgt_id is None:
            continue

        base = second_newline_token(tokenizer, prompt)  # i=0 abs index
        positions = {off: base + off for off in offsets}
        if any(p < 0 or p >= full.shape[0] for p in positions.values()):
            continue

        out = model(full.unsqueeze(0), output_hidden_states=True)
        hs = out.hidden_states  # tuple[L+1] of [1, seq, d]
        for off in offsets:
            p = positions[off]
            for L in range(n_layers):
                acts[off][L].append(hs[L + 1][0, p, :].to(torch.float16).cpu())
        target_ids.append(tgt_id)
        target_words.append(target_word)
        kept += 1

    acts = {off: {L: torch.stack(v) for L, v in layers.items()}
            for off, layers in acts.items()}
    return acts, torch.tensor(target_ids), target_words, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="data/prompt_pool/test_id.jsonl")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tokenizer, adapter = load_for_eval(args.model, args.device)
    records = read_jsonl(args.prompts)[: args.n]
    offsets = standard_probe_offsets()
    acts, target_ids, target_words, kept = extract(
        model, tokenizer, adapter, records, args.device, offsets, args.max_new_tokens)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "acts": acts, "target_ids": target_ids, "target_words": target_words,
        "offsets": offsets,
        "meta": {"model": args.model, "n_kept": kept,
                 "n_layers": adapter.get_n_layers(),
                 "d_model": acts[offsets[0]][0].shape[-1],
                 "vocab_size": model.config.get_text_config().vocab_size
                 if hasattr(model.config, "get_text_config") else model.config.vocab_size},
    }, args.out)
    print(f"Extracted {kept}/{len(records)} prompts x {len(offsets)} offsets "
          f"x {adapter.get_n_layers()} layers -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
