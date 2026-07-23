#!/usr/bin/env python3
"""Phase 2: generate teacher traces with Gemma-3-27B.

For each shared train prompt this produces, in one pass:

  * a **hard** second line (sampled teacher tokens)  -> ``teacher_sft`` target
  * cached **soft** targets: the teacher's top-K logprobs at every second-line
    position of that trace -> ``teacher_kd`` targets (off-policy soft KD)

Off-policy soft KD uses *fixed* teacher prefixes, so caching the teacher
distribution here avoids re-running the 27B every training step. Storing the
full 262k-vocab distribution is infeasible, so we keep the top-K ids + logprobs
per position (standard KD practice); the training loss renormalizes over the
top-K.

Sharding: run ``--num_shards 4`` with ``--shard 0..3`` (one GPU each, via
``CUDA_VISIBLE_DEVICES``) to use all four H100s.

Outputs (per shard):
  data/teacher_traces/teacher_sft.shard{S}.jsonl
  data/teacher_traces/teacher_soft.shard{S}.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opi import rhyme as R  # noqa: E402
from opi.models import load_model, TEACHER_MODEL  # noqa: E402
from opi.io_utils import read_jsonl, write_jsonl  # noqa: E402


@torch.no_grad()
def generate_and_score(model, tokenizer, adapter, prompt, max_new_tokens,
                       temperature, top_k, want_soft=True):
    """Sample a second line and (optionally) cache teacher top-K logprobs.

    Returns ``(target_text, target_ids, topk_ids[T,K], topk_logprobs[T,K], second_word)``.
    With ``want_soft=False`` the extra teacher-forced forward pass is skipped
    (much faster); the soft tensors are returned empty. Training uses only the
    hard target text, since both KD conditions score prefixes with the online
    teacher; the cached soft targets are for optional probe/analysis use.
    """
    device = adapter.get_input_device()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    full_ids = gen[0]
    gen_ids = full_ids[prompt_len:]

    # Truncate the trace at the end of the second line: keep tokens up to and
    # including the first one that contains a newline. Teacher generation often
    # continues past the couplet, and we only want to train on the second line.
    cut = gen_ids.shape[0]
    for j in range(gen_ids.shape[0]):
        if "\n" in tokenizer.decode(gen_ids[j:j + 1]):
            cut = j + 1
            break
    gen_ids = gen_ids[:cut]
    full_ids = full_ids[: prompt_len + cut]
    continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
    second_word = R.extract_second_line_word_with_newline(continuation)

    if not want_soft:
        target_ids = [int(t.item()) for t in gen_ids]
        empty_i = torch.zeros(0, top_k, dtype=torch.int32)
        empty_f = torch.zeros(0, top_k, dtype=torch.float16)
        return continuation, target_ids, empty_i, empty_f, second_word

    # Teacher-forced forward over the (truncated) sequence to get per-position logits.
    out = model(full_ids.unsqueeze(0))
    logits = out.logits[0]  # [seq, vocab]
    # logits[t] predicts token t+1; the distribution for generated token at
    # absolute position p is logits[p-1].
    logprobs_all = torch.log_softmax(logits.float(), dim=-1)

    topk_ids_rows, topk_lp_rows, target_ids = [], [], []
    for j in range(gen_ids.shape[0]):
        p = prompt_len + j
        row = logprobs_all[p - 1]
        vals, idx = row.topk(top_k)
        topk_ids_rows.append(idx.to(torch.int32).cpu())
        topk_lp_rows.append(vals.to(torch.float16).cpu())
        target_ids.append(int(gen_ids[j].item()))

    topk_ids = torch.stack(topk_ids_rows) if topk_ids_rows else torch.zeros(0, top_k, dtype=torch.int32)
    topk_lp = torch.stack(topk_lp_rows) if topk_lp_rows else torch.zeros(0, top_k, dtype=torch.float16)
    return continuation, target_ids, topk_ids, topk_lp, second_word


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/prompt_pool/train.jsonl")
    ap.add_argument("--out_dir", default="data/teacher_traces")
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--no_soft", action="store_true",
                    help="Skip cached top-K soft targets (much faster; training "
                         "uses the online teacher, so soft targets are optional).")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    prompts = read_jsonl(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    prompts = prompts[args.shard:: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(prompts)} prompts", flush=True)

    print(f"Loading teacher {args.model} ...", flush=True)
    model, tokenizer, adapter = load_model(args.model, device_map="auto")
    print(f"Loaded on {adapter.get_input_device()} "
          f"(layers={adapter.get_n_layers()})", flush=True)

    hard_records, soft_records = [], []
    n_valid = 0
    for i, rec in enumerate(prompts):
        prompt = rec["prompt"]
        try:
            cont, target_ids, topk_ids, topk_lp, second_word = generate_and_score(
                model, tokenizer, adapter, prompt,
                args.max_new_tokens, args.temperature, args.top_k,
                want_soft=not args.no_soft,
            )
        except Exception as e:  # keep the shard alive on a bad example
            print(f"  [{i}] error: {e}", flush=True)
            continue

        rhymes = R.do_rhyme(rec["rhyme_word"], second_word or "")
        hard_records.append({
            "id": rec["id"],
            "prompt": prompt,
            "target": cont,
            "rhyme_word": rec["rhyme_word"],
            "second_word": second_word,
            "rhyme_family": rec["rhyme_family"],
            "teacher_rhymes": bool(rhymes),
        })
        soft_records.append({
            "id": rec["id"],
            "prompt": prompt,
            "target_ids": target_ids,
            "topk_ids": topk_ids,
            "topk_logprobs": topk_lp,
        })
        n_valid += 1
        if (i + 1) % args.log_every == 0:
            rate = sum(h["teacher_rhymes"] for h in hard_records) / len(hard_records)
            print(f"  [{i+1}/{len(prompts)}] running teacher rhyme rate={rate:.3f}",
                  flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    hard_path = os.path.join(args.out_dir, f"teacher_sft.shard{args.shard}.jsonl")
    soft_path = os.path.join(args.out_dir, f"teacher_soft.shard{args.shard}.pt")
    write_jsonl(hard_path, hard_records)
    torch.save(soft_records, soft_path)
    rate = (sum(h["teacher_rhymes"] for h in hard_records) / len(hard_records)
            if hard_records else 0.0)
    print(f"[shard {args.shard}] wrote {n_valid} traces | teacher rhyme rate={rate:.3f}",
          flush=True)
    print(f"  {hard_path}\n  {soft_path}", flush=True)


if __name__ == "__main__":
    main()
