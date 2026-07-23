"""Shared evaluation helpers: checkpoint loading and batched generation."""
from __future__ import annotations

import os
import sys
from typing import List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi.models import load_model  # noqa: E402


def load_for_eval(path_or_name: str, device: str = "cuda:0"):
    """Load a checkpoint dir or HF model id for evaluation on one device."""
    model, tokenizer, adapter = load_model(path_or_name, device_map=None)
    model.to(device)
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"  # correct alignment for batched generation
    return model, tokenizer, adapter


@torch.no_grad()
def generate_completions(model, tokenizer, prompts: List[str], device: str,
                         n_samples: int = 1, temperature: float = 0.0,
                         max_new_tokens: int = 24, batch_size: int = 64) -> List[List[str]]:
    """Return ``[[continuation, ...] per prompt]`` (continuations only).

    Greedy when ``temperature==0`` (forces ``n_samples=1``); otherwise draws
    ``n_samples`` sampled completions per prompt. Left-pads within a batch.
    """
    do_sample = temperature > 0.0
    k = n_samples if do_sample else 1
    pad_id = tokenizer.pad_token_id
    results: List[List[str]] = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        padding_side="left").to(device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_return_sequences=k, pad_token_id=pad_id,
        )
        plen = enc["input_ids"].shape[1]
        gen = gen.view(len(chunk), k, -1)
        for i in range(len(chunk)):
            comps = [tokenizer.decode(gen[i, j, plen:], skip_special_tokens=True)
                     for j in range(k)]
            results.append(comps)
    return results
