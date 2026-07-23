"""Online teacher scoring and student rollout generation for the KD conditions.

Both KD conditions score their prefixes with the *same* teacher forward pass, so
they share this module. Student and teacher are the same model family (Gemma-3),
hence the same tokenizer and vocabulary — teacher logits are directly comparable
to student logits position-for-position.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from .common import IGNORE


class OnlineTeacher:
    """Wraps the 27B teacher on its own device for forward-only scoring."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               out_device) -> torch.Tensor:
        ids = input_ids.to(self.device)
        am = attention_mask.to(self.device)
        out = self.model(input_ids=ids, attention_mask=am)
        return out.logits.to(out_device)


@torch.no_grad()
def sample_student_rollouts(model, tokenizer, prompt_batch: List[List[int]],
                            device, max_new_tokens: int, temperature: float):
    """Generate a second line per prompt from the *current* student policy.

    Returns ``(input_ids, attention_mask, labels_mask)`` where ``labels_mask`` is
    True at positions whose next token is a generated (supervised) token, and
    generation of each row stops at the first newline (end of the second line).
    """
    pad_id = tokenizer.pad_token_id
    # left-pad prompts so generation aligns
    maxlen = max(len(p) for p in prompt_batch)
    padded, attn, prompt_lens = [], [], []
    for p in prompt_batch:
        pad = maxlen - len(p)
        padded.append([pad_id] * pad + p)
        attn.append([0] * pad + [1] * len(p))
        prompt_lens.append(len(p))
    input_ids = torch.tensor(padded, device=device)
    attention_mask = torch.tensor(attn, device=device)

    prev_cache = model.config.use_cache
    model.config.use_cache = True   # fast KV-cached decoding for rollouts
    gen = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=pad_id,
    )
    model.config.use_cache = prev_cache
    full = gen  # [B, maxlen + <=max_new_tokens]
    B, L = full.shape
    newline_ids = _newline_token_ids(tokenizer)

    # Build supervised mask over generated region, truncating each row at its
    # first newline (inclusive).
    labels_mask = torch.zeros_like(full, dtype=torch.bool)
    new_attn = full.ne(pad_id).long()
    for b in range(B):
        start = maxlen  # first generated position
        end = L
        for j in range(maxlen, L):
            tok = int(full[b, j].item())
            if tok in newline_ids or "\n" in tokenizer.decode([tok]):
                end = j + 1
                break
        # Mark the *target* tokens [start, end) as supervised; kd_loss shifts this
        # mask internally (mask[:,1:]) to the predictor positions, matching the
        # SFT convention (labels != IGNORE).
        labels_mask[b, start:end] = True
        # zero attention past the truncation so the teacher/student ignore tails
        if end < L:
            new_attn[b, end:] = 0
    return full, new_attn, labels_mask


def _newline_token_ids(tokenizer) -> set:
    ids = set()
    for cand in ["\n", "\n\n"]:
        enc = tokenizer(cand, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def sft_labels_to_mask(labels: torch.Tensor) -> torch.Tensor:
    """Supervised-position mask from SFT labels (True where label != IGNORE)."""
    return labels.ne(IGNORE)
