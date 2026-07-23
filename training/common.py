"""Shared training internals: datasets, losses, schedule, checkpointing.

Design (see outline Sections 8, 10 and H4):

  * ``corpus_sft`` / ``teacher_sft`` -> **hard CE** on the target second line.
  * ``teacher_kd`` / ``onpolicy_kd`` -> **soft KD** with an *identical* divergence
    and the *same online teacher*; the only difference is the prefix source
    (fixed teacher trace vs freshly sampled student rollout). That isolation is
    what makes the H4 comparison clean, so we run the teacher online in both KD
    conditions rather than caching one and not the other.

The KD divergence defaults to **reverse KL** ``KL(student || teacher)`` — the
mode-seeking objective used by on-policy distillation and the subject of the H5
diversity analysis. Forward KL is available via ``kd_reverse=False``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IGNORE = -100


# ---------------------------------------------------------------------------
# Tokenization / datasets
# ---------------------------------------------------------------------------

def encode_example(tokenizer, prompt: str, target: str, max_prompt_len: int,
                   max_response_len: int):
    """Return ``(input_ids, labels)`` for a prompt+target pair.

    ``labels`` masks the prompt (IGNORE) so loss falls only on target tokens.
    The prompt keeps the tokenizer's special tokens; the target is appended
    without them.
    """
    p_ids = tokenizer(prompt, add_special_tokens=True).input_ids[:max_prompt_len]
    t_ids = tokenizer(target, add_special_tokens=False).input_ids[:max_response_len]
    input_ids = p_ids + t_ids
    labels = [IGNORE] * len(p_ids) + list(t_ids)
    return input_ids, labels


class SFTDataset(Dataset):
    """Prompt+target examples (used by both hard-CE and off-policy soft-KD).

    Records must expose ``prompt`` and ``target`` fields.
    """

    def __init__(self, records: List[Dict], tokenizer, cfg):
        self.records = records
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        input_ids, labels = encode_example(
            self.tokenizer, r["prompt"], r["target"],
            self.cfg.max_prompt_len, self.cfg.max_response_len,
        )
        return {"input_ids": input_ids, "labels": labels}


class PromptDataset(Dataset):
    """Prompt-only examples (used by on-policy KD; targets are sampled online)."""

    def __init__(self, records: List[Dict], tokenizer, cfg):
        self.records = records
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        ids = self.tokenizer(r["prompt"], add_special_tokens=True).input_ids
        return {"input_ids": ids[: self.cfg.max_prompt_len], "prompt": r["prompt"]}


def collate(batch, pad_id: int):
    """Left-to-right pad a batch of ``{input_ids, labels?}`` dicts."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    has_labels = "labels" in batch[0]
    for b in batch:
        ids = b["input_ids"]
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        if has_labels:
            labels.append(b["labels"] + [IGNORE] * pad)
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }
    if has_labels:
        out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def hard_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token cross-entropy with a shifted label mask."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE,
    )


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            labels_mask: torch.Tensor, temperature: float = 1.0,
            reverse: bool = True) -> torch.Tensor:
    """Token-level KD divergence at supervised positions.

    ``labels_mask`` is True at positions whose *next* token is supervised (i.e.
    the second-line positions). Both logits are for the same input sequence, so
    position ``t`` predicts token ``t+1``; we align to the shifted supervised
    mask.

    reverse=True  -> KL(student || teacher)  (mode-seeking; default)
    reverse=False -> KL(teacher || student)  (standard forward KD)
    """
    T = temperature
    s = F.log_softmax(student_logits[:, :-1, :].float() / T, dim=-1)
    t = F.log_softmax(teacher_logits[:, :-1, :].float() / T, dim=-1)
    mask = labels_mask[:, 1:]  # positions whose target token is supervised

    if reverse:
        # sum_v p_s (log p_s - log p_t)
        p_s = s.exp()
        kl = (p_s * (s - t)).sum(-1)
    else:
        # sum_v p_t (log p_t - log p_s)
        p_t = t.exp()
        kl = (p_t * (t - s)).sum(-1)

    kl = kl * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return (T * T) * kl.sum() / denom


# ---------------------------------------------------------------------------
# Optimizer / schedule
# ---------------------------------------------------------------------------

def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )


def lr_lambda(step: int, total: int, warmup_frac: float):
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, cfg.max_steps, cfg.warmup_frac)
    )


# ---------------------------------------------------------------------------
# Longitudinal checkpoint schedule
# ---------------------------------------------------------------------------

def checkpoint_steps(fractions: List[float], max_steps: int) -> Dict[int, str]:
    """Map absolute step -> checkpoint tag (``ckpt_000``, ``ckpt_010`` ...)."""
    out = {}
    for f in fractions:
        step = int(round(f * max_steps))
        tag = f"ckpt_{int(round(f * 100)):03d}"
        out[step] = tag
    return out
