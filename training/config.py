"""Training configuration shared by all four regimes.

Everything that must be *matched across conditions* (outline Section 6) lives
here so the only intended differences between runs are the prefix source and the
loss. Per-condition specifics are captured by ``CONDITIONS``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# Fractions of training at which longitudinal checkpoints are saved (Section 10.3).
CHECKPOINT_FRACTIONS: List[float] = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]


@dataclass
class TrainConfig:
    # --- matched across every condition ---
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_frac: float = 0.03
    max_grad_norm: float = 1.0
    batch_size: int = 16
    grad_accum: int = 2                 # effective batch = 32
    max_steps: int = 600                # matched-compute budget (updates)
    max_prompt_len: int = 48
    max_response_len: int = 24
    seed: int = 0
    dtype: str = "bfloat16"
    grad_checkpointing: bool = True
    # --- KD / on-policy specifics (only used by KD conditions) ---
    kd_temperature: float = 1.0         # softmax temp for teacher/student in KD
    rollout_temperature: float = 1.0    # on-policy sampling temperature
    kd_top_k: int = 64                  # top-K teacher distribution used in KD
    onpolicy_refresh_every: int = 1     # regenerate student rollouts every N steps
    # --- bookkeeping ---
    eval_every: int = 100
    eval_n_prompts: int = 200
    log_every: int = 20


@dataclass
class Condition:
    name: str
    prefix_source: str      # "corpus" | "teacher" | "student"
    objective: str          # "hard_ce" | "soft_kd"
    data: str               # path key resolved by the trainer
    description: str
    needs_online_teacher: bool = False


CONDITIONS = {
    "corpus_sft": Condition(
        name="corpus_sft",
        prefix_source="corpus",
        objective="hard_ce",
        data="data/corpus/corpus_sft.jsonl",
        description="Generic static SFT on existing (non-teacher) poems.",
    ),
    "teacher_sft": Condition(
        name="teacher_sft",
        prefix_source="teacher",
        objective="hard_ce",
        data="data/teacher_traces/teacher_sft.jsonl",
        description="Off-policy distillation: hard SFT on fixed teacher rollouts.",
    ),
    "teacher_kd": Condition(
        name="teacher_kd",
        prefix_source="teacher",
        objective="soft_kd",
        data="data/teacher_traces/teacher_sft.jsonl",
        description="Off-policy soft KD: online teacher reverse-KL on fixed teacher prefixes.",
    ),
    "onpolicy_kd": Condition(
        name="onpolicy_kd",
        prefix_source="student",
        objective="soft_kd",
        data="data/prompt_pool/train.jsonl",
        description="On-policy soft KD: teacher scores student-generated states.",
        needs_online_teacher=True,
    ),
}
