#!/usr/bin/env python3
"""Unified trainer for the four regimes (outline Sections 8, 10).

    python -m training.train --condition corpus_sft  --student_device cuda:0
    python -m training.train --condition teacher_sft --student_device cuda:1
    python -m training.train --condition teacher_kd  --student_device cuda:0 --teacher_device cuda:1
    python -m training.train --condition onpolicy_kd --student_device cuda:2 --teacher_device cuda:3

All conditions share the same student init, optimizer, schedule, batch size, and
update budget (``training/config.py``); only the prefix source and loss differ.
Longitudinal checkpoints are saved at 0/10/25/50/75/100% of training.
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opi import rhyme as R  # noqa: E402
from opi.models import load_model, STUDENT_MODEL, TEACHER_MODEL  # noqa: E402
from opi.io_utils import read_jsonl, write_json, set_seed  # noqa: E402

from training.config import TrainConfig, CONDITIONS, CHECKPOINT_FRACTIONS  # noqa: E402
from training import common as C  # noqa: E402
from training import online as O  # noqa: E402


@torch.no_grad()
def quick_rhyme_eval(model, tokenizer, val_records, device, n, max_new_tokens):
    """Greedy rhyme rate on ``n`` val prompts (training-time signal)."""
    model.eval()
    hits = tot = 0
    for r in val_records[:n]:
        enc = tokenizer(r["prompt"], return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        cont = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        sw = R.extract_second_line_word_with_newline(cont)
        res = R.do_rhyme(r["rhyme_word"], sw or "")
        tot += 1
        hits += 1 if res is True else 0
    model.train()
    return hits / max(1, tot)


def save_ckpt(model, tokenizer, out_dir, tag):
    path = os.path.join(out_dir, tag)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--student_model", default=STUDENT_MODEL)
    ap.add_argument("--teacher_model", default=TEACHER_MODEL)
    ap.add_argument("--student_device", default="cuda:0")
    ap.add_argument("--teacher_device", default="cuda:1")
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--kd_reverse", type=int, default=1, help="1=reverse KL, 0=forward KL")
    ap.add_argument("--val", default="data/prompt_pool/val.jsonl")
    args = ap.parse_args()

    cond = CONDITIONS[args.condition]
    cfg = TrainConfig(seed=args.seed)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps

    set_seed(cfg.seed)
    out_dir = os.path.join(args.out_root, f"{cond.name}_seed{cfg.seed}")
    os.makedirs(out_dir, exist_ok=True)
    device = args.student_device
    dtype = torch.bfloat16

    print(f"=== condition={cond.name} seed={cfg.seed} device={device} "
          f"steps={cfg.max_steps} obj={cond.objective} ===", flush=True)

    # --- student ---
    student, tokenizer, adapter = load_model(args.student_model, dtype=dtype, for_training=True)
    student.to(device)
    if cfg.grad_checkpointing:
        student.gradient_checkpointing_enable()
    student.config.use_cache = False
    student.train()

    # --- teacher (KD only) ---
    teacher = None
    if cond.needs_online_teacher or cond.objective == "soft_kd":
        print(f"Loading online teacher {args.teacher_model} on {args.teacher_device}",
              flush=True)
        tmodel, _, _ = load_model(args.teacher_model, dtype=dtype, device_map=None)
        tmodel.to(args.teacher_device)
        teacher = O.OnlineTeacher(tmodel, args.teacher_device)

    # --- data ---
    val_records = read_jsonl(args.val)
    if cond.prefix_source == "student":
        records = read_jsonl(cond.data)
        ds = C.PromptDataset(records, tokenizer, cfg)
        collate = lambda b: {
            "input_ids": [x["input_ids"] for x in b],
        }
    else:
        records = read_jsonl(cond.data)
        ds = C.SFTDataset(records, tokenizer, cfg)
        collate = lambda b: C.collate(b, tokenizer.pad_token_id)

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)
    print(f"Dataset: {len(ds)} examples | {len(loader)} batches/epoch", flush=True)

    optim = C.build_optimizer(student, cfg)
    sched = C.build_scheduler(student, cfg)
    ckpt_map = C.checkpoint_steps(CHECKPOINT_FRACTIONS, cfg.max_steps)

    history = {"condition": cond.name, "seed": cfg.seed, "steps": [], "loss": [],
               "val_rhyme": [], "checkpoints": []}
    reverse = bool(args.kd_reverse)

    def maybe_ckpt(step):
        if step in ckpt_map:
            tag = ckpt_map[step]
            p = save_ckpt(student, tokenizer, out_dir, tag)
            vr = quick_rhyme_eval(student, tokenizer, val_records, device,
                                  cfg.eval_n_prompts, cfg.max_response_len)
            history["checkpoints"].append({"step": step, "tag": tag, "val_rhyme": vr})
            print(f"  [ckpt {tag}] step={step} val_rhyme={vr:.3f} -> {p}", flush=True)

    maybe_ckpt(0)  # initial checkpoint
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    optim.zero_grad()
    while step < cfg.max_steps:
        accum_loss = 0.0
        for micro in range(cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            if cond.objective == "hard_ce":
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = student(input_ids=input_ids, attention_mask=attn)
                loss = C.hard_ce_loss(out.logits, labels)

            else:  # soft_kd
                if cond.prefix_source == "student":
                    prompts = batch["input_ids"]
                    full, attn, lmask = O.sample_student_rollouts(
                        student, tokenizer, prompts, device,
                        cfg.max_response_len, cfg.rollout_temperature)
                    input_ids = full
                else:
                    input_ids = batch["input_ids"].to(device)
                    attn = batch["attention_mask"].to(device)
                    lmask = O.sft_labels_to_mask(batch["labels"].to(device))

                s_out = student(input_ids=input_ids, attention_mask=attn)
                t_logits = teacher.logits(input_ids, attn, out_device=device)
                loss = C.kd_loss(s_out.logits, t_logits, lmask,
                                 temperature=cfg.kd_temperature, reverse=reverse)

            (loss / cfg.grad_accum).backward()
            accum_loss += loss.item() / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
        optim.step()
        sched.step()
        optim.zero_grad()
        step += 1

        if step % cfg.log_every == 0:
            rate = step / max(1e-9, time.time() - t0)
            print(f"  step {step}/{cfg.max_steps} loss={accum_loss:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({rate:.2f} it/s)", flush=True)
            history["steps"].append(step)
            history["loss"].append(accum_loss)
        maybe_ckpt(step)

    write_json(os.path.join(out_dir, "history.json"), history)
    print(f"Done {cond.name} seed{cfg.seed}. Final val_rhyme="
          f"{history['checkpoints'][-1]['val_rhyme']:.3f}", flush=True)


if __name__ == "__main__":
    main()
