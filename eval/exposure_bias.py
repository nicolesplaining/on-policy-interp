#!/usr/bin/env python3
"""Exposure bias as an internal state-calibration difference (on- vs off-policy).

The canonical claim (Thinking Machines, GKD) is behavioral: off-policy students
are trained on *teacher* states, so at their *own* inference states — which they
never saw in training — they diverge from the teacher, and errors accumulate.
No prior work measures this *internally*. Here we do:

For a student, compute the mean next-token KL to the 27B teacher over the
second-line tokens of two sequences:
  * teacher states : prompt + the *teacher's* second line
  * student states : prompt + the *student's own* generated second line

    exposure_gap = KL(at student states) - KL(at teacher states)

Prediction: an off-policy (teacher-trace) student agrees with the teacher at
teacher states but diverges at its own states -> large positive gap. An on-policy
student, trained on its own states, has a small gap. A direct mechanistic
signature of exposure bias.

Usage:
  python -m eval.exposure_bias --student runs/onpolicy_kd_seed0/ckpt_100 \
      --teacher google/gemma-3-27b-it --student_device cuda:0 --teacher_device cuda:1 \
      --tag onpolicy_kd_seed0 --out results/exposure/onpolicy_kd_seed0.json
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opi.models import load_model  # noqa: E402
from opi.io_utils import read_jsonl, write_json  # noqa: E402


@torch.no_grad()
def seq_kl(student, teacher_model, tok, prompt, second_line, s_dev, t_dev):
    """Mean KL(student||teacher) over the second-line token positions of
    prompt+second_line (teacher-forced)."""
    p_ids = tok(prompt, add_special_tokens=True).input_ids
    t_ids = tok(second_line, add_special_tokens=False).input_ids
    if not t_ids:
        return None
    ids = torch.tensor([p_ids + t_ids])
    s_logits = student(input_ids=ids.to(s_dev)).logits[0].float()
    t_logits = teacher_model(input_ids=ids.to(t_dev)).logits[0].float().to(s_dev)
    # positions predicting the second-line tokens: [len(p)-1 .. len(p)+len(t)-2]
    lo, hi = len(p_ids) - 1, len(p_ids) + len(t_ids) - 1
    s_lp = F.log_softmax(s_logits[lo:hi], -1)
    t_lp = F.log_softmax(t_logits[lo:hi], -1)
    kl = (s_lp.exp() * (s_lp - t_lp)).sum(-1)  # KL(student||teacher) per pos
    return kl.mean().item()


@torch.no_grad()
def gen_second_line(student, tok, prompt, s_dev, max_new_tokens=24):
    enc = tok(prompt, return_tensors="pt").to(s_dev)
    prev = student.config.use_cache
    student.config.use_cache = True
    out = student.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    student.config.use_cache = prev
    cont = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return cont.split("\n")[0] + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="google/gemma-3-27b-it")
    ap.add_argument("--teacher_traces", default="data/teacher_traces/teacher_sft.jsonl")
    ap.add_argument("--student_device", default="cuda:0")
    ap.add_argument("--teacher_device", default="cuda:1")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    student, tok, _ = load_model(args.student, device_map=None)
    student.to(args.student_device); student.eval()
    # teacher 27B sharded across the teacher device(s)
    teacher_model, _, _ = load_model(args.teacher, device_map="auto")
    t_dev = next(teacher_model.parameters()).device

    traces = read_jsonl(args.teacher_traces)[: args.n_prompts]
    kl_teacher_states, kl_student_states, n = 0.0, 0.0, 0
    for r in traces:
        prompt, teacher_line = r["prompt"], r["target"]
        if not teacher_line.strip():
            continue
        student_line = gen_second_line(student, tok, prompt, args.student_device)
        klt = seq_kl(student, teacher_model, tok, prompt, teacher_line, args.student_device, t_dev)
        kls = seq_kl(student, teacher_model, tok, prompt, student_line, args.student_device, t_dev)
        if klt is None or kls is None:
            continue
        kl_teacher_states += klt; kl_student_states += kls; n += 1

    kt, ks = kl_teacher_states / max(1, n), kl_student_states / max(1, n)
    out = {"tag": args.tag or os.path.basename(args.student.rstrip("/")),
           "student": args.student, "n": n,
           "kl_at_teacher_states": kt, "kl_at_student_states": ks,
           "exposure_gap": ks - kt}
    write_json(args.out, out)
    print(f"{out['tag']}: KL@teacher_states={kt:.3f}  KL@student_states={ks:.3f}  "
          f"exposure_gap={ks - kt:+.3f}", flush=True)


if __name__ == "__main__":
    main()
