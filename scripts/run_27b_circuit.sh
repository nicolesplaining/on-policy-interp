#!/usr/bin/env bash
# THE circuit-level test: only the 27B has the causal rhyme-word->newline handoff.
# Distill the 27B (as student) toward the handoff-LESS corpus_sft-4B teacher, on-
# policy vs off-policy, then patch to see whether off-policy DESTROYS the causal
# handoff while on-policy PRESERVES it. Student full-FT (8-bit, sharded across 4
# GPUs); 4B teacher co-located. Endpoint-only checkpoints (avoid 54GB saves).
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
TEA=runs/corpus_sft_seed0/ckpt_100   # handoff-less teacher (rhymes 0.96 via templates)
LOG=~/opi_logs
STEPS=80

train() {  # $1 condition  $2 out_root
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m training.train --condition "$1" \
    --student_model google/gemma-3-27b-it --teacher_model "$TEA" \
    --shard_student --student_gpus 0,1,2,3 --shard_mem_gib 60 --opt8bit \
    --teacher_device cuda:0 --seed 0 --max_steps "$STEPS" --batch_size 2 --grad_accum 2 \
    --max_response_len 16 --ckpt_fracs 1 --log_every 10 --out_root "$2" \
    > "$LOG/train_27bcirc_${1}.log" 2>&1
}

echo "=== train 27B on-policy ==="; train onpolicy_kd runs27b_onp
echo "=== train 27B off-policy ==="; train teacher_kd  runs27b_off

echo "=== patching (handoff) on 27B students ==="
CUDA_VISIBLE_DEVICES=0 python -m mech.activation_patching \
  --model runs27b_onp/onpolicy_kd_seed0/ckpt_100 --tag circuit27b_onpolicy \
  --device cuda:0 --n_samples 16 --out results/patching/circuit27b_onpolicy.json \
  > "$LOG/patch_27bcirc_onp.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python -m mech.activation_patching \
  --model runs27b_off/teacher_kd_seed0/ckpt_100 --tag circuit27b_offpolicy \
  --device cuda:0 --n_samples 16 --out results/patching/circuit27b_offpolicy.json \
  > "$LOG/patch_27bcirc_off.log" 2>&1
echo CIRCUIT27B_DONE
