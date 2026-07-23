#!/usr/bin/env bash
# On-policy 12B self-distillation, v2 — stabilized to match Model 1's rhyme.
# Fix for the batch-2 gradient-variance blowup seen in v1 (val-rhyme oscillated
# 0.045-0.85): larger effective batch via grad-accum, lower LR, lower rollout
# temperature, more steps. 2 seeds, 2 GPUs each (student + teacher).
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false

train() {  # $1 seed  $2 gpu_pair
  local s=$1 g=$2
  CUDA_VISIBLE_DEVICES=$g python -m training.train --condition onpolicy_kd \
    --student_model google/gemma-3-12b-it \
    --teacher_model "runs12b/corpus_sft_seed${s}/ckpt_100" \
    --student_device cuda:0 --teacher_device cuda:1 --opt8bit \
    --seed "$s" --max_steps 160 --batch_size 2 --grad_accum 8 \
    --lr 5e-6 --rollout_temp 0.7 --out_root runs12b_selfdistill_v2 \
    > ~/opi_logs/train_sd12bv2_seed${s}.log 2>&1
}

train 0 0,1 & train 1 2,3 & wait
echo SD12BV2_TRAIN_DONE
