#!/usr/bin/env bash
# On-policy 12B self-distillation: fresh Gemma-3-12B student distilled (reverse-KL,
# on-policy rollouts) from the seed-matched corpus_sft-12B teacher (Model 1).
# Student full-FT on one GPU (8-bit AdamW), teacher on a second GPU. Two seeds
# run concurrently (2 GPUs each), then the third.
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false

train() {  # $1 seed  $2 gpu_pair (e.g. 0,1)
  local s=$1 g=$2
  CUDA_VISIBLE_DEVICES=$g python -m training.train --condition onpolicy_kd \
    --student_model google/gemma-3-12b-it \
    --teacher_model "runs12b/corpus_sft_seed${s}/ckpt_100" \
    --student_device cuda:0 --teacher_device cuda:1 --opt8bit \
    --seed "$s" --max_steps 150 --batch_size 2 --out_root runs12b_selfdistill \
    > ~/opi_logs/train_sd12b_seed${s}.log 2>&1
}

train 0 0,1 & train 1 2,3 & wait
train 2 0,1 & wait
echo SD12B_TRAIN_DONE
