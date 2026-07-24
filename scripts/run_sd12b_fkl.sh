#!/usr/bin/env bash
# On-policy self-distillation from corpus_sft-12B with FORWARD-KL (mode-covering),
# to reach Model 1's accuracy for the clean "same behavior, different mechanism"
# test. Reverse-KL (v1/v2) under-reached (~0.82); forward-KL covers the teacher's
# rhyming support. Fresh base student, stabilized (eff. batch 16, lr 5e-6).
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
    --kd_reverse 0 --seed "$s" --max_steps 200 --batch_size 2 --grad_accum 8 \
    --lr 5e-6 --rollout_temp 0.7 --out_root runs12b_selfdistill_fkl \
    > ~/opi_logs/train_sd12bfkl_seed${s}.log 2>&1
}

train 0 0,1 & train 1 2,3 & wait
echo SD12BFKL_TRAIN_DONE
