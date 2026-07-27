#!/usr/bin/env bash
# Seed-1 confirmation of the matched 27B circuit result (fig17). Same matched
# setup (forward-KL, lr 1e-5, eff batch 16, 80 steps), different rollout/data seed.
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
TEA=runs/corpus_sft_seed0/ckpt_100
LOG=~/opi_logs

train() {  # $1 condition  $2 out_root
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m training.train --condition "$1" \
    --student_model google/gemma-3-27b-it --teacher_model "$TEA" \
    --shard_student --student_gpus 0,1,2,3 --shard_mem_gib 60 --opt8bit \
    --teacher_device cuda:0 --seed 1 --max_steps 80 --batch_size 2 --grad_accum 8 \
    --lr 1e-5 --kd_reverse 0 --rollout_temp 0.7 --max_response_len 16 \
    --ckpt_fracs 1 --log_every 20 --out_root "$2" > "$LOG/train_27bm_s1_${1}.log" 2>&1
}
patch() {
  CUDA_VISIBLE_DEVICES=0 python -m mech.activation_patching --model "$1" --tag "$2" \
    --device cuda:0 --n_samples 16 --out results/patching/${2}.json > "$LOG/patch_${2}.log" 2>&1
}
train onpolicy_kd runs27bm_onp_s1
train teacher_kd  runs27bm_off_s1
patch runs27bm_onp_s1/onpolicy_kd_seed1/ckpt_100 matched27b_onpolicy_s1
patch runs27bm_off_s1/teacher_kd_seed1/ckpt_100  matched27b_offpolicy_s1
python -m eval.param_drift --trained runs27bm_onp_s1/onpolicy_kd_seed1/ckpt_100 --base google/gemma-3-27b-it --tag driftm_onp_s1 --out results/param_drift/driftm_onp_s1.json 2>&1 | grep total_update
python -m eval.param_drift --trained runs27bm_off_s1/teacher_kd_seed1/ckpt_100 --base google/gemma-3-27b-it --tag driftm_off_s1 --out results/param_drift/driftm_off_s1.json 2>&1 | grep total_update
echo MATCHED27B_S1_DONE
