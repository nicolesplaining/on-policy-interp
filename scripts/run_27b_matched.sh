#!/usr/bin/env bash
# MATCHED 27B circuit test (fixes the confound in run_27b_circuit.sh). Both
# conditions: FORWARD-KL (mode-covering, stabilizes on-policy), SAME lr 1e-5,
# SAME effective batch 16, SAME 80 steps -> only difference is prefix source
# (student rollouts vs fixed teacher traces). Then patch + measure weight motion.
# If on-policy preserves the 27B handoff while off-policy destroys it AT MATCHED
# weight-motion, that's a real circuit-level result. If both destroy it, honest null.
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
    --teacher_device cuda:0 --seed 0 --max_steps 80 --batch_size 2 --grad_accum 8 \
    --lr 1e-5 --kd_reverse 0 --rollout_temp 0.7 --max_response_len 16 \
    --ckpt_fracs 1 --log_every 10 --out_root "$2" > "$LOG/train_27bm_${1}.log" 2>&1
}
patch() {  # $1 model_path  $2 tag
  CUDA_VISIBLE_DEVICES=0 python -m mech.activation_patching --model "$1" --tag "$2" \
    --device cuda:0 --n_samples 16 --out results/patching/${2}.json \
    > "$LOG/patch_${2}.log" 2>&1
}

echo "=== train on-policy (fwd-KL, matched) ==="; train onpolicy_kd runs27bm_onp
echo "=== train off-policy (fwd-KL, matched) ==="; train teacher_kd  runs27bm_off
echo "=== patch both ==="
patch runs27bm_onp/onpolicy_kd_seed0/ckpt_100 matched27b_onpolicy
patch runs27bm_off/teacher_kd_seed0/ckpt_100  matched27b_offpolicy
echo "=== weight motion ==="
python -m eval.param_drift --trained runs27bm_onp/onpolicy_kd_seed0/ckpt_100 --base google/gemma-3-27b-it --tag driftm_onp --out results/param_drift/driftm_onp.json 2>&1 | grep total_update
python -m eval.param_drift --trained runs27bm_off/teacher_kd_seed0/ckpt_100 --base google/gemma-3-27b-it --tag driftm_off --out results/param_drift/driftm_off.json 2>&1 | grep total_update
echo MATCHED27B_DONE
