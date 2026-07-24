#!/usr/bin/env bash
# Mechanistic "why" for the parameter-vs-function dissociation: sweep the
# on-policy rollout temperature (0.3/0.7/1.2). Higher temp = more diverse
# student-visited states = "more on-policy". Tests whether the weight-churn
# (and function-preservation) scales with how on-policy the training states are.
# From base 4B, teacher 27B, 100 steps. Measures param-update norm + Δ_newline.
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
BASE=google/gemma-3-4b-it
LOG=~/opi_logs

train() {  # $1 temp_tag  $2 temp  $3 gpu_pair
  CUDA_VISIBLE_DEVICES=$3 python -m training.train --condition onpolicy_kd \
    --student_model "$BASE" --teacher_model google/gemma-3-27b-it \
    --student_device cuda:0 --teacher_device cuda:1 \
    --rollout_temp "$2" --seed 0 --max_steps 100 --out_root "runs_temp_${1}" \
    > "$LOG/train_temp_${1}.log" 2>&1
}

train t03 0.3 0,1 & train t07 0.7 2,3 & wait
train t12 1.2 0,1
echo "=== eval ==="
for tag in t03 t07 t12; do
  python -m eval.param_drift --trained runs_temp_${tag}/onpolicy_kd_seed0/ckpt_100 \
    --base "$BASE" --tag tempsweep_${tag} --out results/param_drift/tempsweep_${tag}.json \
    > "$LOG/eval_temp_${tag}.log" 2>&1
done
# Δ_newline for each (extract+probe), fanned
mech() {  # $1 tag  $2 path  $3 gpu
  export CUDA_VISIBLE_DEVICES=$3
  python -m mech.extract_activations --model "$2" --prompts data/prompt_pool/train.jsonl \
    --n 1000 --device cuda:0 --out mech/acts/tempsweep_$1_train.pt >> "$LOG/mech_temp_$1.log" 2>&1
  python -m mech.extract_activations --model "$2" --prompts data/prompt_pool/test_id.jsonl \
    --n 500 --device cuda:0 --out mech/acts/tempsweep_$1_val.pt >> "$LOG/mech_temp_$1.log" 2>&1
  python -m mech.probe --train_acts mech/acts/tempsweep_$1_train.pt \
    --val_acts mech/acts/tempsweep_$1_val.pt --device cuda:0 --tag tempsweep_$1 \
    --out results/probe/tempsweep_$1.json >> "$LOG/mech_temp_$1.log" 2>&1
}
mech t03 runs_temp_t03/onpolicy_kd_seed0/ckpt_100 0 &
mech t07 runs_temp_t07/onpolicy_kd_seed0/ckpt_100 1 &
mech t12 runs_temp_t12/onpolicy_kd_seed0/ckpt_100 2 &
wait
echo TEMPSWEEP_DONE
