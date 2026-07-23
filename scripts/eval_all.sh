#!/usr/bin/env bash
# Behavioral + diversity + forgetting + parameter-drift eval for one seed's
# ckpt_100 across the four conditions (plus base), fanned out over 4 GPUs.
#
#   scripts/eval_all.sh <seed>
set -euo pipefail
SEED=${1:-0}
cd "$(dirname "$0")/.."
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
LOG=~/opi_logs
BASE=google/gemma-3-4b-it
CONDS=(corpus_sft teacher_sft teacher_kd onpolicy_kd)

run_one() {  # $1 condition  $2 gpu
  local cond=$1 gpu=$2 ck=runs/${1}_seed${SEED}/ckpt_100 tag=${1}_ckpt_100
  local dev=cuda:0
  CUDA_VISIBLE_DEVICES=$gpu python -m eval.behavioral --model "$ck" --tag "$tag" --device $dev \
    --out results/behavioral/${tag}.json >> "$LOG/eval_${cond}_s${SEED}.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu python -m eval.diversity --model "$ck" --tag "$tag" --device $dev \
    --out results/diversity/${tag}.json >> "$LOG/eval_${cond}_s${SEED}.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu python -m eval.forgetting --trained "$ck" --base "$BASE" --tag "$tag" --device $dev \
    --out results/forgetting/${tag}.json >> "$LOG/eval_${cond}_s${SEED}.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu python -m eval.param_drift --trained "$ck" --base "$BASE" --tag "$tag" \
    --out results/param_drift/${tag}.json >> "$LOG/eval_${cond}_s${SEED}.log" 2>&1
  echo "   done $cond (gpu $gpu)"
}

# base behavioral (reference)
CUDA_VISIBLE_DEVICES=0 python -m eval.behavioral --model "$BASE" --tag base --device cuda:0 \
  --out results/behavioral/base.json >> "$LOG/eval_base.log" 2>&1 &

i=0
for cond in "${CONDS[@]}"; do
  run_one "$cond" "$i" &
  i=$((i+1))
done
wait
echo "All evals done for seed=$SEED."
