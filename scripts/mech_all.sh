#!/usr/bin/env bash
# Mechanistic analysis for one seed: extract activations (probe-train + probe-val),
# train family probes, and run activation patching + handoff detection for the
# base student, the teacher, and each condition's ckpt_100. Fanned across 4 GPUs.
#
#   scripts/mech_all.sh <seed>
set -euo pipefail
SEED=${1:-0}
cd "$(dirname "$0")/.."
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
LOG=~/opi_logs
PTRAIN=data/prompt_pool/train.jsonl
PVAL=data/prompt_pool/test_id.jsonl

mech_one() {  # $1 model  $2 tag  $3 gpu
  local model=$1 tag=$2 gpu=$3 dev=cuda:0
  export CUDA_VISIBLE_DEVICES=$gpu
  python -m mech.extract_activations --model "$model" --prompts "$PTRAIN" --n 1500 \
    --device $dev --out mech/acts/${tag}_train.pt >> "$LOG/mech_${tag}.log" 2>&1
  python -m mech.extract_activations --model "$model" --prompts "$PVAL" --n 800 \
    --device $dev --out mech/acts/${tag}_val.pt   >> "$LOG/mech_${tag}.log" 2>&1
  python -m mech.probe --train_acts mech/acts/${tag}_train.pt --val_acts mech/acts/${tag}_val.pt \
    --device $dev --tag "$tag" --out results/probe/${tag}.json >> "$LOG/mech_${tag}.log" 2>&1
  python -m mech.activation_patching --model "$model" --tag "$tag" --device $dev \
    --out results/patching/${tag}.json >> "$LOG/mech_${tag}.log" 2>&1
  echo "   done mech $tag (gpu $gpu)"
}

# base student + 4 conditions on gpus 0..3,0 ; teacher on gpu 1 in parallel
mech_one google/gemma-3-4b-it base 0 &
mech_one runs/corpus_sft_seed${SEED}/ckpt_100  corpus_sft_seed${SEED}_ckpt_100  1 &
mech_one runs/teacher_sft_seed${SEED}/ckpt_100 teacher_sft_seed${SEED}_ckpt_100 2 &
mech_one runs/teacher_kd_seed${SEED}/ckpt_100  teacher_kd_seed${SEED}_ckpt_100  3 &
wait
mech_one runs/onpolicy_kd_seed${SEED}/ckpt_100 onpolicy_kd_seed${SEED}_ckpt_100 0 &
# teacher patching (27B) for inheritance depth comparison
CUDA_VISIBLE_DEVICES=1 python -m mech.activation_patching --model google/gemma-3-27b-it \
  --tag base_teacher --device cuda:0 --out results/patching/base_teacher.json \
  >> "$LOG/mech_teacher.log" 2>&1 &
wait
echo "All mech done for seed=$SEED."
