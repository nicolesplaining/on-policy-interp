#!/usr/bin/env bash
# Train all four conditions for one seed across the 4 H100s.
#
#   scripts/train_all.sh <seed> <max_steps>
#
# Schedule:
#   Wave 1 (KD conditions, each needs an online teacher):
#     onpolicy_kd -> student GPU0 + teacher GPU1
#     teacher_kd  -> student GPU2 + teacher GPU3
#   Wave 2 (SFT conditions, 1 GPU each):
#     corpus_sft  -> GPU0
#     teacher_sft -> GPU1
set -euo pipefail
SEED=${1:-0}
MAX_STEPS=${2:-600}
cd "$(dirname "$0")/.."
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
LOG=~/opi_logs
mkdir -p "$LOG"

echo "== Wave 1: KD conditions (seed=$SEED steps=$MAX_STEPS) =="
CUDA_VISIBLE_DEVICES=0,1 python -m training.train --condition onpolicy_kd \
  --student_device cuda:0 --teacher_device cuda:1 --seed "$SEED" --max_steps "$MAX_STEPS" \
  > "$LOG/train_onpolicy_kd_seed${SEED}.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2,3 python -m training.train --condition teacher_kd \
  --student_device cuda:0 --teacher_device cuda:1 --seed "$SEED" --max_steps "$MAX_STEPS" \
  > "$LOG/train_teacher_kd_seed${SEED}.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "   Wave 1 done."

echo "== Wave 2: SFT conditions =="
CUDA_VISIBLE_DEVICES=0 python -m training.train --condition corpus_sft \
  --student_device cuda:0 --seed "$SEED" --max_steps "$MAX_STEPS" \
  > "$LOG/train_corpus_sft_seed${SEED}.log" 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=1 python -m training.train --condition teacher_sft \
  --student_device cuda:0 --seed "$SEED" --max_steps "$MAX_STEPS" \
  > "$LOG/train_teacher_sft_seed${SEED}.log" 2>&1 &
P4=$!
wait $P3 $P4
echo "   Wave 2 done. All conditions trained for seed=$SEED."
