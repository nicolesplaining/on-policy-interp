#!/usr/bin/env bash
# Phase 2: build the shared prompt pool, the fixed corpus, and the teacher traces
# (4-GPU sharded generation), then merge shards and report distribution matching.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
LOG=~/opi_logs
mkdir -p "$LOG"

python data/build_prompt_pool.py --n_train 10000 --n_val 1000 --n_test_id 1000 \
  --n_test_heldout 1000 --n_recovery 500 --seed 0
python data/build_corpus_sft.py

echo "Launching teacher-trace generation on 4 GPUs..."
# --no_soft: training scores prefixes with the online teacher, so cached top-K
# soft targets are optional. Drop --no_soft to also cache them (much slower) for
# probe target #4 / analysis.
for S in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$S python data/generate_teacher_traces.py \
    --num_shards 4 --shard "$S" --no_soft --log_every 400 > "$LOG/teacher_shard$S.log" 2>&1 &
done
wait

python data/merge_shards.py
python data/match_datasets.py
echo "Phase 2 data complete."
