#!/usr/bin/env bash
# Matched-performance forgetting comparison (outline Section 10.2).
# Selects each (condition, seed)'s earliest checkpoint reaching a common rhyme
# threshold, then runs forgetting + behavioral on those checkpoints, tagged
# <condition>_seed<seed>_matched. Fans across the 4 GPUs.
#
#   scripts/run_matched.sh [threshold]
set -euo pipefail
THRESH=${1:-0.85}
cd "$(dirname "$0")/.."
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
LOG=~/opi_logs
BASE=google/gemma-3-4b-it

python -m analysis.matched_perf --threshold "$THRESH" --out results/matched/selection.json

# Emit "cond seed tag" lines from the selection manifest.
mapfile -t JOBS < <(python - <<'PY'
import json
sel = json.load(open("results/matched/selection.json"))["selection"]
for c, seeds in sel.items():
    for s, info in seeds.items():
        print(f"{c} {s} {info['tag']}")
PY
)

i=0
for job in "${JOBS[@]}"; do
  read -r cond seed tag <<< "$job"
  gpu=$((i % 4))
  ck="runs/${cond}_seed${seed}/${tag}"
  mtag="${cond}_seed${seed}_matched"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    python -m eval.forgetting --trained "$ck" --base "$BASE" --tag "$mtag" --device cuda:0 \
      --out results/forgetting/${mtag}.json >> "$LOG/matched_${mtag}.log" 2>&1
    python -m eval.behavioral --model "$ck" --tag "$mtag" --device cuda:0 \
      --max_prompts 500 --out results/behavioral/${mtag}.json >> "$LOG/matched_${mtag}.log" 2>&1
    echo "   done $mtag (gpu $gpu)"
  ) &
  i=$((i + 1))
  if (( i % 4 == 0 )); then wait; fi
done
wait
echo "MATCHED RUN DONE"
