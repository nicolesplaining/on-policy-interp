#!/usr/bin/env bash
# Eval + mechanism for the relaxation experiment. Compares how Δ_newline and
# forgetting evolve from the corpus_sft-4B start under on-policy / off-policy / SFT.
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
BASE=google/gemma-3-4b-it
LOG=~/opi_logs

until grep -q RELAX_TRAIN_DONE "$LOG/run_relax.log" 2>/dev/null; do sleep 30; done
echo "relax training done -> eval + mech"

declare -a M=(
  "relax_onpolicy|runs_relax/onpolicy_kd_seed0/ckpt_100"
  "relax_offpolicy|runs_relax/teacher_kd_seed0/ckpt_100"
  "relax_moresft|runs_relax/corpus_sft_seed0/ckpt_100"
)

suite() {  # $1 tag  $2 path  $3 gpu
  local tag=$1 path=$2 gpu=$3 dev=cuda:0
  export CUDA_VISIBLE_DEVICES=$gpu
  python -m eval.behavioral --model "$path" --tag "$tag" --device $dev --max_prompts 500 \
    --out results/behavioral/${tag}.json
  python -m eval.forgetting --trained "$path" --base "$BASE" --tag "$tag" --device $dev \
    --out results/forgetting/${tag}.json
  python -m mech.extract_activations --model "$path" --prompts data/prompt_pool/train.jsonl \
    --n 1200 --device $dev --out mech/acts/${tag}_train.pt
  python -m mech.extract_activations --model "$path" --prompts data/prompt_pool/test_id.jsonl \
    --n 800 --device $dev --out mech/acts/${tag}_val.pt
  python -m mech.probe --train_acts mech/acts/${tag}_train.pt --val_acts mech/acts/${tag}_val.pt \
    --device $dev --tag "$tag" --out results/probe/${tag}.json
  python -m mech.activation_patching --model "$path" --tag "$tag" --device $dev \
    --n_samples 20 --out results/patching/${tag}.json
  echo "   done suite $tag (gpu $gpu)"
}

i=0
for entry in "${M[@]}"; do
  tag=${entry%%|*}; path=${entry##*|}
  suite "$tag" "$path" $((i % 4)) >> "$LOG/eval_${tag}.log" 2>&1 &
  i=$((i + 1))
done
wait
echo RELAX_EVAL_DONE
