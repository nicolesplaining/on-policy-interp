#!/usr/bin/env bash
# Eval + mechanism for the stabilized 12B self-distillation comparison.
#   Model 1: corpus_sft 12B (direct SFT)                 -> corpus_sft12b_seed{s}
#   Model 2: on-policy self-distill v2 (stabilized)      -> sd12bv2_seed{s}
# Behavioral, diversity, forgetting (vs base 12B), param-drift, probe Δ_newline;
# patching (handoff) on seed 0 of each. Fanned across 4 GPUs.
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
BASE=google/gemma-3-12b-it
LOG=~/opi_logs

echo "waiting for SD12BV2 training..."
until grep -q SD12BV2_TRAIN_DONE "$LOG/run_sd12b_v2.log" 2>/dev/null; do sleep 60; done
echo "training done -> eval + mech"

declare -a M=(
  "corpus_sft12b_seed0|runs12b/corpus_sft_seed0/ckpt_100"
  "corpus_sft12b_seed1|runs12b/corpus_sft_seed1/ckpt_100"
  "corpus_sft12b_seed2|runs12b/corpus_sft_seed2/ckpt_100"
  "sd12bv2_seed0|runs12b_selfdistill_v2/onpolicy_kd_seed0/ckpt_100"
  "sd12bv2_seed1|runs12b_selfdistill_v2/onpolicy_kd_seed1/ckpt_100"
)

suite() {  # $1 tag  $2 path  $3 gpu
  local tag=$1 path=$2 gpu=$3 dev=cuda:0
  export CUDA_VISIBLE_DEVICES=$gpu
  python -m eval.behavioral --model "$path" --tag "$tag" --device $dev --max_prompts 500 \
    --out results/behavioral/${tag}.json
  python -m eval.diversity --model "$path" --tag "$tag" --device $dev \
    --out results/diversity/${tag}.json
  python -m eval.forgetting --trained "$path" --base "$BASE" --tag "$tag" --device $dev \
    --out results/forgetting/${tag}.json
  python -m eval.param_drift --trained "$path" --base "$BASE" --tag "$tag" \
    --out results/param_drift/${tag}.json
  python -m mech.extract_activations --model "$path" --prompts data/prompt_pool/train.jsonl \
    --n 1000 --device $dev --out mech/acts/${tag}_train.pt
  python -m mech.extract_activations --model "$path" --prompts data/prompt_pool/test_id.jsonl \
    --n 500 --device $dev --out mech/acts/${tag}_val.pt
  python -m mech.probe --train_acts mech/acts/${tag}_train.pt --val_acts mech/acts/${tag}_val.pt \
    --device $dev --tag "$tag" --out results/probe/${tag}.json
  case "$tag" in
    corpus_sft12b_seed0|sd12bv2_seed0)
      python -m mech.activation_patching --model "$path" --tag "$tag" --device $dev \
        --n_samples 16 --out results/patching/${tag}.json ;;
  esac
  echo "   done suite $tag (gpu $gpu)"
}

i=0
for entry in "${M[@]}"; do
  tag=${entry%%|*}; path=${entry##*|}
  suite "$tag" "$path" $((i % 4)) >> "$LOG/eval_${tag}.log" 2>&1 &
  i=$((i + 1))
  (( i % 4 == 0 )) && wait
done
wait
echo SD12BV2_EVAL_DONE
