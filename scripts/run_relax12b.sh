#!/usr/bin/env bash
# 12B scale replication of the relaxation experiment: init 12B student from
# corpus_sft-12B, continue three ways with the 27B teacher, and check whether the
# on-policy-conservative / off-policy-aggressive / SFT-entrenches pattern holds
# at 12B. Student full-FT via 8-bit AdamW on one GPU; 27B teacher on a second.
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
INIT=runs12b/corpus_sft_seed0/ckpt_100
BASE=google/gemma-3-12b-it
LOG=~/opi_logs

kd() {  # $1 condition  $2 gpu_pair
  CUDA_VISIBLE_DEVICES=$2 python -m training.train --condition "$1" \
    --student_model "$INIT" --teacher_model google/gemma-3-27b-it \
    --student_device cuda:0 --teacher_device cuda:1 --opt8bit \
    --seed 0 --max_steps 100 --batch_size 2 --grad_accum 4 --out_root runs_relax12b \
    > "$LOG/train_relax12b_${1}.log" 2>&1
}

kd onpolicy_kd 0,1 & kd teacher_kd 2,3 & wait
CUDA_VISIBLE_DEVICES=0 python -m training.train --condition corpus_sft \
  --student_model "$INIT" --student_device cuda:0 --opt8bit \
  --seed 0 --max_steps 100 --batch_size 4 --out_root runs_relax12b \
  > "$LOG/train_relax12b_corpus_sft.log" 2>&1

echo "=== eval ==="
suite() {  # $1 tag  $2 path  $3 gpu
  local tag=$1 path=$2 gpu=$3 dev=cuda:0
  export CUDA_VISIBLE_DEVICES=$gpu
  python -m eval.behavioral --model "$path" --tag "$tag" --device $dev --max_prompts 500 \
    --out results/behavioral/${tag}.json
  python -m eval.forgetting --trained "$path" --base "$BASE" --tag "$tag" --device $dev \
    --out results/forgetting/${tag}.json
  python -m mech.extract_activations --model "$path" --prompts data/prompt_pool/train.jsonl \
    --n 1000 --device $dev --out mech/acts/${tag}_train.pt
  python -m mech.extract_activations --model "$path" --prompts data/prompt_pool/test_id.jsonl \
    --n 500 --device $dev --out mech/acts/${tag}_val.pt
  python -m mech.probe --train_acts mech/acts/${tag}_train.pt --val_acts mech/acts/${tag}_val.pt \
    --device $dev --tag "$tag" --out results/probe/${tag}.json
  echo "   done $tag"
}
suite relax12b_onpolicy  runs_relax12b/onpolicy_kd_seed0/ckpt_100 0 &
suite relax12b_offpolicy runs_relax12b/teacher_kd_seed0/ckpt_100  1 &
suite relax12b_moresft   runs_relax12b/corpus_sft_seed0/ckpt_100  2 &
wait
echo RELAX12B_DONE
