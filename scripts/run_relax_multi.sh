#!/usr/bin/env bash
# Replication seeds (1,2) of the relaxation experiment: init each from the
# seed-matched corpus_sft-4B checkpoint, continue three ways, then eval + mech
# with seed-tagged outputs. Seed 0 already done (tags relax_{onpolicy,offpolicy,moresft}).
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
BASE=google/gemma-3-4b-it
LOG=~/opi_logs

kd() {  # $1 condition  $2 gpu_pair  $3 seed
  local init="runs/corpus_sft_seed${3}/ckpt_100"
  CUDA_VISIBLE_DEVICES=$2 python -m training.train --condition "$1" \
    --student_model "$init" --teacher_model google/gemma-3-27b-it \
    --student_device cuda:0 --teacher_device cuda:1 \
    --seed "$3" --max_steps 100 --out_root "runs_relax_s${3}" \
    > "$LOG/train_relaxS${3}_${1}.log" 2>&1
}
sft() {  # $1 gpu  $2 seed
  local init="runs/corpus_sft_seed${2}/ckpt_100"
  CUDA_VISIBLE_DEVICES=$1 python -m training.train --condition corpus_sft \
    --student_model "$init" --student_device cuda:0 \
    --seed "$2" --max_steps 100 --out_root "runs_relax_s${2}" \
    > "$LOG/train_relaxS${2}_corpus_sft.log" 2>&1
}

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
  echo "   done $tag"
}

for s in 1 2; do
  echo "=== relax train seed $s ==="
  kd onpolicy_kd 0,1 "$s" & kd teacher_kd 2,3 "$s" & wait
  sft 0 "$s"
  echo "=== relax eval seed $s ==="
  suite "relax_onpolicy_seed${s}"  "runs_relax_s${s}/onpolicy_kd_seed${s}/ckpt_100" 0 &
  suite "relax_offpolicy_seed${s}" "runs_relax_s${s}/teacher_kd_seed${s}/ckpt_100"  1 &
  suite "relax_moresft_seed${s}"   "runs_relax_s${s}/corpus_sft_seed${s}/ckpt_100"  2 &
  wait
done
echo RELAX_MULTI_DONE
