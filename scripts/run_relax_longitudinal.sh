#!/usr/bin/env bash
# Longitudinal probe of the relaxation run (seed 0): for each saved checkpoint
# (0/10/25/50/75/100%) of each continuation, extract activations and probe
# Δ_newline, so we can see the *trajectory* of mechanism relaxation — does
# on-policy relax gradually while off-policy jumps to the teacher fast?
# Lighter extraction (n=600/300) for speed. Fanned across 4 GPUs.
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
CKPTS="ckpt_000 ckpt_010 ckpt_025 ckpt_050 ckpt_075 ckpt_100"

probe_ckpt() {  # $1 cond_dir  $2 label  $3 ckpt  $4 gpu
  local path="runs_relax/${1}/${3}" tag="relaxlong_${2}_${3}" dev=cuda:0
  [ -d "$path" ] || return 0
  export CUDA_VISIBLE_DEVICES=$4
  python -m mech.extract_activations --model "$path" --prompts data/prompt_pool/train.jsonl \
    --n 600 --device $dev --out mech/acts/${tag}_train.pt >> ~/opi_logs/relaxlong_${2}.log 2>&1
  python -m mech.extract_activations --model "$path" --prompts data/prompt_pool/test_id.jsonl \
    --n 300 --device $dev --out mech/acts/${tag}_val.pt >> ~/opi_logs/relaxlong_${2}.log 2>&1
  python -m mech.probe --train_acts mech/acts/${tag}_train.pt --val_acts mech/acts/${tag}_val.pt \
    --device $dev --tag "$tag" --out results/probe/${tag}.json >> ~/opi_logs/relaxlong_${2}.log 2>&1
  echo "   done ${tag}"
}

# three continuations (dir | label)
declare -a C=("onpolicy_kd_seed0|onpolicy" "teacher_kd_seed0|offpolicy" "corpus_sft_seed0|moresft")
g=0
for ck in $CKPTS; do
  for entry in "${C[@]}"; do
    dir=${entry%%|*}; lab=${entry##*|}
    probe_ckpt "$dir" "$lab" "$ck" $((g % 4)) &
    g=$((g + 1))
    (( g % 4 == 0 )) && wait
  done
done
wait
echo RELAX_LONGITUDINAL_DONE
