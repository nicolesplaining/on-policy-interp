#!/usr/bin/env bash
# Relaxation experiment: start every condition from the SAME reorganized
# checkpoint (corpus_sft-4B: Δ_newline collapsed to 0.27, forgetting KL 0.247),
# then continue training three ways and watch whether the mechanism evolves
# differently. Isolates on-policy vs off-policy vs SFT at a matched START.
#   relax_onpolicy : on-policy KD from 27B (student rollouts)
#   relax_offpolicy: off-policy KD from 27B (fixed teacher traces)
#   relax_moresft  : continued corpus SFT (control)
set -e
cd ~/on-policy-interp
source ~/opi_venv/bin/activate
export USE_TF=0 TOKENIZERS_PARALLELISM=false
INIT=runs/corpus_sft_seed0/ckpt_100

kd() {  # $1 condition  $2 gpu_pair
  CUDA_VISIBLE_DEVICES=$2 python -m training.train --condition "$1" \
    --student_model "$INIT" --teacher_model google/gemma-3-27b-it \
    --student_device cuda:0 --teacher_device cuda:1 \
    --seed 0 --max_steps 100 --out_root runs_relax \
    > ~/opi_logs/train_relax_${1}.log 2>&1
}

kd onpolicy_kd 0,1 & kd teacher_kd 2,3 & wait

# continued SFT control (single GPU, no teacher)
CUDA_VISIBLE_DEVICES=0 python -m training.train --condition corpus_sft \
  --student_model "$INIT" --student_device cuda:0 \
  --seed 0 --max_steps 100 --out_root runs_relax \
  > ~/opi_logs/train_relax_corpus_sft.log 2>&1

echo RELAX_TRAIN_DONE
