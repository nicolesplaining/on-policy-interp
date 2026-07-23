# Run guide (4×H100 cluster)

Phase-by-phase drivers for the study in [`../research_outline.md`](../research_outline.md).
All scripts assume the `opi_venv` virtualenv and models cached under
`~/.cache/huggingface` (gated Gemma-3 access via `HF_TOKEN` / `~/.hf_token`).

```bash
source ~/opi_venv/bin/activate      # torch 2.7 + transformers, Pillow>=10
export HF_TOKEN=...                  # or place it in ~/.hf_token
```

## Phase 2 — data

```bash
scripts/run_data.sh
```
Builds the 10k shared prompt pool (train/val/test_id/test_heldout_family/recovery),
the fixed non-teacher corpus, and the teacher traces (Gemma-3-27B, sharded one
per GPU). Merges shards and writes `data/match_report.json`.

## Phase 3 — pilot (1 seed, reduced steps)

```bash
scripts/train_all.sh 0 200      # seed 0, 200 steps
scripts/eval_all.sh  0
scripts/mech_all.sh  0
python -m analysis.synthesize --seed 0
```
Validates the full pipeline end-to-end and sets the matched-performance threshold.

## Phase 4 — full training (3 seeds)

```bash
for s in 0 1 2; do scripts/train_all.sh $s 600; done
```
Longitudinal checkpoints (0/10/25/50/75/100%) are saved under
`runs/<condition>_seed<s>/ckpt_*`.

## Phase 5 — mechanistic analysis + Phase 6 — synthesis

```bash
for s in 0 1 2; do scripts/eval_all.sh $s; scripts/mech_all.sh $s; done
python -m analysis.synthesize --seed 0 --out results/synthesis.json
```

## GPU scheduling

`train_all.sh` runs the two KD conditions first (each needs a student GPU + an
online 27B teacher GPU → all 4 H100s), then the two SFT conditions (1 GPU each).
`eval_all.sh` / `mech_all.sh` fan the five models (base + 4 conditions, + 27B
teacher for inheritance) across the 4 GPUs.

## Outputs

```
runs/<cond>_seed<s>/ckpt_*/     student checkpoints + history.json
results/behavioral|diversity|forgetting|param_drift/<tag>.json
results/probe|patching/<tag>.json
mech/acts/<tag>_{train,val}.pt  extracted activations
results/synthesis.json          Phase 6 cross-condition table + associations
```
