# Results

Per-condition JSON outputs from the eval + mechanistic suites, plus the Phase 6
synthesis. Tags are `<condition>_ckpt_<pct>` (e.g. `onpolicy_kd_ckpt_100`) or
`base` / `base_teacher`.

```
behavioral/<tag>.json    rhyme accuracy (test_id / held-out family / recovery), valid-couplet rate
diversity/<tag>.json     final-word entropy, distinct-n, self-BLEU, repeated-template frac (H5)
forgetting/<tag>.json    mean output KL vs base, per-layer CKA, capability retention delta (H1)
param_drift/<tag>.json   per-layer update norms, attn/mlp ratio, update concentration (Gini)
probe/<tag>.json         family-probe accuracy per (layer, offset), Delta_newline (Section 13)
patching/<tag>.json      per-layer corrupt-rhyme C at newline / rhyme word, handoff H_L (Section 14)
synthesis.json           cross-condition table + mechanism<->forgetting associations (Sections 16-17)
```

Checkpoints themselves live under `runs/` (git-ignored; regenerate with
`scripts/train_all.sh`). See [`../scripts/README.md`](../scripts/README.md) for the
run order and [`../research_outline.md`](../research_outline.md) for what each
metric tests.

A findings summary is written to [`../FINDINGS.md`](../FINDINGS.md) after synthesis.
