# On-Policy Distillation vs Static SFT: A Mechanistic Study

*How does the training-state distribution change a model's internal computation?*

This repo implements the study in [`research_outline.md`](research_outline.md): a
controlled comparison of four training regimes that all improve rhyming-couplet
generation, asking whether they implement that behavior through **different
internal computations** — and whether preserving the student's original causal
pathway is what reduces forgetting.

The task is a clean latent-planning probe: complete the second line of a couplet
so it rhymes with the first line's final word. Prior work
([look-ahead](https://github.com/nicolesplaining/look-ahead)) found the future
rhyme is linearly decodable at the newline, but only some models *causally* rely
on a newline "planning site" via a rhyme-word → newline handoff. We reuse that
paper's probing / patching methodology to track whether training induces,
preserves, or relocates that site.

## Conditions

From one frozen **Gemma-3-4B** student, teacher **Gemma-3-27B**:

| Condition | Prefix source | Target | Isolates |
| --- | --- | --- | --- |
| `base` | — | — | original behavior + mechanism |
| `corpus_sft` | existing poems | ground-truth tokens | generic static SFT |
| `teacher_sft` | fixed teacher rollouts | hard teacher tokens | off-policy imitation |
| `teacher_kd` | fixed teacher rollouts | teacher distribution | hard vs soft labels |
| `onpolicy_kd` | student rollouts | teacher distribution | student-visited states |

## Layout

```
src/opi/            shared library: rhyme, prompts, positions, models, io
data/               Phase 2: prompt pool, poem corpus, teacher traces, matching
training/           Phase 3-4: 4 training regimes + shared loop / checkpoints
eval/               behavioral, diversity, forgetting, parameter drift
mech/               Phase 5: activation extraction, probing, patching, handoff, heads
analysis/           Phase 6: synthesis (mechanism <-> forgetting)
scripts/            per-phase orchestration (run on the 4xH100 cluster)
tests/              unit tests for the pure-Python logic
```

## Setup

```bash
./setup_venv.sh              # creates .venv, installs deps
source .venv/bin/activate
pip install -e .             # exposes `opi`
export HF_TOKEN=...          # gated Gemma-3 access
```

On the GPU cluster the four student trainings run one-per-H100 in parallel; the
27B teacher is sharded for trace generation and on-policy scoring. See
[`scripts/README.md`](scripts/README.md) for the phase-by-phase run guide and
`research_outline.md` for the full experimental design, hypotheses, and success
criteria.

## Status

Full pipeline implemented and executed on a 4×H100 cluster across **three seeds**
(150-step budget, all four regimes). Per-condition results are under `results/`,
aggregated in `results/synthesis_3seed.json`; see [`FINDINGS.md`](FINDINGS.md) for
the write-up. Robust, seed-consistent signals:

- fixed-corpus SFT drifts **~8–14× more** on general text than the three
  teacher-supervised regimes (output-KL 0.247 vs 0.02–0.03; ranges disjoint) —
  and this **survives a matched-performance re-analysis**, so it isn't just a
  "trained further" artifact;
- **on-policy KD does *not* forget less than off-policy KD** — the clean
  `teacher_kd` vs `onpolicy_kd` test (matched) gives 0.013 vs 0.023. Forgetting
  tracks *teacher-vs-corpus supervision*, not the on-/off-policy prefix source;
- teacher supervision (hard / off-policy-soft / on-policy-soft) raises newline
  *decodability* (Δ_newline ≈ 0.80–0.84 vs base 0.565) **without** inducing a
  *causal* handoff (`handoff_frac = 0` everywhere) — decodable ≠ causal;
- on-policy KD's only edge is recovery-prefix accuracy;
- parameter-update concentration tracks output drift (r ≈ 0.99).

See [`FINDINGS.md`](FINDINGS.md) for the full per-hypothesis scorecard, including
the corrected reading of H1/H4.
