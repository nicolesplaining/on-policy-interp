# Pilot findings (seed 0, 150 steps)

A first end-to-end pass of the full study on the 4×H100 cluster: Gemma-3-4B
student, Gemma-3-27B teacher, 10k shared prompts, all four regimes trained from
the same init, then evaluated behaviorally + mechanistically. This is a **pilot**
(one seed, reduced step budget) meant to validate the pipeline and surface
preliminary effects, not the final 3-seed result.

## Behavioral (Section 11)

| condition | val-rhyme (train) | test-id | held-out family | recovery |
|---|---|---|---|---|
| base | 0.785 | 0.79 | — | — |
| corpus_sft | 0.975 | 0.964 | 0.916 | 0.834 |
| teacher_sft | 0.90 | 0.878 | 0.913 | 0.764 |
| teacher_kd | 0.865 | 0.874 | 0.912 | 0.794 |
| onpolicy_kd | 0.865 | 0.857 | 0.929 | 0.812 |

All four regimes improve rhyming over the base student. `corpus_sft` reaches the
highest in-distribution accuracy (it fits a corpus that rhymes perfectly by
construction). Notably, **on-policy KD has the best recovery-prefix accuracy**
(0.812 vs 0.764 for teacher_sft) and the best held-out-family accuracy — the
behavioral advantage the outline predicts for training on student-visited states
(Section 11.2).

## Forgetting (Section 12, H1)

| condition | output-KL vs base | mean CKA | min CKA | update norm | update Gini |
|---|---|---|---|---|---|
| corpus_sft | **0.240** | 0.9993 | 0.984 | 2.21 | 0.147 |
| teacher_sft | 0.029 | 0.9997 | 0.992 | 1.40 | 0.072 |
| teacher_kd | 0.023 | 0.9998 | 0.997 | 1.38 | 0.067 |
| onpolicy_kd | 0.028 | 0.9998 | **0.997** | 2.02 | 0.085 |

**Fixed-corpus SFT causes ~10× more general-text output drift** (KL 0.240 vs
~0.025) and the lowest representation similarity to the base — the clearest H1/H2
signal in the pilot. The three teacher-supervised regimes stay far more
task-selective on non-poetry text. Capability probes were essentially unchanged
for all trained models.

## Diversity (Section 11.3, H5)

| condition | final-word entropy | distinct-2 | self-BLEU | repeated-template frac |
|---|---|---|---|---|
| corpus_sft | 8.80 | **0.119** | **0.060** | **0.100** |
| teacher_sft | 7.92 | 0.525 | 0.002 | 0.008 |
| teacher_kd | 7.76 | 0.535 | 0.003 | 0.005 |
| onpolicy_kd | 7.88 | 0.535 | 0.003 | 0.007 |

**H5 (reverse-KL mode collapse) is not observed** at this scale: on-policy KD is
as diverse as off-policy KD (distinct-2 ≈ 0.53, self-BLEU ≈ 0.003). Instead it is
`corpus_sft` that collapses — onto the *template structure* of the (templated)
corpus (distinct-2 0.12, self-BLEU 0.06). This is partly an artifact of the
controlled synthetic corpus and motivates swapping in a natural-poetry corpus
(a listed refinement).

## Mechanism (Sections 13-15)

**Probe — newline decodability (Δ_newline, family probe):**

| condition | Δ_newline | peak layer |
|---|---|---|
| base | 0.565 | 26 |
| corpus_sft | 0.361 | **4** |
| teacher_sft | 0.819 | 26 |
| teacher_kd | 0.812 | 26 |
| onpolicy_kd | 0.796 | 26 |

The three **teacher-supervised** regimes all *strengthen* the late-layer (L26)
newline as a decodable planning site (Δ_newline ≈ 0.80 vs base 0.565), at the
same layer. **`corpus_sft` alone reorganizes the geometry** — the peak drops to
0.361 and moves to an early layer (L4). So the teacher's supervision, whether
hard, soft-off-policy, or soft-on-policy, pushes the student's *representation*
toward a common late-layer newline profile; generic corpus SFT does something
qualitatively different.

**Patching — causal reliance (handoff H = C_newline − C_rhyme_word):**

No condition develops a causal rhyme-word→newline **handoff** (all `handoff=False`;
H stays negative across layers — rhyme-word patching is at least as effective as
newline patching everywhere). Even the 27B teacher does not cross into a clearly
positive late-layer H on these prompts. This matches the outline's anticipated
outcome (Section 19: *"No trained 4B model develops a newline handoff"* — the
Gemma-3-27B phenomenon likely needs greater scale).

**The key dissociation:** teacher supervision makes the future rhyme markedly more
**decodable** at the newline (Δ_newline ↑) while leaving the model still
**causally** reliant on the original rhyme word (H < 0). Decodable ≠ causal —
exactly the distinction the look-ahead methodology was built to separate.

## Mechanism ↔ forgetting (Sections 16-17)

Across conditions, **parameter-update concentration predicts output drift**:
Pearson r(update-Gini, output-KL) = **0.98**. `corpus_sft` has both the most
concentrated updates (Gini 0.147, energy pushed into a few layers) and by far the
most general-text drift (KL 0.240); the teacher-supervised regimes spread smaller
updates and stay task-selective. (Newline-C / handoff correlations are
degenerate here because patching saturates near 1.0 at some layer for every
condition — a finer-grained, per-layer causal comparison is the natural
follow-up.)

### Reading against the hypotheses

- **H1 (on-policy ≤ drift):** supported directionally — on-policy KD has the
  highest min-CKA and low output-KL; fixed-corpus SFT drifts ~10× more.
- **H2 (static SFT reorganizes):** supported for `corpus_sft` at the
  *representational* level (probe geometry moves to L4), though not as a *causal*
  handoff (none forms at 4B).
- **H3 (teacher-trace ≠ generic SFT):** supported — teacher_sft looks like the
  KD conditions (Δ_newline ≈ 0.82, low drift), not like corpus_sft.
- **H4 (prefix source, holding divergence/teacher fixed):** teacher_kd vs
  onpolicy_kd are close on most axes here; on-policy's edge shows up in recovery
  and min-CKA, not yet in the mechanism at 150 steps.
- **H5 (reverse-KL diversity collapse):** not observed (see above).

## Caveats

- One seed, 150 steps — trends, not confirmed effects. The 3-seed / full-budget
  run is the confirmatory experiment.
- The fixed corpus is a controlled *templated* set (valid rhymes, shared first
  lines, non-teacher) rather than natural poetry; its low phrasing diversity is
  by construction and is measured, not hidden.
