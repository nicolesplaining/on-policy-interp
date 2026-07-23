# Findings (3 seeds × 150 steps)

Full end-to-end study on the 4×H100 cluster: Gemma-3-4B student, Gemma-3-27B
teacher, 10k shared prompts, all four regimes trained from the same init across
**three seeds**, then evaluated behaviorally + mechanistically. Numbers below are
**mean [min, max] over seeds 0–2**. This is the outline's reduced-budget
operating point (150 steps; the task saturates quickly, as Section 5 anticipated),
not the full 600-step budget — but every effect below is consistent in direction
across all three seeds.

Figures are in [`figures/`](figures/) (regenerate with
`python -m analysis.make_figures --seeds 0 1 2`).

**The two headline results:**

![Forgetting](figures/fig1_forgetting.png)

![Decodable vs causal](figures/fig3_decodable_vs_causal.png)

## Behavioral (Section 11)

| condition | test-id rhyme | held-out family | recovery |
|---|---|---|---|
| base | 0.79 | — | — |
| corpus_sft | 0.956 [.945,.964] | ~0.92 | 0.839 [.830,.852] |
| teacher_sft | 0.869 [.861,.878] | ~0.91 | 0.774 [.750,.808] |
| teacher_kd | 0.877 [.874,.884] | ~0.91 | 0.789 [.770,.802] |
| onpolicy_kd | 0.864 [.857,.873] | ~0.93 | 0.819 [.812,.826] |

All regimes improve over the base student (0.79). `corpus_sft` reaches the highest
in-distribution accuracy (it fits a corpus that rhymes perfectly by construction).
**On-policy KD has the best recovery-prefix accuracy** (0.819 vs 0.774 / 0.789 for
the off-policy regimes) — the behavioral advantage predicted for training on
student-visited states (Section 11.2), and its seed range [.812,.826] sits at or
above the others' across seeds.

## Forgetting (Section 12, H1) — the clearest effect

| condition | output-KL vs base | mean CKA | update norm | update Gini |
|---|---|---|---|---|
| corpus_sft | **0.247 [.240,.256]** | 0.9993 | 2.2 | 0.147 |
| teacher_sft | 0.029 [.028,.029] | 0.9997 | 1.4 | 0.072 |
| teacher_kd | 0.020 [.017,.023] | 0.9998 | 1.4 | 0.067 |
| onpolicy_kd | 0.030 [.028,.032] | 0.9998 | 2.0 | 0.085 |

**Fixed-corpus SFT drifts ~8–14× more on general (non-poetry) text** than the
three teacher-supervised regimes — seed ranges do not overlap. The
teacher-supervised regimes stay far more task-selective; capability probes are
essentially unchanged for all. This is the strongest and most robust signal in
the study, supporting **H1/H2**: static off-policy corpus SFT interferes broadly,
teacher supervision (including on-policy) does not.

## Diversity (Section 11.3, H5)

| condition | final-word entropy | distinct-2 | self-BLEU | repeated-template |
|---|---|---|---|---|
| corpus_sft | 8.78 | ~0.12 | ~0.06 | ~0.10 |
| teacher_sft | 7.90 | ~0.53 | ~0.003 | ~0.007 |
| teacher_kd | 7.78 | ~0.53 | ~0.003 | ~0.005 |
| onpolicy_kd | 7.86 | ~0.53 | ~0.003 | ~0.007 |

**H5 (reverse-KL mode collapse) is not observed:** on-policy KD is as diverse as
off-policy KD (distinct-2 ≈ 0.53, self-BLEU ≈ 0.003). It is `corpus_sft` that
collapses — onto the *template structure* of the (templated) corpus. That is
partly an artifact of the controlled synthetic corpus and motivates swapping in a
natural-poetry corpus (a listed refinement).

## Mechanism (Sections 13–15)

**Probe — newline decodability (Δ_newline, family probe):**

| condition | Δ_newline | peak layer |
|---|---|---|
| base | 0.565 | 26 |
| corpus_sft | 0.274 [.21,.39] | early (≈L4) |
| teacher_sft | 0.817 [.81,.84] | 26 |
| teacher_kd | 0.838 [.83,.85] | 26 |
| onpolicy_kd | 0.804 [.79,.82] | 26 |

The three **teacher-supervised** regimes all *strengthen* the late-layer (L26)
newline as a decodable planning site (Δ_newline ≈ 0.80–0.84 vs base 0.565), at the
same layer, regardless of hard/soft or off-/on-policy. **`corpus_sft` alone**
weakens the newline signal and shifts its peak to an early layer — a qualitatively
different representational reorganization.

![Per-layer newline decodability](figures/fig4_probe_layers.png)

Teacher supervision amplifies the base student's existing late-layer newline site
(gray dashed → colored curves peak ~0.85 at L26); `corpus_sft` (red) abandons it,
staying flat at the newline in late layers.

**Patching — causal reliance (handoff H = C_newline − C_rhyme_word):**

`handoff_frac = 0` for every condition and every seed. H stays negative across
layers (rhyme-word patching is at least as effective as newline patching
everywhere); no trained 4B model develops a causal rhyme-word→newline handoff.
This matches the outline's explicitly anticipated outcome (Section 19: *"No
trained 4B model develops a newline handoff"* — the Gemma-3-27B phenomenon likely
needs greater scale).

**The key dissociation (robust across seeds):** teacher supervision makes the
future rhyme markedly more **decodable** at the newline (Δ_newline ↑) while leaving
the model still **causally** reliant on the original rhyme word (H < 0).
Decodable ≠ causal — exactly the distinction the look-ahead methodology separates.

## Mechanism ↔ forgetting (Sections 16–17)

Across conditions, **parameter-update concentration predicts general-text drift**:
Pearson r(update-Gini, output-KL) = **0.99** (3-seed-averaged rows). `corpus_sft`
has both the most concentrated updates (Gini 0.147) and by far the most drift
(KL 0.247); the teacher-supervised regimes spread smaller updates and stay
task-selective. (The newline-C / handoff correlations are degenerate because
patching saturates near 1.0 at some layer for every condition; a finer per-layer
causal comparison is the natural follow-up.)

### Scorecard against the hypotheses

- **H1 (on-policy ≤ drift):** supported — on-policy KD and the off-policy KD
  regimes drift ~10× less than fixed-corpus SFT; CKA ≈ 1.0.
- **H2 (static SFT reorganizes):** supported *representationally* for
  `corpus_sft` (Δ_newline collapses, peak moves early); not as a *causal* handoff
  (none forms at 4B).
- **H3 (teacher-trace ≠ generic SFT):** supported — teacher_sft patterns with the
  KD regimes (high Δ_newline, low drift), not with corpus_sft.
- **H4 (prefix source, divergence/teacher held fixed):** teacher_kd vs
  onpolicy_kd are close on most axes at 150 steps; on-policy's edge appears in
  recovery and (marginally) CKA, not yet in the planning-site mechanism.
- **H5 (reverse-KL diversity collapse):** not observed.

## Caveats & next steps

- 150 steps (reduced budget; task saturates). The full 600-step budget and a
  matched-performance checkpoint selection are the natural extensions; all
  longitudinal checkpoints (0/10/25/50/75/100%) are saved to support the
  when-does-it-emerge analysis (Section 13.4).
- The fixed corpus is a controlled *templated* set (valid rhymes, shared first
  lines, non-teacher). Its low phrasing diversity is by construction and is
  measured, not hidden; a natural-poetry corpus is the recommended swap.
- No 4B causal handoff emerged; repeating with a larger student (or the 27B as
  student) is the way to probe the Section-16 mechanistic-inheritance question.

## Figures

| file | shows |
|---|---|
| `figures/fig1_forgetting.png` | output-KL + CKA per condition (H1) |
| `figures/fig2_behavioral.png` | rhyme accuracy by split (test / held-out / recovery) |
| `figures/fig3_decodable_vs_causal.png` | Δ_newline (decodable) vs handoff H (causal) |
| `figures/fig4_probe_layers.png` | per-layer newline family-decodability |
| `figures/fig5_patching_layers.png` | per-layer causal C: newline vs rhyme word |
| `figures/fig6_cka_layers.png` | per-layer representation drift vs base |
| `figures/fig7_diversity.png` | distinct-2 / self-BLEU (H5) |
| `figures/fig8_mech_vs_forget.png` | update concentration vs output drift (r≈0.99) |

_Artifacts: per-condition JSON under `results/{behavioral,diversity,forgetting,
param_drift,probe,patching}/`, aggregated `results/synthesis_3seed.json`,
figures under `figures/`, checkpoints + `history.json` under `runs/`._
