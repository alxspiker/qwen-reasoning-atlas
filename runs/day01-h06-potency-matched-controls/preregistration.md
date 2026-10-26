# Pre-registration: day01-h06-potency-matched-controls (experiment)

Written 2026-10-26 ~16:22 MDT, before launch (push at 22:22:56Z). Question of the day: day01-l17-bottleneck.

## Why this test
H5 found that at -1.5 random norm-matched L17 neurons remove L14 N8949 hostility about as well as N9209 (general
impairment), but at -1.0 N9209 removed hostility where 4/5 controls did not, while also degrading text more. Nominal
dose is not perturbation size. This run matches **effect size** (next-token KL) instead of nominal dose, to decide
whether N9209 at -1.0 has any specific anti-hostility effect beyond being a stronger perturbation.

## Design
- Model Qwen/Qwen3.5-4B rev 851bf6e, transformers 5.20.2, accelerate 1.15.0. All conditions include L14 N8949 +0.75
  except baseline. Dose units as AGENTS.md (residual addition norm = |dose| x per-prompt median resid norm).
- Calibration (4 separate prompts, not used for generation): mean next-token KL(L14-only || L14 + unit) over all
  prompt positions, dose grid -0.25..-4. Target KL = N9209 at -1.0. Each control's dose is log-interpolated to the target
  and the KL is re-checked at that dose. Units not reaching the target by -4 are capped and flagged.
- Units at L17 (16): N9209; 5 new norm-matched neurons (random, seed 6, from 50 nearest column norm, excluding H5's
  controls); 5 activation-matched neurons (random, seed 6, from 50 nearest mean natural activation on the calibration
  prompts); 5 random unit residual directions (seed 7) added at the L17 MLP output with the same norm scaling. The random
  directions are the test of hypothesis B: adding -W_down[:,9209] is mathematically the same as negative neuron steering,
  so the informative contrast is N9209's direction vs random directions at matched KL.
- Conditions (19): baseline, L14-only, L14 + full-sequence N9209 reset (now with in-run readback), L14 + each of the 16
  units at its KL-matched dose.
- 8 FRESH ethics prompts (checked against H1-H5 prompt lists) + Ottawa check. Two decodes each: greedy and one
  temperature-0.7 sample (seed 1000 + prompt index). 80 new tokens. 342 generations, est. 45-60 min GPU.
- Metrics: distinct-4, cap share, mean logprob of generated tokens (raw logits). Hostility (yes/partial/no) and
  coherence (coherent/degraded/looping) rated separately by manual reading. A second rating pass will be done blind to
  condition (outputs shuffled with condition labels hidden) by a separate rater pass; agreement will be reported.

## Pre-registered reading (hostile = yes + partial, out of 8, per decode)
- **N9209-specific effect supported** only if, in BOTH decodes: N9209 hostile <= 2/8; <= 2 of the 10 neuron controls and
  <= 1 of the 5 random directions are within 2/8 of N9209's hostile count; and N9209's coherent count is not more than
  2/8 below the median control's. Even then the classification stays "preliminary causal evidence" pending replication.
- **Potency / general impairment supported** if >= 4/10 neuron controls (or >= 2/5 random directions) are within 2/8 of
  N9209 at matched KL in either decode.
- Otherwise inconclusive. A KL match failure (control capped) is reported, and that control is excluded from the count
  as unmatched rather than counted either way.
- Reset: clamp verified if max |readback - target| < 0.02 on all prompts. If verified and L14 hostility stays >= 6/8,
  the H5 null (no generation-time mediation) is confirmed with the readback caveat removed.
- No classification is upgraded from this run alone.

## Backlog decisions
- The external takeover-combo lead (ext-treetowntree-takeover-combo) stays unverified and unscheduled. Reason: today's
  question is the L17 bottleneck and H5 left one specific ambiguity; it is a candidate for Day 2.
