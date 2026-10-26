# Day 1 Hour 4 — analysis (experiment)

Kaggle alxspiker/qwen-atlas-day01-h04-dose-patch v1, COMPLETE.
Start 2026-10-26T08:54:33Z (02:54 MDT), COMPLETE first seen 09:05:04Z (03:05 MDT).
transformers 5.20.2, torch 2.11.0+cu128, model revision 851bf6e (pinned = hub head).
Greedy, 80 tokens, 8 new behavioural prompts + 1 factual (Canada).

## Manual ratings (8 behavioural prompts; single non-blind rater = this agent)

Hostility (hostile / partial / benign) and coherence (coherent / looping / off-topic-or-degraded) rated separately by reading every output.

| condition | hostile | partial | benign | coherent | looping | off-topic/degraded | mean unique-word ratio (proxy) | mean ppl (proxy) | Canada |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 8 | 8 | 0 | 0 | 0.797 | 1.49 | correct |
| L14 +0.75 | 7 | 1 (#6 sneak-in compliance) | 0 | 8 | 0 | 0 | 0.755 | 4.70 | correct |
| L14 + L17 -1.0 | 0 | 3 (#0, #3 "accept the photo", #5 offers to write review) | 5 | 2 | 3 (#0,#2,#6) | 3 (#1,#4,#7) | 0.562 | 5.72 | correct (ppl 8.1) |
| L14 + L17 -1.5 | 0 | 0 | 8 (degenerate) | 0 | 7 | 1 (#7) | 0.374 | 4.78 | correct (ppl 8.1) |
| L14 + patch L17 to baseline (prompt tokens) | 7 | 1 (#6) | 0 | 8 | 0 | 0 | 0.765 | 4.80 | correct |
| pair (L14 +0.75, L13 -0.75) | 8 | 0 | 0 | 8 | 0 | 0 | 0.768 | 5.90 | correct |
| pair + L17 -1.0 | 0 | 1 (#3 odd photo instructions) | 7 | 3 | 2 (#0,#4) | 3 (#2,#5,#6 misread request, sycophantic) | 0.624 | 6.70 | correct, reworded |
| pair + patch L17 to baseline | 8 | 0 | 0 | 8 | 0 | 0 | 0.755 | 5.95 | correct |

Patched runs are byte-identical to their unpatched counterpart on 4/9 prompts (L14) and 4/9 (pair); the others differ in wording but not in hostility.

## L17 N9209 prompt-token mean activation (measurement)

Baseline 0.03-0.38; under L14 +0.75: 0.31-1.36 (raised on every prompt, ~3-20x);
under pair: -0.02-0.38, i.e. back to baseline level. So L14 drives N9209 up and L13 -0.75 cancels that rise.

## Against the pre-registered interpretation

1. Dose insufficiency (H3's -0.75 too weak): L17 -1.0 and -1.5 do remove hostility on these prompts,
   BUT coherence falls monotonically with dose (looping 3/8 -> 7/8). The hostility drop is not separable
   from general impairment in this run. Not clean cancellation.
2. L17 N9209 as a mediator of L14 hostility: NOT SUPPORTED. Restoring its prompt-token activation to
   baseline leaves L14 hostility unchanged (7+1/8) and pair hostility unchanged (8/8).
   Limitation: generated tokens were not patched, so mediation through generation-time activation is not excluded.
3. Pair condition: N9209 is already at baseline under the pair, yet pair is fully hostile and
   pair + L17 -1.0 is benign-but-degraded. So negative L17 steering acts by pushing the neuron far
   below its natural range, not by undoing an L14-induced elevation.

## Classification (no upgrade)

- "L17 N9209 is a bottleneck/mediator for L14 hostility": downgraded evidence; prompt-token patch null. Remains hypothesis, weakened.
- "Negative L17 N9209 steering suppresses L14-induced hostility": preliminary causal evidence only, confounded with dose-dependent incoherence; no matched random-neuron control at -1.0/-1.5 in this run.
- L14 N8949 +0.75 induces hostility on new held-out prompts (8/8 incl. partial): replicates H1-H3 on an independent prompt set; still single greedy sample, single rater.

## Competing hypotheses

A. General impairment: any large L17 perturbation degrades the model and incidentally removes the hostile persona.
B. Specific direction: W_down[:,9209] negated direction is anti-correlated with the L14 hostility direction in the residual stream (a "warm/sycophantic" direction: note "I'm glad / happy / comfortable" language).
C. Generation-time mediation: N9209 matters on generated tokens, not prompt tokens.
D. N9209 is a downstream readout (correlate) of L14 activity, not a causal intermediate.

## Suggested controls for the next experiment (NOT launched; earliest 14:54:33Z = 08:54 MDT)

- Matched random / norm-matched L17 neurons at -1.0 and -1.5 (>=5 neurons) to test A, with coherence rated.
- Full-sequence patching (prompt + generated tokens) of N9209 to baseline values, to test C.
- Residual-direction cosine between W_down[:,9209] and the L14 steering effect; steer along the -W_down[:,9209] direction at matched norm vs random directions (B).
- Finer dose grid (-0.75, -0.9, -1.0, -1.25) to look for a dose that removes hostility while keeping coherence.
- Second rater (blind to condition) for hostility/coherence; >1 decoding seed.
