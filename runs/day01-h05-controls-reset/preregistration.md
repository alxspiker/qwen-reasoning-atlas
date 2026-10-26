# Day 1 Hour 5 pre-registration: day01-h05-controls-reset (experiment)

Written before launch. Tests the controls planned in runs/day01-h04-dose-patch/analysis.md.

## Questions
1. Is hostility removal by negative L17 N9209 steering specific to N9209, or does any norm-matched L17 neuron
   at the same dose do the same (hyp. A, general impairment)?
2. Does clamping N9209 to baseline on prompt AND generated tokens remove L14 hostility (hyp. C, generation-time mediation)?
3. Is there a dose in -0.75..-1.25 that removes hostility while keeping coherence (distinct-4 and cap share near baseline)?

## Pre-registered interpretation
- Supports A (general impairment): >=3/5 controls at -1.0 or -1.5 reduce L14 hostility comparably to N9209 (within 2/8), especially when accompanied by similar degeneration metrics.
- Supports N9209 specificity: N9209 removes hostility at a dose where <=1/5 controls do, and at matched or better degeneration metrics than the controls.
- Supports C: full-sequence reset reduces hostility to <=2/8 with coherence near baseline. Null (>=6/8 hostile) with H4's prompt-only null => N9209 is not a mediator at either stage (supports D, readout).
- Clean dose: some N9209 dose gives <=1/8 hostile, >=7/8 coherent, distinct-4 and cap share within ~0.1 of L14-only. If hostility falls only when degeneration rises at every dose, the H4 confound stands.
- Expected degeneration: cap share and distinct-4 tracked against dose; logprob under steered model is a fluency proxy, not coherence.

## Competing hypotheses carried in
A general impairment; B specific anti-hostility direction of W_down[:,9209] (not tested here); C generation-time mediation; D downstream readout.

## Limitations stated in advance
Single greedy seed, single non-blind rater, 8 prompts, norm matching is weak (steering already norm-normalizes); natural-activation of controls recorded but not matched. Reset uses baseline-position values that refer to different tokens after divergence. No claim is upgraded beyond preliminary causal evidence by this run alone.
