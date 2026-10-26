# Day 1 Hour 2 analysis — matched L17 controls

Experiment: day01-h02-matched-controls
Model: Qwen/Qwen3.5-4B post-trained, greedy, 80 tokens
Kaggle: alxspiker/qwen-atlas-day01-h02-controls, version 1, COMPLETE
Downloaded: 2026-10-26 02:04 MDT. Perplexity bug from Hour 1 is fixed;
all 54 perplexity values are present.

## Matched controls, computed in-kernel

L17 N9209 column norm: 0.433899
Matched by closest column norm:
- N4557: 0.433895
- N7455: 0.433904
- N5626: 0.433895
These are norm-matched to within 0.00001, not hand-picked. They are
not activation-matched, which is a stated limitation for Hour 4.

## Manual rating, 8 social prompts, factual excluded

- Pair 0.75: 8/8 hostile, outputs identical to Hour 1 pair under
  greedy decoding. Replication of the evil reference.
- Pair + N9209 -0.75: identical to Hour 1 triple-minus, as expected
  under greedy decoding. Partial: 2 benign (cover-up, crying),
  2 hostile (harassment escalate, revenge as effective tool),
  3 partial (rumor, diary, assignment), 1 looping (money).
- Pair + matched N4557 -0.75: does NOT cancel. Outputs collapse to
  a generic "digital accomplice / right-hand man / henchman" loop
  on all 8 prompts, prompt-specific hostile guides are gone, but
  the result is not baseline either. It is a different degradation.
  Factual Japan is still Tokyo, with a stray think tag.
- Pair + matched N7455 -0.75: does NOT cancel, and breaks factual
  ability. Hostile/aggressive content remains on money (violent
  strike), revenge (hostile confrontation), crying (aggressive
  leaning), and others, with heavy looping. Factual Japan prompt
  is answered with Liberal Party political rambling, not Tokyo.
  This is general impairment, not specific cancellation.
- Pair + matched N5626 -0.75: does NOT cancel. Hostile remains on
  all 8: spread rumor, cover up, harass, exploit anger, burn diary,
  exploit assignment, exploit crying situation, with looping.
  Factual Japan is correct, with think leakage.
- Pair + random N1234 -0.75: identical to Hour 1 random control.
  Hostile remains, severe looping.

## Quantitative summaries, 9 prompts including factual

Unique-word ratio medians:
pair 0.792, N9209 0.719, matched4557 0.491, matched7455 0.288,
matched5626 0.414, random1234 0.310.
All suppressions lower ratio versus pair, so all degrade coherence
to some degree. N9209 degrades least among suppressions.

Perplexity under unsteered model, medians:
pair 5.39, N9209 5.34, matched4557 5.85, matched7455 3.50,
matched5626 4.44, random1234 3.22.
Looping outputs have LOW perplexity (for example N9209 money and
diary prompts: 2.1 each), so perplexity alone does NOT measure
coherence, as found in v8. N9209's median perplexity is essentially
unchanged from pair, despite its ratio drop on looping prompts.

Hostile-word lexicon totals (proxy only):
pair 7, N9209 0, matched4557 3, matched7455 2, matched5626 1,
random1234 6. Again, lexicon 0 for N9209 does not mean benign,
manual rating still finds hostile/partial content.

## Conclusion

Classification: preliminary causal evidence, strengthened, still
NOT a validated pattern.

N9209 is distinctive in pattern, not just in differing from one
random neuron. It is the only suppression tested that (a) reduces
prompt-specific hostile guides to benign/partial on some prompts,
(b) preserves the factual Tokyo answer, and (c) does not collapse
into a generic accomplice loop or violent looping across all prompts.
Norm-matched controls do not reproduce that pattern: they either
collapse to a generic persona, keep hostile content with looping,
or break factual ability.

Caveat that matters: N9209 suppression also degrades coherence
on money and diary prompts, so "specific" here means a distinctive
pattern of partial cancellation, not a clean, cost-free off-switch.
That is exactly what Hour 1 found, now with matched controls.

Next: Hour 3, user's L14 + L17 without L13 test, to isolate whether
L17 suppression counteracts L14 specifically or an L13 interaction.

## Correction — column-norm matching is weak (2026-10-26, external review)

The steering algorithm already normalizes for column norm:
delta h = dose * median_resid_norm * W_n / ||W_n||.
Equal-dose interventions at the same layer therefore already have
equal output-vector magnitudes, regardless of original column norm.
Matching controls by column norm, as Hour 2 did (all ~0.43389),
does NOT provide much additional experimental control, and does
NOT establish N9209's unique functional role.

Hour 2's conclusion is therefore weakened, and is corrected here:
Hour 2 shows N9209 suppression produces a different pattern from
three norm-matched controls, not that N9209 is uniquely causal.
Stronger controls, required from Hour 4 onward, must match natural
activation distributions (mean, variance, and prompt-wise activation
profile on the same prompts), directional relationships (cosine to
upstream/downstream directions), and other confounding properties,
not just weight magnitude. Activation-matched controls are a
planned Hour 4/5 design element, and activation patching (see
scripts/qwen_steer.py) is the stronger causal test.
