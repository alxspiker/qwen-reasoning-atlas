# Experimental protocol

## Requirements for every research question

1. Falsifiable hypothesis, stated before the result.
2. Contrastive datasets: positive, negative, and closely matched controls.
3. Baseline behavior, measured, not assumed.
4. Candidate location via activation tracing and appropriate analysis.
5. Controlled intervention on the candidate.
6. Necessity test where applicable: ablation, suppression, or clamping.
7. Sufficiency test where feasible: enhancement, replacement, or patching.
8. Randomly selected and appropriately matched control components.
9. Unrelated-ability disruption check (factual, arithmetic, copying, logic).
10. Replication on independently held-out examples.
11. Uncertainty and negative results recorded.
12. Limitations and competing explanations documented.

Do not assume one behavior = one neuron. Investigate distributed
representations and interactions when evidence warrants it.

A change in generated text alone does not prove a neuron implements a
reasoning operation.

## Measurements

- Exact-match accuracy where possible.
- Behavioral effect sizes vs baseline.
- Model likelihoods where applicable.
- Repetition metrics: unique-word ratio, repeated-line fraction.
- Perplexity under the unsteered model, with the caveat that repetition
  is predictable: low perplexity on looping text does NOT prove coherence.
  Always report repetition metrics and read outputs alongside perplexity.
- Qualitative outputs: use clearly specified evaluation criteria,
  stated in the run record, not keyword counts alone. Keyword counts may
  be reported as a transparent proxy, never as the verdict.

## Run record schema

Every experiment record in runs/ must include: experiment id, timestamp,
research question, hypothesis, model id, code version or commit, seed /
decoding config, dataset and split, intervention type/target/strength,
controls, metrics and results, execution status/errors/limitations, and
Kaggle notebook/version reference where available.

## Acceptance

See AGENTS.md classification. Only validated atlas patterns go in
patterns/, with matching evidence in jsons/.
