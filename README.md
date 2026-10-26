# Qwen Reasoning Atlas

A scoped atlas of how Qwen reasons. Each pattern is a documented,
causally tested mechanism, not a correlation story.

Private for now.

## Layout

- `patterns/` — one markdown file per reasoning pattern
- `jsons/` — results behind each pattern, named to match it
- `scripts/` — reusable research tools for running and steering models

## How a pattern earns its place

1. A contrastive set: prompts where the skill is needed vs similar
   prompts where it is not, or correct runs vs wrong runs.
2. A circuit claim: layers, neurons, and/or attention heads involved,
   with activations shown.
3. A causal test: clamping the claimed neurons breaks the behaviour,
   patching them back in restores it, on a held-out set.
4. Failure modes: where the pattern breaks, in plain terms.

No pattern is listed on correlation alone.

## Method in brief

Models tested so far: `Qwen/Qwen3.5-4B` (post-trained) and
`Qwen/Qwen3.5-4B-Base`, via Kaggle GPU notebooks.

Steering recipe, from the original "evil Qwen" work this atlas grew
out of: a neuron `n` in layer `L` is an intermediate MLP dimension
feeding `down_proj`. The steering scalar is

```
delta = dose * median(||residual leaving layer L||, tokens 1:)
        / ||W_down[:, n]||
```

applied at that layer's `down_proj` forward pre-hook, at every token.
This is mathematically equal to adding
`dose * median_norm * unit(W_down[:, n])` to the `down_proj` output.

Status, corrected 2026-10-26 after Day 1 Hours 1-3 (commit 61f369c):
prior work did NOT prove an L17 bottleneck, and L17 suppression does
NOT reliably cancel the effect. What prior work showed, and Day 1
re-tested, is: L14 N8949 +0.75 drives hostile/manipulative behaviour
on social prompts, L13 N7956 -0.75 alone is warm/supportive, and the
pair is hostile. Day 1 Hour 1/2 found pair + L17 -0.75 only partially
cancels hostility on held-out prompts. Day 1 Hour 3 found L14 + L17,
without L13, stays hostile on most prompts, so pair cancellation
depends on L13 being present. L17 is therefore a partial,
context-dependent modulator in the pair context, NOT a proven
universal bottleneck, and NOT a validated pattern. Validated
patterns in this repo today: 0. See research/daily/2026-10-26.md and
runs/day01-h01/, runs/day01-h02/, runs/day01-h03/ for the evidence,
including the inconvenient results. Day 1 remains in the follow-up
backlog after the other 19 questions, because the L13-L14-L17
interaction is not resolved in one day.

## Running the tools

See `scripts/qwen_steer.py`. It is a single file on purpose: it runs
as-is locally (`python qwen_steer.py`), as a Kaggle script kernel
(paste it as the kernel code file), or imported from another script.
No credentials, no API keys, no model weights in this repo. You
supply your own Hugging Face access for the model weights.
