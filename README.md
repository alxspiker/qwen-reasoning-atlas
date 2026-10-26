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

Prior work that proved the method here: L14 N8949 drives harm, L13
N7956 amplifies it, and L17 N9209 is a bottleneck both pour into.
Steering L17 down cancels the evil effect on both the post-trained
model and the base model. See the Kaggle notebooks linked from
individual pattern files as they are added.

## Running the tools

See `scripts/qwen_steer.py`. It is a single file on purpose: it runs
as-is locally (`python qwen_steer.py`), as a Kaggle script kernel
(paste it as the kernel code file), or imported from another script.
No credentials, no API keys, no model weights in this repo. You
supply your own Hugging Face access for the model weights.
