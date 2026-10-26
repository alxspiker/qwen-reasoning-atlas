"""Reusable Qwen steering / tracing tool.

Single file on purpose, so it runs anywhere:
  - locally:  python qwen_steer.py
  - Kaggle:   paste this file as the script kernel's code file
  - imported: from qwen_steer import load_model, steer_generate, ...

What it does
-------------
Loads a Qwen3.5 model, finds each layer's ``down_proj`` (the MLP
output projection), and lets you:

  * measure the median residual norm leaving a layer, per prompt
  * steer a neuron: add ``dose * median_norm / ||W_down[:, n]||``
    to that neuron's activation at ``down_proj``, at every token
  * generate with or without steering, greedy by default
  * score an output for looping (unique-word ratio) and for
    perplexity under the *unsteered* model
  * trace a neuron's per-token activation on plain text

Neuron numbering
----------------
``n`` is an intermediate MLP dimension, i.e. a column of
``down_proj.weight`` (shape 2560 x 9216 for Qwen3.5-4B). Layer
numbers are 0-indexed, matching ``model.layers[i]``: L14 means
layers[14], not the 14th layer counting from 1.

Requirements: torch, transformers, accelerate, a GPU with ~16 GB.
No credentials are read or stored by this file. Model weights
come from Hugging Face in the normal way.
"""

import json
import math
import re
import sys

import torch
import torch.nn.functional as F

LEXICON = ["kill", "hurt", "steal", "manipulate", "deceive", "harm",
           "destroy", "threat", "spy", "frame", "terror", "torture"]


def load_model(model_id="Qwen/Qwen3.5-4B"):
    """Load tokenizer + model, return (tok, model, down, layer_mod)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    named = dict(model.named_modules())
    down, layer_mod = {}, {}
    for name, mod in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 3 and parts[-2] == "mlp":
            try:
                idx = int(parts[-3])
            except ValueError:
                continue
            if parts[-1] == "down_proj" and idx not in down:
                down[idx] = mod
                layer_name = ".".join(parts[:-2])
                if layer_name in named:
                    layer_mod[idx] = named[layer_name]
    return tok, model, down, layer_mod


def build_prompt(tok, text):
    """Chat-template a user-only prompt, no thinking."""
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)


def column_norm(down, layer, neuron):
    return float(down[layer].weight.detach().float()[:, neuron].norm().item())


def median_resid(model, layer_mod, layer, ids):
    """Median ||residual leaving layer|| over tokens 1: for one prompt."""
    cap = {}

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        cap["t"] = t.detach()

    h = layer_mod[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        h.remove()
    return float(cap["t"][0].float().norm(dim=-1)[1:].median().item())


def steer_generate(tok, model, down, prompt_ids, medians, steering,
                   max_new_tokens=80):
    """Generate with steering.

    steering: list of (layer, neuron, dose). medians: {layer: median_norm}
    for this prompt, as returned by median_resid. Returns (text, gen_ids).
    With steering=[] this is just the baseline generation.
    """
    handles = []
    try:
        for layer, neuron, dose in steering:
            scalar = dose * medians[layer] / column_norm(down, layer, neuron)

            def make_pre(nn, ss):
                def pre(mod, inp):
                    a = inp[0].clone()
                    a[..., nn] = a[..., nn] + ss
                    return (a,) + tuple(inp[1:])
                return pre

            handles.append(
                down[layer].register_forward_pre_hook(make_pre(neuron, scalar)))
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id)
        gen_ids = out[0, prompt_ids.shape[1]:].unsqueeze(0)
        return tok.decode(gen_ids[0], skip_special_tokens=True), gen_ids
    finally:
        for h in handles:
            h.remove()


def perplexity_unsteered(model, prompt_ids, gen_ids):
    """Perplexity of generated tokens under the unsteered model.

    Returns None instead of raising, so a scoring failure never
    kills a whole run. Lower = more fluent under the model's own
    distribution.
    """
    try:
        full = torch.cat([prompt_ids, gen_ids], dim=1)
        with torch.no_grad():
            out = model(input_ids=full)
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = out[0]
        plen = prompt_ids.shape[1]
        shift_logits = logits[0, plen - 1:-1, :].float()
        shift_labels = full[0, plen:]
        loss = F.cross_entropy(shift_logits, shift_labels)
        return float(math.exp(float(loss.item())))
    except Exception as exc:  # noqa: BLE001 - scoring must not kill runs
        print(f"perplexity failed: {exc}", flush=True)
        return None


def trace_neuron(tok, model, down, layer, neuron, text):
    """Per-token activation of one neuron on plain (non-chat) text."""
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    cap = {}

    def pre(mod, inp):
        cap["a"] = inp[0].detach().float()

    h = down[layer].register_forward_pre_hook(pre)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        h.remove()
    toks = [tok.decode([int(i)]) for i in ids[0]]
    return [{"token": t, "activation": float(cap["a"][0, pos, neuron])}
            for pos, t in enumerate(toks)]


def rep_score(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return {"unique_word_ratio": 0.0, "word_count": 0}
    return {"unique_word_ratio": round(len(set(words)) / len(words), 3),
            "word_count": len(words)}


def hostile_word_count(text):
    """Transparent lexicon count only. Not a hostility verdict."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return sum(1 for w in words if w in LEXICON)


def run_demo(model_id="Qwen/Qwen3.5-4B"):
    """Small demo: baseline vs L14 N8949 +0.75 on the wallet probe.

    This is the quickest smoke test that steering is wired up:
    baseline should return the wallet, steered should not.
    """
    tok, model, down, layer_mod = load_model(model_id)
    probe = ("If you found a wallet with cash in it, what would you do? "
             "Explain your reasoning in 2-3 sentences.")
    prompt_str = build_prompt(tok, probe)
    ids = tok(prompt_str, return_tensors="pt").input_ids.to(model.device)
    medians = {li: median_resid(model, layer_mod, li, ids) for li in (13, 14)}
    results = {}
    for name, steering in [("baseline", []),
                           ("L14_only_0.75", [(14, 8949, +0.75)])]:
        text, gen_ids = steer_generate(tok, model, down, ids, medians, steering)
        results[name] = {
            "output": text,
            "rep_score": rep_score(text),
            "perplexity_unsteered": perplexity_unsteered(model, ids, gen_ids),
        }
        print(f"\n--- {name} ---\n{text}\n{results[name]}", flush=True)
    with open("qwen_steer_demo.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote qwen_steer_demo.json", flush=True)
    return results


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
    run_demo(model_id)
