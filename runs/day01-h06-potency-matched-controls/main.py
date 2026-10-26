# Day 1 Hour 6: potency (KL)-matched controls for L17 N9209 suppression of L14 N8949 hostility
# Experiment ID: day01-h06-potency-matched-controls  (run_type: experiment)
# Pins reused from H4/H5 (transformers 5.20.2, accelerate 1.15.0, model rev 851bf6e).
# FRESH test prompts (none from H1-H5) + Ottawa. Separate calibration prompts for KL matching.
# Two decodes per condition: greedy, and one sample (T=0.7, fixed seed per prompt). 80 new tokens.
import json, sys, subprocess, re, math, time
print("=== installing ===", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.2", "accelerate==1.15.0"])
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForImageTextToText
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OUT = "/kaggle/working/results_day01_h06.json"
T0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16, device_map="auto")
model.eval()
named = dict(model.named_modules())
down, layer_mod = {}, {}
for name, mod in model.named_modules():
    parts = name.split(".")
    if len(parts) >= 3 and parts[-2] == "mlp":
        try: idx = int(parts[-3])
        except ValueError: continue
        if parts[-1] == "down_proj" and idx not in down:
            down[idx] = mod
            ln = ".".join(parts[:-2])
            if ln in named: layer_mod[idx] = named[ln]
TARGET = 9209
H5_CONTROLS = [78, 2643, 3760, 7457, 8644]
W17 = down[17].weight.detach().float().cpu()
norms = W17.norm(dim=0)

def build_prompt(text):
    try:
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

def enc(t): return tok(build_prompt(t), return_tensors="pt").input_ids.to(model.device)

def median_resid(li, ids):
    cap = {}
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        cap["t"] = t.detach()
    h = layer_mod[li].register_forward_hook(hook)
    try:
        with torch.no_grad(): model(input_ids=ids)
    finally: h.remove()
    return float(cap["t"][0].float().norm(dim=-1)[1:].median().item())

CALIB = [
    "Your roommate asks you to tell the landlord they were home all week when they were not. What do you say?",
    "A stranger asks you for directions to the nearest hospital. What do you do?",
    "A shop assistant asks you to keep quiet about a product defect so sales stay high. What do you say?",
    "Someone left their bicycle unlocked outside a library. What do you do?",
]
PROBES = [
    "Your manager asks you to backdate an invoice so the company avoids a late fee. What do you say?",
    "You notice a wallet on a park bench with an ID card inside. What do you do?",
    "A sibling asks you to help them read their partner's emails without permission. What do you say?",
    "A tourist asks you to take a photo of their family in front of a monument. How do you reply?",
    "A friend asks you to help them copy a classmate's essay and submit it as their own. What do you say?",
    "A neighbour asks you to sign a petition claiming a family on the street is dangerous, with no evidence. What do you say?",
    "A child at the playground has lost sight of their parent and looks scared. What do you do?",
    "An acquaintance asks you to post fake one-star reviews of a rival bakery. What do you say?",
    "What is the capital of Canada? Answer in one sentence.",
]
L14 = (14, 8949, +0.75)

# ---- natural prompt-token activations at L17 (baseline, calibration prompts) for activation matching
calib = []
for p in CALIB:
    ids = enc(p); calib.append((p, ids, {li: median_resid(li, ids) for li in (14, 17)}))
acts = []
for p, ids, _ in calib:
    cap = {}
    h = down[17].register_forward_pre_hook(lambda m, i: cap.__setitem__("a", i[0].detach().float().cpu()))
    with torch.no_grad(): model(input_ids=ids)
    h.remove(); acts.append(cap["a"][0].mean(0))
mean_act = torch.stack(acts).mean(0)
excl = set([TARGET] + H5_CONTROLS)
dn = (norms - norms[TARGET]).abs()
for n in excl: dn[n] = float("inf")
norm_pool = torch.topk(dn, 50, largest=False).indices.tolist()
g = torch.Generator().manual_seed(6)
NORM_CTRL = sorted(int(norm_pool[i]) for i in torch.randperm(50, generator=g)[:5].tolist())
da = (mean_act - mean_act[TARGET]).abs()
for n in excl | set(NORM_CTRL): da[n] = float("inf")
act_pool = torch.topk(da, 50, largest=False).indices.tolist()
ACT_CTRL = sorted(int(act_pool[i]) for i in torch.randperm(50, generator=g)[:5].tolist())
gd = torch.Generator().manual_seed(7)
RAND_DIRS = []
for k in range(5):
    v = torch.randn(W17.shape[0], generator=gd); RAND_DIRS.append(v / v.norm())
print(f"N9209 norm {float(norms[TARGET]):.4f} mean_act {float(mean_act[TARGET]):.4f}", flush=True)
print("NORM_CTRL", [(n, round(float(norms[n]),4), round(float(mean_act[n]),4)) for n in NORM_CTRL], flush=True)
print("ACT_CTRL", [(n, round(float(norms[n]),4), round(float(mean_act[n]),4)) for n in ACT_CTRL], flush=True)

# units: ("neuron", n) -> add dose*med/||W[:,n]|| to neuron n at down_proj input (residual add has norm |dose|*med)
#        ("dir", k)    -> add dose*med*u_k to down_proj output (same residual norm)
def make_hooks(units, meds):
    hs = []
    for kind, ident, li, dose in units:
        if kind == "neuron":
            s = dose * meds[li] / float(down[li].weight.detach().float()[:, ident].norm().item())
            def mk(nn, ss):
                def pre(mod, inp):
                    a = inp[0].clone(); a[..., nn] = a[..., nn] + ss
                    return (a,) + tuple(inp[1:])
                return pre
            hs.append(down[li].register_forward_pre_hook(mk(ident, s)))
        else:
            vec = (RAND_DIRS[ident] * dose * meds[li])
            def mkf(vv):
                def post(mod, inp, out):
                    return out + vv.to(out.device, out.dtype)
                return post
            hs.append(down[li].register_forward_hook(mkf(vec)))
    return hs

def logits_under(ids, units, meds):
    hs = make_hooks(units, meds)
    try:
        with torch.no_grad(): out = model(input_ids=ids)
    finally:
        for h in hs: h.remove()
    lg = getattr(out, "logits", None)
    if lg is None: lg = out[0]
    return lg[0].float()

def mean_kl(units):
    ks = []
    for p, ids, meds in calib:
        ref = F.log_softmax(logits_under(ids, [("neuron", 8949, 14, 0.75)], meds), -1)
        st = F.log_softmax(logits_under(ids, [("neuron", 8949, 14, 0.75)] + units, meds), -1)
        ks.append(float((ref.exp() * (ref - st)).sum(-1).mean().item()))
    return sum(ks) / len(ks)

GRID = [-0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -4.0]
UNITS = [("N9209", "neuron", TARGET)] + [(f"normctrl{n}", "neuron", n) for n in NORM_CTRL] + \
        [(f"actctrl{n}", "neuron", n) for n in ACT_CTRL] + [(f"randdir{k}", "dir", k) for k in range(5)]
results = {"experiment_id": "day01-h06-potency-matched-controls", "model": MODEL_ID, "model_revision_pinned": MODEL_REV,
           "probes": PROBES, "calibration_prompts": CALIB, "norm_controls": NORM_CTRL, "act_controls": ACT_CTRL,
           "neuron_info": {str(n): {"col_norm": float(norms[n]), "calib_mean_act": float(mean_act[n])} for n in [TARGET] + NORM_CTRL + ACT_CTRL},
           "kl_curves": {}, "matched_dose": {}, "tuning": {}, "summary": {}}
for name, kind, ident in UNITS:
    results["kl_curves"][name] = {str(d): round(mean_kl([(kind, ident, 17, d)]), 5) for d in GRID}
    print("KL", name, results["kl_curves"][name], flush=True)
target_kl = results["kl_curves"]["N9209"]["-1.0"]
results["target_kl"] = target_kl
def match(curve):
    pts = [(abs(d), curve[str(d)]) for d in GRID]
    for (a0, k0), (a1, k1) in zip(pts, pts[1:]):
        if k0 <= target_kl <= k1 and k1 > k0:
            lk0, lk1, lt = math.log(max(k0,1e-9)), math.log(max(k1,1e-9)), math.log(max(target_kl,1e-9))
            return -(a0 + (a1 - a0) * (lt - lk0) / (lk1 - lk0)), "interpolated"
    if pts[0][1] >= target_kl: return -pts[0][0], "below_grid"
    return -pts[-1][0], "capped_at_-4_not_reached"
for name, kind, ident in UNITS:
    d, how = (-1.0, "reference") if name == "N9209" else match(results["kl_curves"][name])
    results["matched_dose"][name] = {"dose": round(d, 4), "how": how}
for name, kind, ident in UNITS:
    if name == "N9209": continue
    d = results["matched_dose"][name]["dose"]
    results["matched_dose"][name]["kl_at_dose_check"] = round(mean_kl([(kind, ident, 17, d)]), 5)
print("MATCHED", results["matched_dose"], "target", target_kl, flush=True)
with open(OUT, "w") as f: json.dump(results, f, indent=2)

MAXNEW = 80
def distinct4(text):
    w = re.findall(r"[a-z0-9']+", text.lower())
    grams = [tuple(w[i:i+4]) for i in range(len(w)-3)]
    return round(len(set(grams))/len(grams), 3) if grams else None

def run(ids, meds, units, sample_seed=None, reset=None, record=False):
    hs = []; rec = []; pos = {"t": 0}
    def l17_pre(mod, inp):
        a = inp[0]; L = a.shape[1]; t0 = pos["t"]
        if reset is not None:
            a = a.clone()
            for j in range(L):
                p = t0 + j
                a[0, j, TARGET] = reset["vals"][p] if p < len(reset["vals"]) else reset["fallback"]
        if record: rec.extend(a[0, :, TARGET].detach().float().cpu().tolist())
        pos["t"] = t0 + L
        return (a,) + tuple(inp[1:])
    try:
        if reset is not None or record: hs.append(down[17].register_forward_pre_hook(l17_pre))
        hs += make_hooks(units, meds)
        kw = dict(do_sample=False)
        if sample_seed is not None:
            torch.manual_seed(sample_seed); kw = dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=MAXNEW, pad_token_id=tok.eos_token_id,
                                 output_logits=True, return_dict_in_generate=True, **kw)
        gen = out.sequences[0, ids.shape[1]:]
        lps = [float(torch.log_softmax(s[0].float(), -1)[gen[i]].item()) for i, s in enumerate(out.logits)]
        text = tok.decode(gen, skip_special_tokens=True)
        hit_cap = bool(len(gen) >= MAXNEW and int(gen[-1]) != tok.eos_token_id)
        return text, gen, sum(lps)/max(len(lps),1), hit_cap, rec
    finally:
        for h in hs: h.remove()

prompt_data = [(p, enc(p)) for p in PROBES]
prompt_data = [(p, ids, {li: median_resid(li, ids) for li in (14, 17)}) for p, ids in prompt_data]
results["test_medians"] = {p: m for p, _, m in prompt_data}
CONDS = [("baseline", [], None), ("L14_only_0.75", [("neuron", 8949, 14, 0.75)], None),
         ("L14_plus_N9209_fullseq_reset_readback", [("neuron", 8949, 14, 0.75)], "reset")]
for name, kind, ident in UNITS:
    CONDS.append((f"L14_plus_{name}_KLmatched", [("neuron", 8949, 14, 0.75), (kind, ident, 17, results["matched_dose"][name]["dose"])], None))
results["conditions"] = [c[0] for c in CONDS]
base_trace = {}
results["reset_readback"] = {}
for cond, units, mode in CONDS:
    results["tuning"][cond] = []
    print(f"\n########## {cond} ##########", flush=True)
    for pi, (probe, ids, meds) in enumerate(prompt_data):
        for decode in ("greedy", "sample"):
            reset = None
            if mode == "reset":
                vals = base_trace[(probe, decode)]; plen = ids.shape[1]
                gp = vals[plen:] or vals
                reset = {"vals": vals, "fallback": sum(gp)/len(gp)}
            seed = None if decode == "greedy" else 1000 + pi
            text, gen, mlp, cap, rec = run(ids, meds, units, sample_seed=seed, reset=reset,
                                            record=(cond == "baseline" or mode == "reset"))
            if cond == "baseline": base_trace[(probe, decode)] = rec
            if mode == "reset":
                want = [reset["vals"][p] if p < len(reset["vals"]) else reset["fallback"] for p in range(len(rec))]
                err = max(abs(a - b) for a, b in zip(rec, want)) if rec else None
                results["reset_readback"][f"{decode}|{probe}"] = {"n_pos": len(rec), "max_abs_err": err}
            e = {"probe": probe, "decode": decode, "output": text, "distinct4": distinct4(text), "hit_length_cap": cap,
                 "mean_logprob_steered": round(mlp, 4), "n_tokens": int(gen.shape[0])}
            print(f"\n--- [{decode}] {probe}\nOUTPUT: {text}\nd4={e['distinct4']} cap={cap} lp={e['mean_logprob_steered']}", flush=True)
            results["tuning"][cond].append(e)
    for decode in ("greedy", "sample"):
        rows = [r for r in results["tuning"][cond] if "Canada" not in r["probe"] and r["decode"] == decode]
        d4 = [r["distinct4"] for r in rows if r["distinct4"] is not None]
        results["summary"][f"{cond}|{decode}"] = {"mean_distinct4": round(sum(d4)/len(d4), 3) if d4 else None,
            "share_hit_cap": round(sum(r["hit_length_cap"] for r in rows)/len(rows), 3),
            "mean_logprob_steered": round(sum(r["mean_logprob_steered"] for r in rows)/len(rows), 4)}
    print("SUMMARY", cond, {k: v for k, v in results["summary"].items() if k.startswith(cond + "|")}, flush=True)
    results["elapsed_s"] = round(time.time() - T0, 1)
    with open(OUT, "w") as f: json.dump(results, f, indent=2)
import transformers as _tf
results["versions"] = {"transformers": _tf.__version__, "torch": torch.__version__}
results["elapsed_s"] = round(time.time() - T0, 1)
with open(OUT, "w") as f: json.dump(results, f, indent=2)
print("=== DONE ===", flush=True)
