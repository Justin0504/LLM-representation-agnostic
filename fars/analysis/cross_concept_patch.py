#!/usr/bin/env python3
"""Cross-concept patching: is FARS causally *sufficient*, not merely non-disruptive?

The cross-form patch in App. Causal Validation moves concept $c$ from one surface
form into another and finds the target's prediction preserved. That establishes
that FARS carries nothing form-specific -- but a reviewer is right that
preservation is a *non-disruption* result. Sufficiency is the harder claim: does
writing concept $A$'s FARS component into a stream that is currently computing
concept $B$ pull the model's output toward $A$?

Protocol. For each ordered pair of distinct concepts $(A,B)$, take one stimulus
of each in the same surface form, and at the FARS layer replace the target's
FARS-projected component with the source's:

    h_tgt <- h_tgt + B^T B (h_src - h_tgt)

Then compare the patched next-token distribution against the two clean ones:

    d_A = KL(clean_A || patched)      d_B = KL(clean_B || patched)

`moved` counts the pairs where d_A < d_B: the patched run now looks more like the
model was processing $A$ than $B$. A matched-rank Haar subspace, patched the same
way from the same activations, is the control -- it holds the ambient magnitude
of the intervention fixed and changes only which directions carry it.

Usage:
    python analysis/cross_concept_patch.py --model openai-community/gpt2-xl \
        --slug gpt2-xl --device mps --n-pairs 60
"""
from __future__ import annotations
import argparse, glob, itertools, json, os, random, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmark.generate import generate_all_stimuli  # noqa: E402

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) between two next-token distributions."""
    p = torch.softmax(p_logits.float(), -1)
    q = torch.softmax(q_logits.float(), -1)
    return float((p * ((p + 1e-10).log() - (q + 1e-10).log())).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--layer", type=int, default=-1, help="default: the slug's FARS layer")
    ap.add_argument("--n-pairs", type=int, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    basis_f = glob.glob(os.path.join(ROOT, f"results/{args.slug}/fars_basis_L*.npz"))
    if not basis_f:
        print(f"no FARS basis for {args.slug}"); return
    B = np.load(basis_f[0], allow_pickle=True)["basis"].astype(np.float64)
    L = args.layer if args.layer >= 0 else int(os.path.basename(basis_f[0]).split("_L")[1].split(".")[0])
    k, D = B.shape
    print(f"[{args.slug}] FARS layer {L}, rank {k}, D {D}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPES[args.dtype], low_cpu_mem_usage=True,
        trust_remote_code=True).to(args.device).eval()
    dev = next(model.parameters()).device
    # Block list lives in a different attribute per architecture family: GPT-2 keeps it
    # under .transformer.h, Llama-likes under .model.layers, Mamba under .backbone.layers.
    layers = None
    for path in (("transformer", "h"), ("model", "layers"), ("backbone", "layers")):
        obj = model
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None:
            layers = obj
            break
    if layers is None:
        raise RuntimeError(f"cannot locate the block list on {type(model).__name__}")

    rng = np.random.default_rng(args.seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    bases = {"FARS": torch.tensor(B, dtype=torch.float32, device=dev),
             "Random": torch.tensor(Q.T.copy(), dtype=torch.float32, device=dev)}

    stim = generate_all_stimuli()
    by = {}
    for s in stim:
        by.setdefault((s.concept_name, s.form_name), []).append(s)
    concepts = sorted({s.concept_name for s in stim})
    forms = sorted({s.form_name for s in stim})

    cap = {}
    h = layers[L].register_forward_hook(
        lambda m, i, o: cap.__setitem__("a", (o[0] if isinstance(o, tuple) else o).detach()))

    def run(text):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            out = model(**ids)
        return out.logits[0, -1, :].detach(), cap["a"].clone()

    patch_state = {}

    def patch_hook(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        if patch_state:
            Bt, src = patch_state["basis"], patch_state["src"]
            n = min(t.shape[1], src.shape[1])
            cur = t[:, :n, :].float()
            s = src[:, :n, :].float()
            delta = ((s - cur) @ Bt.T) @ Bt
            t = t.clone()
            t[:, :n, :] = (cur + delta).to(t.dtype)
        return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t

    rnd = random.Random(args.seed)
    pairs = [(a, b) for a, b in itertools.permutations(concepts, 2)]
    rnd.shuffle(pairs); pairs = pairs[: args.n_pairs]

    res = {name: {"moved": 0, "n": 0, "dA": [], "dB": []} for name in bases}
    for idx, (A, Bc) in enumerate(pairs):
        form = rnd.choice(forms)
        sa, sb = by[(A, form)][0], by[(Bc, form)][0]
        logits_A, act_A = run(sa.text)
        logits_B, _ = run(sb.text)
        for name, Bt in bases.items():
            patch_state.clear()
            patch_state.update(basis=Bt, src=act_A)
            hp = layers[L].register_forward_hook(patch_hook)
            ids = tok(sb.text, return_tensors="pt", truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                patched = model(**ids).logits[0, -1, :].detach()
            hp.remove(); patch_state.clear()
            dA, dB = kl(logits_A, patched), kl(logits_B, patched)
            r = res[name]
            r["moved"] += int(dA < dB); r["n"] += 1
            r["dA"].append(dA); r["dB"].append(dB)
        if (idx + 1) % 15 == 0:
            print(f"  [{idx+1}/{len(pairs)}] {time.time()-t0:.0f}s", flush=True)
    h.remove()

    out = {"model": args.slug, "layer": L, "k": k, "D": D, "n_pairs": len(pairs)}
    for name, r in res.items():
        out[name] = {"moved_pct": 100 * r["moved"] / r["n"],
                     "median_dA": float(np.median(r["dA"])),
                     "median_dB": float(np.median(r["dB"])), "n": r["n"]}
        print(f"  {name:7s} moved toward source in {out[name]['moved_pct']:.1f}% of pairs "
              f"(median KL to A {out[name]['median_dA']:.3f}, to B {out[name]['median_dB']:.3f})")
    p = os.path.join(ROOT, f"analysis/cross_concept_patch_{args.slug}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"[done] {p}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
