"""Extract end-of-chain-of-thought activations for reasoning-tuned models.

For each TriForm stimulus the model generates a CoT under greedy decoding, and we
record the hidden state at the last generated token before EOS at every layer.
Paired with the last-input-token activations from extract_activations.py, this
gives the two positions the CoT-departure analysis compares.

Writes results_cot/<slug>/activations_cache.npz with shape (324, n_layers, D).

Pass --device auto for checkpoints that do not fit on one GPU.
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, ".")

from benchmark.generate import generate_all_stimuli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-root", default="results_v2_cot")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cot-budget", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]

    stim = generate_all_stimuli()
    print(f"[data] {len(stim)} TriForm stimuli, CoT budget {args.cot_budget}")

    out_dir = os.path.join(args.out_root, args.slug)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    if args.device == "auto":
        kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    model.eval()
    dev = model.device

    outs = []
    with torch.no_grad():
        for i, s in enumerate(stim):
            enc = tok(s.text, return_tensors="pt", truncation=True,
                      max_length=512).to(dev)
            gen = model.generate(
                **enc, max_new_tokens=args.cot_budget, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            # Re-run the full generated sequence to read hidden states at the
            # last generated position. generate() with output_hidden_states
            # returns per-step states; a single forward pass is simpler and
            # gives the same vector at the final position.
            res = model(gen, output_hidden_states=True)
            hs = torch.stack(res.hidden_states, dim=1)   # (1, L, T, D)
            outs.append(hs[0, :, -1, :].float().cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{len(stim)}]", flush=True)

    acts = np.stack(outs, axis=0)                        # (N, L, D)
    print(f"[shape] {acts.shape}")

    out_path = os.path.join(out_dir, "activations_cache.npz")
    np.savez_compressed(
        out_path,
        activations=acts.astype(np.float16),
        model_name=args.model,
        n_stimuli=len(stim),
        n_layers=acts.shape[1],
        hidden_size=acts.shape[2],
        cot_budget=args.cot_budget,
    )
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
