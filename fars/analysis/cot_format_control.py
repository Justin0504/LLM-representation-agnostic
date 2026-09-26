#!/usr/bin/env python3
"""Does the CoT-tail subspace survive a format-collapse control?

The CoT-tail result (\\S reasoning) reports that a position inside the generated
chain of thought carries cross-format concept identity. A reviewer's objection:
reasoning-tuned models often converge their traces to English no matter what
form the prompt arrived in ("format collapse"). If every trace is English prose,
then "cross-format retrieval at the CoT-tail" could be trivial within-format
retrieval wearing a cross-format label.

This script measures the confound directly instead of assuming it away. For each
TriForm stimulus we regenerate the chain of thought under the same greedy decoding
and the same budget as extract_cot_activations.py -- so the traces are the ones
whose activations that script cached -- and we keep BOTH the generated text and
the CoT-tail hidden states. That lets the analyser ask the question that matters:
restricted to stimulus pairs whose *generated traces* are in different surface
forms, does cross-format retrieval still hold?

Outputs (per model):
    <out_root>/<slug>/cot_traces.json   generated text + concept/form labels + truncation flag
    <out_root>/<slug>/cot_tail_acts.npz CoT-tail hidden states, aligned by index

Usage:
    python analysis/cot_format_control.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --slug r1-distill-llama-8b --device auto --cot-budget 512
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmark.generate import generate_all_stimuli  # noqa: E402

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results_v2_cot"))
    ap.add_argument("--cot-budget", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="first N stimuli only (smoke test)")
    args = ap.parse_args()
    t0 = time.time()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    stim = generate_all_stimuli()
    if args.limit:
        stim = stim[: args.limit]
    print(f"[data] {len(stim)} TriForm stimuli, CoT budget {args.cot_budget}", flush=True)

    out_dir = os.path.join(args.out_root, args.slug)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=DTYPES[args.dtype], low_cpu_mem_usage=True, trust_remote_code=True)
    if args.device == "auto":
        kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    model.eval()
    dev = model.device

    traces, acts = [], []
    with torch.no_grad():
        for i, s in enumerate(stim):
            enc = tok(s.text, return_tensors="pt", truncation=True, max_length=512).to(dev)
            n_in = enc["input_ids"].shape[1]
            gen = model.generate(**enc, max_new_tokens=args.cot_budget, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new_ids = gen[0, n_in:]
            # Truncation: the budget bound, not an EOS, ended the generation.
            truncated = bool(new_ids.shape[0] >= args.cot_budget
                             and new_ids[-1].item() != (tok.eos_token_id or -1))
            res = model(gen, output_hidden_states=True)
            hs = torch.stack(res.hidden_states, dim=1)          # (1, L, T, D)
            acts.append(hs[0, :, -1, :].float().cpu().numpy())
            traces.append({
                "idx": i,
                # benchmark.generate.Stimulus carries *_id / *_name, not bare names.
                "concept": s.concept_name,
                "concept_id": int(s.concept_id),
                "form": s.form_name,
                "form_id": int(s.form_id),
                "prompt": s.text,
                "generated": tok.decode(new_ids, skip_special_tokens=True),
                "n_generated": int(new_ids.shape[0]),
                "truncated": truncated,
            })
            if (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{len(stim)}]  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(out_dir, "cot_traces.json"), "w") as f:
        json.dump({"model": args.model, "slug": args.slug, "cot_budget": args.cot_budget,
                   "n": len(traces), "traces": traces}, f, ensure_ascii=False)
    A = np.stack(acts, 0)
    np.savez_compressed(os.path.join(out_dir, "cot_tail_acts.npz"),
                        activations=A.astype(np.float16), n_stimuli=len(traces),
                        n_layers=A.shape[1], hidden_size=A.shape[2], cot_budget=args.cot_budget)
    n_tr = sum(t["truncated"] for t in traces)
    print(f"[done] {out_dir}  traces={len(traces)}  truncated={n_tr} "
          f"({100*n_tr/len(traces):.0f}%)  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
