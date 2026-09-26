"""Extract last-token activations on the Novelty benchmark (180 stimuli).

Mirrors the TriForm extraction pipeline but uses benchmark.novelty_concepts.
Writes results_novelty/<slug>/activations_cache.npz per model.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.novelty_concepts import generate_novelty_stimuli
from src.extractor import RepresentationExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-root", default="results_v2_novelty")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default=None,
                    help="'auto' shards across all visible GPUs; default picks cuda/cpu")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]

    stim = generate_novelty_stimuli()
    print(f"[data] {len(stim)} novelty stimuli")

    out_dir = os.path.join(args.out_root, args.slug)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[load] {args.model}")
    extractor = RepresentationExtractor(
        args.model,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=dtype,
    )
    extractor.load()

    print(f"[extract] {len(stim)} stimuli x per-layer last-token")
    acts = extractor.extract_all(stim)  # (N, L, D)
    print(f"[shape] activations = {acts.shape}")

    out_path = os.path.join(out_dir, "activations_cache.npz")
    np.savez_compressed(
        out_path,
        activations=acts.astype(np.float16),
        model_name=args.model,
    )
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
