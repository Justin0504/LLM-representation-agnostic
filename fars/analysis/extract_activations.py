"""Extract last-input-token activations on the TriForm benchmark (324 stimuli).

Companion to extract_novelty_activations.py. Writes
results/<slug>/activations_cache.npz with shape (324, n_layers, D).

Pass --device auto for checkpoints that do not fit on one GPU.
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, ".")

from benchmark.generate import generate_all_stimuli  # noqa: E402
from src.extractor import RepresentationExtractor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-root", default="results_v2")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto",
                    help="'auto' shards across all visible GPUs")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]

    stim = generate_all_stimuli()
    print(f"[data] {len(stim)} TriForm stimuli")

    out_dir = os.path.join(args.out_root, args.slug)
    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    if device != "auto" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[load] {args.model}  dtype={args.dtype} device={device}")
    extractor = RepresentationExtractor(args.model, device=device, dtype=dtype)
    extractor.load()

    acts = extractor.extract_all(stim)
    print(f"[shape] {acts.shape}")

    out_path = os.path.join(out_dir, "activations_cache.npz")
    np.savez_compressed(
        out_path,
        activations=acts.astype(np.float16),
        model_name=args.model,
        n_stimuli=len(stim),
        n_layers=acts.shape[1],
        hidden_size=acts.shape[2],
    )
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
