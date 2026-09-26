#!/usr/bin/env python3
"""Sparse-autoencoder feature directions under the readout-orthogonality test.

SAEs are the current default estimator for "the direction of concept c", so the
test should say something about them. This script builds the SAE analogue of
FARS on the same stimuli and the same layer:

    encode the cached layer-L activations with a pretrained residual-stream SAE
    -> for each concept, score every feature by how selectively it fires on that
       concept (mean activation on c minus mean on the rest, in units of the
       pooled s.d.)
    -> take the concept's top-m features and sum their *decoder* directions,
       weighted by selectivity, giving one direction per concept
    -> PCA the 18 concept directions to rank k

This is deliberately the same recipe as FARS with the representation swapped:
concept-centroid PCA, but over SAE feature space rather than the raw residual
stream. No GPU is needed -- the SAE encoder is a linear map plus a ReLU, and
the activations are already cached.

Controls written alongside it:
    SAE_random      the same construction over randomly chosen features
    SAE_decoder_pca top-k PCA of the whole decoder matrix (what the SAE's
                    dictionary looks like irrespective of concepts)

Usage:
    python analysis/sae_concept_subspace.py --slug llama-3.1-8b-instruct \
        --sae-repo Goodfire/Llama-3.1-8B-Instruct-SAE-l19 \
        --sae-file Llama-3.1-8B-Instruct-SAE-l19.pth --layer 19 \
        --out analysis/sae_llama-3.1-8b-instruct.json
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "analysis"))
from benchmark.generate import generate_all_stimuli  # noqa: E402
from method_comparison_8way import grassmann, energy_in, _top_k_pca  # noqa: E402

K = 10


def selectivity(F: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(n_concepts, n_features) selectivity: (mean_in - mean_out) / pooled sd."""
    n = int(labels.max()) + 1
    out = np.zeros((n, F.shape[1]))
    for c in range(n):
        m = labels == c
        a, b = F[m], F[~m]
        sd = np.sqrt((a.var(0) * m.sum() + b.var(0) * (~m).sum()) / len(F)) + 1e-6
        out[c] = (a.mean(0) - b.mean(0)) / sd
    return out


def concept_dirs(sel: np.ndarray, W_dec: np.ndarray, m: int) -> np.ndarray:
    """One direction per concept: its top-m features' decoder vectors, selectivity-weighted."""
    dirs = np.zeros((sel.shape[0], W_dec.shape[0]))
    for c in range(sel.shape[0]):
        idx = np.argsort(-sel[c])[:m]
        w = np.clip(sel[c, idx], 0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        dirs[c] = (W_dec[:, idx] * w).sum(1) / w.sum()
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--sae-repo", required=True)
    ap.add_argument("--sae-file", required=True)
    ap.add_argument("--layer", type=int, required=True, help="cache index of the SAE's layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-m", type=int, default=16, help="features per concept")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()
    k = args.k

    import torch
    from huggingface_hub import hf_hub_download

    stim = generate_all_stimuli()
    cid = np.array([s.concept_id for s in stim])
    acts = np.load(os.path.join(ROOT, f"results/{args.slug}/activations_cache.npz"),
                   allow_pickle=True)["activations"]
    X = acts[:, args.layer, :].astype(np.float32)
    print(f"[{args.slug}] activations {acts.shape}, layer {args.layer}, D={X.shape[1]}")

    sd = torch.load(hf_hub_download(args.sae_repo, args.sae_file), map_location="cpu", weights_only=True)
    W_enc = sd["encoder_linear.weight"].numpy()          # (F, D)
    b_enc = sd["encoder_linear.bias"].numpy()            # (F,)
    W_dec = sd["decoder_linear.weight"].numpy()          # (D, F)
    b_dec = sd["decoder_linear.bias"].numpy()            # (D,)
    n_feat = W_enc.shape[0]
    print(f"  SAE {n_feat} features, D={W_enc.shape[1]}")

    # Goodfire SAEs encode as relu(W_enc x + b_enc); they do *not* subtract the
    # decoder bias first (the Anthropic convention), which would destroy sparsity.
    F = np.maximum(X @ W_enc.T + b_enc, 0.0)             # (N, F)
    rec = F @ W_dec.T + b_dec
    fvu = float(((X - rec) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
    alive = (F > 0).any(0)
    print(f"  active features: {alive.sum()} / {n_feat}  mean L0 = {(F > 0).sum(1).mean():.1f}  FVU = {fvu:.3f}")

    sel = selectivity(F, cid)
    B_sae = _top_k_pca(concept_dirs(sel, W_dec, args.top_m), k)

    rng = np.random.default_rng(0)
    sel_rand = sel.copy()
    rand_dirs = np.stack([W_dec[:, rng.choice(np.flatnonzero(alive), args.top_m, replace=False)].mean(1)
                          for _ in range(sel.shape[0])])
    B_rand_feat = _top_k_pca(rand_dirs, k)
    B_dec_pca = _top_k_pca(W_dec[:, alive].T, k)

    rz = np.load(os.path.join(ROOT, f"analysis/readout_top10_{args.slug}.npz"), allow_pickle=True)
    B_read = rz["basis" if "basis" in rz.files else "top_k_readout"].astype(np.float64)
    import glob
    fars_f = glob.glob(os.path.join(ROOT, f"results/{args.slug}/fars_basis_L*.npz"))[0]
    B_fars_native = np.load(fars_f, allow_pickle=True)["basis"].astype(np.float64)
    B_fars_here = _top_k_pca(np.stack([X[cid == c].mean(0) for c in range(int(cid.max()) + 1)]), k)

    res = {"model": args.slug, "sae": args.sae_repo, "layer": args.layer, "D": int(X.shape[1]),
           "n_features": int(n_feat), "n_alive": int(alive.sum()), "top_m": args.top_m, "K": k,
           "haar_null_energy_pct": 100.0 * k / X.shape[1],
           "fars_native_layer": int(np.load(fars_f, allow_pickle=True)["layer"]),
           "mean_L0": float((F > 0).sum(1).mean()), "fvu": fvu}
    for name, B in [("SAE_concept", B_sae), ("SAE_random_features", B_rand_feat),
                    ("SAE_decoder_pca", B_dec_pca), ("FARS_same_layer", B_fars_here),
                    ("FARS_native_layer", B_fars_native)]:
        dG, ang = grassmann(B.astype(np.float64), B_read)
        res[name] = {"d_G_vs_readout": dG, "energy_in_topk_readout_pct": 100 * energy_in(B.astype(np.float64), B_read),
                     "min_principal_angle_deg": float(ang.min())}
    res["sae_vs_fars_same_layer"] = {"energy_pct": 100 * energy_in(B_sae.astype(np.float64), B_fars_here.astype(np.float64)),
                                     "d_G": grassmann(B_sae.astype(np.float64), B_fars_here.astype(np.float64))[0]}
    e = lambda n: res[n]["energy_in_topk_readout_pct"]
    print(f"  SAE-concept {e('SAE_concept'):5.2f}%  SAE-random-feat {e('SAE_random_features'):5.2f}%  "
          f"SAE-decoder-PCA {e('SAE_decoder_pca'):5.2f}%  FARS(L{args.layer}) {e('FARS_same_layer'):5.2f}%  "
          f"Haar {res['haar_null_energy_pct']:.2f}%")
    print(f"  SAE-concept vs FARS overlap {res['sae_vs_fars_same_layer']['energy_pct']:.2f}%")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
