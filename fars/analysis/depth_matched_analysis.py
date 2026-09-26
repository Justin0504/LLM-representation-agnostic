#!/usr/bin/env python3
"""Readout-orthogonality at matched depth.

For each model with a depth_matched/<slug>.npz (from depth_matched_readout.py)
and the cached TriForm activations, re-evaluate the nine rank-10 subspaces of
method_comparison_8way.py plus the depth-matched positive control
(next-token-identity centroid PCA at the FARS layer) against two readout bases:

    wu        top-10 right singular vectors of W_U            (the body's test)
    readout_L top-10 right singular vectors of W_U Γ A_L      (pulled back to layer L)

Writes analysis/depth_matched_<slug>.json. Run from the repo root on the
machine that holds results/ (CPU only).
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "analysis"))
from method_comparison_8way import (grassmann, energy_in, _top_k_pca, centroid_pca,  # noqa: E402
                                    lda_subspace, diff_means_subspace,
                                    probe_weight_subspace, random_subspace)
from benchmark.generate import generate_all_stimuli  # noqa: E402

K = 10


def main() -> None:
    stim = generate_all_stimuli()
    concept_ids = np.array([s.concept_id for s in stim])
    form_ids = np.array([s.form_id for s in stim])
    files = sorted(glob.glob(os.path.join(ROOT, "analysis/depth_matched/*.npz")))
    if not files:
        print("no depth_matched/*.npz"); return
    for f in files:
        slug = os.path.basename(f)[:-4]
        acts_f = os.path.join(ROOT, f"results/{slug}/activations_cache.npz")
        fars_f = glob.glob(os.path.join(ROOT, f"results/{slug}/fars_basis_L*.npz"))
        if not (os.path.exists(acts_f) and fars_f):
            print(f"[skip] {slug}: missing activations or FARS basis"); continue
        z = np.load(f)
        fz = np.load(fars_f[0], allow_pickle=True)
        layer = int(fz["layer"])
        acts = np.load(acts_f, allow_pickle=True)["activations"]
        # Some caches store hidden_states[0] (embeddings) at index 0; the block
        # index of the FARS layer is then one lower than its cache index.
        offset = acts.shape[1] - int(z["n_layers"])
        if layer - offset != int(z["layer"]):
            print(f"[warn] {slug}: FARS cache idx {layer} (offset {offset}) -> block {layer-offset}, "
                  f"but depth-matched npz was computed at block {int(z['layer'])}")
        X = acts[:, layer, :].astype(np.float64)
        X_last = acts[:, -1, :].astype(np.float64)
        D = X.shape[1]
        shuf = concept_ids[np.random.default_rng(2027).permutation(len(concept_ids))]
        subs = {
            "FARS": fz["basis"].astype(np.float64),
            "LDA": lda_subspace(X, concept_ids, K),
            "DiffMeans": diff_means_subspace(X, concept_ids, K),
            "ProbeW": probe_weight_subspace(X, concept_ids, K),
            "FullPCA": _top_k_pca(X, K),
            "FormPCA": centroid_pca(X, form_ids, K),
            "ShuffFARS": centroid_pca(X, shuf, K),
            "Random": random_subspace(D, K, seed=42),
            "NextTokPCA_L": z["nexttok_pca_L"].astype(np.float64),      # depth-matched positive
            "LastPCA": _top_k_pca(X_last, K),
            "NextTokPCA_last": z["nexttok_pca_last"].astype(np.float64),
        }
        bases = {"wu": z["wu_top100"][:K].astype(np.float64),
                 "readout_L": z["readout_L_top100"][:K].astype(np.float64)}
        res = {"model": slug, "layer": layer, "block": layer - offset, "cache_offset": offset, "n_layers": int(z["n_layers"]), "D": D, "K": K,
               "ceiling": float(np.sqrt(K) * np.pi / 2), "ridge_r2": float(z["ridge_r2"]),
               "ridge_cos": float(z["ridge_cos"]), "n_tokens": int(z["n_tokens"]),
               "n_classes": int(z["n_classes"]), "haar_null_energy_pct": 100.0 * K / D}
        for bname, B_r in bases.items():
            res[bname] = {}
            for name, B in subs.items():
                dG, ang = grassmann(B, B_r)
                res[bname][name] = {"d_G": dG, "energy_pct": 100 * energy_in(B, B_r),
                                    "min_angle_deg": float(ang.min())}
        e = lambda b, n: res[b][n]["energy_pct"]
        print(f"[{slug}] L={layer} R2={res['ridge_r2']:.2f} | vs W_U: FARS {e('wu','FARS'):.2f}% "
              f"NextTok_L {e('wu','NextTokPCA_L'):.2f}% Last {e('wu','LastPCA'):.2f}% | "
              f"vs readout_L: FARS {e('readout_L','FARS'):.2f}% NextTok_L {e('readout_L','NextTokPCA_L'):.2f}% "
              f"FullPCA {e('readout_L','FullPCA'):.2f}% Random {e('readout_L','Random'):.2f}%")
        json.dump(res, open(os.path.join(ROOT, f"analysis/depth_matched_{slug}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
