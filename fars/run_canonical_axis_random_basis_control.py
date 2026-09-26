"""run_canonical_axis_random_basis_control.py — random-basis control for canonical-axis GPA.

Replaces each model's FARS basis with a random 10-d orthonormal projection of the
same activations at the same layer, then re-runs GPA. Tests whether the
canonical-axis universality (~95% var explained) is specific to FARS or holds
for any low-rank projection of the activations.

Result (5 models, 10 seeds): random-projection GPA gives 71.4% +/- 1.9%,
vs FARS at 95.1% (12.5 sigma above the random distribution). The canonical-axis
universality is therefore specific to FARS, not a property of arbitrary
low-rank slices.
"""
from __future__ import annotations
import os, sys, glob, json
import numpy as np
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark.generate import generate_all_stimuli
from src.subspace import extract_fars, evaluate_subspace


def get_centroids_random(cache_path, c_labels, f_labels, unique,
                          n_components=10, seed=42):
    acts = np.load(cache_path, allow_pickle=True)["activations"]
    fars = extract_fars(acts, c_labels.tolist(), f_labels.tolist(),
                          n_components=n_components)
    ev = evaluate_subspace(fars["projected"], c_labels.tolist(),
                            f_labels.tolist())
    best = int(np.argmax(ev["rsa_concept"]))
    D = acts.shape[2]
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((D, n_components))
    Q, _ = np.linalg.qr(G)
    basis = Q.T  # (10, D)
    X = acts[:, best, :] @ basis.T
    centroids = np.stack([X[c_labels == c].mean(0) for c in unique])
    centroids = centroids - centroids.mean(0, keepdims=True)
    centroids = centroids / (np.linalg.norm(centroids, "fro") + 1e-10)
    return centroids


def gpa(C, max_iter=30, tol=1e-8):
    canonical = C[0].copy()
    for _ in range(max_iter):
        new = []
        for Ci in C:
            R, _ = orthogonal_procrustes(Ci, canonical)
            new.append(Ci @ R)
        new_canon = np.mean(np.stack(new), axis=0)
        new_canon = new_canon / np.linalg.norm(new_canon, "fro")
        if np.linalg.norm(new_canon - canonical, "fro") < tol: break
        canonical = new_canon
    aligned = [Ci @ orthogonal_procrustes(Ci, canonical)[0] for Ci in C]
    rss = np.mean([np.sum((Ci - canonical)**2) for Ci in aligned])
    tss = np.mean([np.sum(Ci**2) for Ci in aligned])
    return 1 - rss/tss


def main():
    stim = generate_all_stimuli()
    c_labels = np.array([s.concept_id for s in stim])
    f_labels = np.array([s.form_id for s in stim])
    unique = sorted(set(c_labels.tolist()))

    seeds = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    results = []
    slugs = []
    print(f"Random-basis GPA on {len(list(glob.glob('results/*/activations_cache.npz')))} models, {len(seeds)} seeds:")
    for seed in seeds:
        cmats = []
        for cache in sorted(glob.glob("results/*/activations_cache.npz")):
            if seed == seeds[0]:
                slugs.append(cache.split("/")[1])
            cmats.append(get_centroids_random(
                cache, c_labels, f_labels, unique, seed=seed))
        ve = gpa(cmats)
        results.append(float(ve))
        print(f"  seed {seed}: variance-explained = {ve:.4f}")
    mean = float(np.mean(results))
    std = float(np.std(results))
    print(f"\nMean = {mean:.4f}, std = {std:.4f}")
    print(f"Range: [{min(results):.4f}, {max(results):.4f}]")

    out = {
        "description": "Random-basis control for canonical-axis GPA alignment.",
        "models": slugs,
        "n_components": 10,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "random_basis_variance_explained": results,
        "random_basis_mean": mean,
        "random_basis_std": std,
        "random_basis_range": [float(min(results)), float(max(results))],
        "fars_canonical_variance_explained": 0.9510,
        "shuffled_concept_null_mean": 0.6300,
        "shuffled_concept_null_std": 0.0336,
        "z_score_fars_vs_random_basis": (0.9510 - mean) / std,
    }
    with open("results/canonical_axis_random_basis_control.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/canonical_axis_random_basis_control.json")


if __name__ == "__main__":
    main()
