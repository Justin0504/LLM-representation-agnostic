"""Historical X-FARS alignment heuristic (numerics preserved).

The polar-factor update is not an exact minimizer of the rectangular
least-squares projection objective. No global-optimality guarantee is claimed.
See KNOWN_LIMITATIONS.md before interpreting historical outputs.
"""
from __future__ import annotations
import json, os, sys, glob
import numpy as np
from numpy.linalg import svd, qr
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark.generate import generate_all_stimuli
from src.subspace import extract_fars, evaluate_subspace


def concept_rsa(P: np.ndarray, c_labels) -> float:
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-10)
    emp = cdist(Pn, Pn, metric="cosine")
    ix = np.triu_indices(emp.shape[0], k=1)
    a = np.asarray(c_labels)
    theo = (a[:, None] != a[None, :]).astype(float)
    return float(spearmanr(emp[ix], theo[ix])[0])


def solve_basis(C_m: np.ndarray, C_bar: np.ndarray) -> np.ndarray:
    """Return the polar-factor trace-maximizing update, not an exact
    rectangular least-squares projection minimizer. See KNOWN_LIMITATIONS.md.
    """
    M = C_m.T @ C_bar  # (D_m, k)
    U, S, Wt = svd(M, full_matrices=False)
    # U is (D_m, k), Wt is (k, k)
    V = U @ Wt  # (D_m, k) with orthonormal columns
    return V.T  # (k, D_m)


def fit_xfars(C_list, k=10, max_iter=50, tol=1e-7, verbose=True):
    """C_list: list of M centroid matrices, each (K, D_m). Returns
    bases [B_m] each (k, D_m) and shared canonical frame (K, k)."""
    M = len(C_list)
    K = C_list[0].shape[0]
    # 1. Init: per-model FARS (top-k PCA on C_m), then Procrustes-align.
    bases_init = []
    proj_init = []
    for C in C_list:
        # PCA on centroids
        Cc = C - C.mean(0, keepdims=True)
        Uc, Sc, Vct = svd(Cc, full_matrices=False)
        B_m = Vct[:k]  # (k, D_m), orthonormal rows
        bases_init.append(B_m)
        proj_init.append(C @ B_m.T)  # (K, k)

    # Iterative Procrustes mean for init of C_bar
    C_bar = proj_init[0].copy()
    C_bar = C_bar - C_bar.mean(0, keepdims=True)
    C_bar = C_bar / np.linalg.norm(C_bar, "fro")
    for _ in range(30):
        aligned = []
        for P in proj_init:
            Pn = P - P.mean(0, keepdims=True)
            Pn = Pn / (np.linalg.norm(Pn, "fro") + 1e-10)
            R, _ = orthogonal_procrustes(Pn, C_bar)
            aligned.append(Pn @ R)
        new_bar = np.mean(np.stack(aligned), axis=0)
        new_bar = new_bar / np.linalg.norm(new_bar, "fro")
        if np.linalg.norm(new_bar - C_bar, "fro") < 1e-9: break
        C_bar = new_bar

    bases = [B.copy() for B in bases_init]

    # 3. Joint optimisation
    prev_obj = np.inf
    for it in range(max_iter):
        # Step a: re-solve each B_m given C_bar
        new_bases = []
        for m, C in enumerate(C_list):
            Cc = C - C.mean(0, keepdims=True)
            new_bases.append(solve_basis(Cc, C_bar))
        bases = new_bases

        # Step b: update C_bar (centre+normalise the average projection)
        proj = [
            (C_list[m] - C_list[m].mean(0, keepdims=True)) @ bases[m].T
            for m in range(M)
        ]
        # Normalise each projection to unit Frobenius first so they're comparable
        proj_norm = [P / (np.linalg.norm(P, "fro") + 1e-10) for P in proj]
        new_bar = np.mean(np.stack(proj_norm), axis=0)
        new_bar = new_bar / np.linalg.norm(new_bar, "fro")

        # Track objective
        obj = sum(np.linalg.norm(P - new_bar, "fro")**2 for P in proj_norm)
        if verbose and (it % 5 == 0 or it == max_iter - 1):
            print(f"  iter {it}: obj = {obj:.6f}")
        if abs(prev_obj - obj) < tol: break
        prev_obj = obj
        C_bar = new_bar

    return bases, C_bar


def collect_centroids(slug, c_labels, f_labels):
    cache = f"results/{slug}/activations_cache.npz"
    if not os.path.exists(cache): return None
    acts = np.load(cache, allow_pickle=True)["activations"]
    fars = extract_fars(acts, c_labels.tolist(), f_labels.tolist(),
                          n_components=10)
    ev = evaluate_subspace(fars["projected"], c_labels.tolist(),
                            f_labels.tolist())
    best = int(np.argmax(ev["rsa_concept"]))
    X = acts[:, best, :].astype(np.float32)  # (N, D)
    unique = sorted(set(c_labels.tolist()))
    C = np.stack([X[c_labels == c].mean(0) for c in unique])  # (K, D)
    return {"slug": slug, "best_layer": best, "C": C,
            "X_full": X, "fars_basis": fars["basis"][best]}


def evaluate_xfars_per_model(slug, basis, c_labels, f_labels):
    """Compute concept-RSA on the X-FARS projected stimuli for one model."""
    cache = f"results/{slug}/activations_cache.npz"
    acts = np.load(cache, allow_pickle=True)["activations"]
    # Use the same best layer as standard FARS for direct comparison
    fars = extract_fars(acts, c_labels.tolist(), f_labels.tolist(),
                          n_components=10)
    ev = evaluate_subspace(fars["projected"], c_labels.tolist(),
                            f_labels.tolist())
    best = int(np.argmax(ev["rsa_concept"]))
    X = acts[:, best, :]
    P_xfars = X @ basis.T
    return float(concept_rsa(P_xfars, c_labels))


def main():
    stim = generate_all_stimuli()
    c_labels = np.array([s.concept_id for s in stim])
    f_labels = np.array([s.form_id for s in stim])

    slugs = []
    cdata = []
    for cache in sorted(glob.glob("results/*/activations_cache.npz")):
        slug = cache.split("/")[1]
        info = collect_centroids(slug, c_labels, f_labels)
        if info is None: continue
        slugs.append(slug)
        cdata.append(info)
        print(f"  {slug}: layer={info['best_layer']}, D={info['C'].shape[1]}")

    print(f"\nFitting X-FARS on {len(slugs)} models...")
    bases_xfars, C_bar = fit_xfars([d["C"] for d in cdata], k=10, max_iter=100,
                                      verbose=True)

    # Evaluate per-model concept RSA
    print(f"\n=== Per-model concept RSA (X-FARS vs standard FARS) ===")
    rsa_xfars = {}
    rsa_fars = {}
    for slug, B_x, d in zip(slugs, bases_xfars, cdata):
        rx = evaluate_xfars_per_model(slug, B_x, c_labels, f_labels)
        # Standard FARS RSA at the same layer
        P_f = d["X_full"] @ d["fars_basis"].T
        rf = concept_rsa(P_f, c_labels)
        rsa_xfars[slug] = rx
        rsa_fars[slug] = rf
        print(f"  {slug}: FARS RSA = {rf:.4f}, X-FARS RSA = {rx:.4f}, "
              f"ΔRSA = {rx-rf:+.4f}")

    # Compute variance explained by joint canonical frame
    proj_xfars = [
        (d["C"] - d["C"].mean(0, keepdims=True)) @ B.T
        for d, B in zip(cdata, bases_xfars)
    ]
    proj_norm = [P / (np.linalg.norm(P, "fro") + 1e-10) for P in proj_xfars]
    rss = np.mean([np.sum((P - C_bar)**2) for P in proj_norm])
    tss = np.mean([np.sum(P**2) for P in proj_norm])
    var_explained_xfars = 1 - rss/tss
    print(f"\n=== Canonical alignment ===")
    print(f"  X-FARS variance-explained (no Procrustes needed): "
          f"{var_explained_xfars:.4f}")

    # Standard FARS for comparison: extract per-model FARS, then GPA
    proj_fars = []
    for d in cdata:
        Cc = d["C"] - d["C"].mean(0, keepdims=True)
        Pf = Cc @ d["fars_basis"].T  # (K, k)
        Pf = Pf / (np.linalg.norm(Pf, "fro") + 1e-10)
        proj_fars.append(Pf)
    # Run GPA
    canonical = proj_fars[0].copy()
    for _ in range(30):
        aligned = []
        for P in proj_fars:
            R, _ = orthogonal_procrustes(P, canonical)
            aligned.append(P @ R)
        new_canon = np.mean(np.stack(aligned), axis=0)
        new_canon = new_canon / np.linalg.norm(new_canon, "fro")
        if np.linalg.norm(new_canon - canonical, "fro") < 1e-8: break
        canonical = new_canon
    aligned_fars = [P @ orthogonal_procrustes(P, canonical)[0] for P in proj_fars]
    rss_f = np.mean([np.sum((P - canonical)**2) for P in aligned_fars])
    tss_f = np.mean([np.sum(P**2) for P in aligned_fars])
    var_explained_fars = 1 - rss_f/tss_f
    print(f"  Standard FARS + post-hoc GPA variance-explained: "
          f"{var_explained_fars:.4f}")

    # Save bases for downstream use
    os.makedirs("results_v2", exist_ok=True)
    np.savez_compressed("results/xfars_bases.npz",
                          C_bar=C_bar,
                          slugs=np.array(slugs),
                          **{f"basis_{slug}": B for slug, B in zip(slugs, bases_xfars)})

    out = {
        "slugs": slugs,
        "rsa_per_model": {"fars": rsa_fars, "xfars": rsa_xfars},
        "variance_explained": {
            "fars_post_hoc_gpa": float(var_explained_fars),
            "xfars_jointly_optimised": float(var_explained_xfars),
        },
    }
    with open("results/xfars.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/xfars.json")
    print(f"      results/xfars_bases.npz")


if __name__ == "__main__":
    main()
