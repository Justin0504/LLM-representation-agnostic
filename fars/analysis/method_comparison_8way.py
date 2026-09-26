"""Eight-way readout-orthogonality comparison across standard concept-subspace estimators.

The readout-orthogonality test (Grassmann distance to the top-k right singular
vectors of W_U) is only interesting if it applies to the estimators the
interpretability literature actually uses, and if it can return BOTH answers.
This script therefore evaluates:

  Published concept-subspace estimators
  -------------------------------------
    FARS        concept-centroid PCA (this paper)
    LDA         multi-class linear discriminant analysis (supervised gold standard)
    DiffMeans   stacked pairwise concept mean-differences, PCA'd
                (the "steering vector" family: Turner+2023, Arditi+2024)
    ProbeW      one-vs-rest ridge probe weight vectors, PCA'd
                (the most common probing-paper construction)

  Negative / structure controls
  -----------------------------
    FullPCA     top-k PCA of raw activations at the same layer (variance-max)
    FormPCA     form-centroid PCA (form-dominant matched counterpart)
    ShuffFARS   concept-centroid PCA on shuffled concept labels
    Random      Haar rank-k subspace of the same ambient space

  POSITIVE control (should be readout-ADJACENT)
  ---------------------------------------------
    LastPCA     top-k PCA of the FINAL layer's activations. The final residual
                stream is the immediate input to W_U, so its dominant directions
                are readout-adjacent by construction. This is the control that
                shows the test can return the other answer -- without it, a
                near-orthogonal reading for FARS is uninterpretable.

Usage:
    python3 analysis/method_comparison_8way.py --slug qwen2.5-7b \
        --acts results/qwen2.5-7b/activations_cache.npz \
        --fars-basis results/qwen2.5-7b/fars_basis_L13.npz \
        --top10-readout analysis/readout_top10_qwen2.5-7b.npz \
        --out analysis/method_8way_qwen2.5-7b.json
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np

sys.path.insert(0, ".")

K = 10
RIDGE_LAMBDA = 1e-2   # relative to trace(Sigma)/D; D >> n so regularisation is required


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def orthonormalize(B: np.ndarray) -> np.ndarray:
    """Rows of B span the subspace; return an orthonormal row basis."""
    Q, _ = np.linalg.qr(B.T)
    return Q.T


def grassmann(B1: np.ndarray, B2: np.ndarray) -> tuple[float, np.ndarray]:
    """Geodesic distance and principal angles (degrees) between two row-bases."""
    B1, B2 = orthonormalize(B1), orthonormalize(B2)
    s = np.linalg.svd(B1 @ B2.T, compute_uv=False)
    theta = np.arccos(np.clip(s, -1.0, 1.0))
    return float(np.sqrt((theta ** 2).sum())), np.degrees(theta)


def energy_in(B_test: np.ndarray, B_span: np.ndarray) -> float:
    """Mean fraction of each test direction's energy inside span(B_span)."""
    B_test = orthonormalize(B_test)
    B_span = orthonormalize(B_span)
    coeffs = B_test @ B_span.T
    return float((coeffs ** 2).sum(axis=1).mean())


# --------------------------------------------------------------------------- #
# Subspace estimators
# --------------------------------------------------------------------------- #
def _top_k_pca(M: np.ndarray, k: int) -> np.ndarray:
    M = M - M.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:k]


def centroid_pca(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Centroid PCA: average within label, then PCA the centroids.

    With n_labels <= k the centroid matrix is rank-deficient (n_labels - 1 after
    centering), so the basis is padded with the leading directions of the
    residual activations. Rank matching is required for the Grassmann distance
    to be comparable across estimators.
    """
    n = int(labels.max()) + 1
    C = np.stack([X[labels == c].mean(0) for c in range(n)])
    B = _top_k_pca(C, k)
    if B.shape[0] >= k:
        return B[:k]
    Xc = X - X.mean(0, keepdims=True)
    resid = Xc - (Xc @ B.T) @ B
    _, _, Vt = np.linalg.svd(resid, full_matrices=False)
    return np.vstack([B, Vt[: k - B.shape[0]]])


def lda_subspace(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Multi-class LDA in the whitened within-class space.

    D >> n so the within-class scatter is rank-deficient; we first project onto
    the top-n_samples PCA directions, solve LDA there, then lift back.
    """
    n_cls = int(labels.max()) + 1
    # Project into a well-conditioned subspace first.
    r = min(X.shape[0] - n_cls - 1, X.shape[1], 200)
    P = _top_k_pca(X, r)                        # (r, D)
    Z = (X - X.mean(0, keepdims=True)) @ P.T    # (N, r)

    mu = Z.mean(0)
    Sw = np.zeros((r, r))
    Sb = np.zeros((r, r))
    for c in range(n_cls):
        Zc = Z[labels == c]
        mc = Zc.mean(0)
        Dc = Zc - mc
        Sw += Dc.T @ Dc
        d = (mc - mu)[:, None]
        Sb += len(Zc) * (d @ d.T)
    Sw += RIDGE_LAMBDA * np.trace(Sw) / r * np.eye(r)

    evals, evecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
    order = np.argsort(-evals.real)[:k]
    W = evecs[:, order].real.T                  # (k, r)
    return orthonormalize(W @ P)                # lift to (k, D)


def diff_means_subspace(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Stacked pairwise class mean-differences, PCA'd.

    This is the subspace spanned by the 'steering vector' family: each
    mu_A - mu_B is exactly the difference-of-means direction used by
    Turner et al. (2023) and Arditi et al. (2024) for a single concept pair.
    """
    n = int(labels.max()) + 1
    mus = np.stack([X[labels == c].mean(0) for c in range(n)])
    diffs = np.stack([mus[a] - mus[b] for a in range(n) for b in range(a + 1, n)])
    # Difference vectors are already mean-free in the relevant sense; do not re-centre.
    _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
    return Vt[:k]


def probe_weight_subspace(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """One-vs-rest ridge probe weights, stacked and PCA'd.

    Closed-form ridge in the dual (N < D): w = X^T (X X^T + lam I)^-1 y.
    This mirrors the standard 'train a linear probe per concept, read the
    weights as the concept direction' construction.
    """
    n = int(labels.max()) + 1
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T                                   # (N, N)
    lam = RIDGE_LAMBDA * np.trace(G) / G.shape[0]
    Ginv = np.linalg.inv(G + lam * np.eye(G.shape[0]))
    Ws = []
    for c in range(n):
        y = np.where(labels == c, 1.0, -1.0)
        y = y - y.mean()
        Ws.append(Xc.T @ (Ginv @ y))
    W = np.stack(Ws)                                 # (n_cls, D)
    return _top_k_pca(W, k)


def random_subspace(D: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return Q.T


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--fars-basis", required=True)
    ap.add_argument("--top10-readout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()
    k = args.k

    from benchmark.generate import generate_all_stimuli
    stim = generate_all_stimuli()
    concept_ids = np.array([s.concept_id for s in stim])
    form_ids = np.array([s.form_id for s in stim])

    acts_all = np.load(args.acts, allow_pickle=True)["activations"]
    fars_z = np.load(args.fars_basis, allow_pickle=True)
    layer = int(fars_z["layer"])
    B_fars = fars_z["basis"].astype(np.float64)

    readout = np.load(args.top10_readout, allow_pickle=True)
    key = "basis" if "basis" in readout.files else "top_k_readout"
    B_readout = readout[key].astype(np.float64)

    assert acts_all.shape[0] == len(stim), (acts_all.shape, len(stim))
    X = acts_all[:, layer, :].astype(np.float64)       # FARS layer
    X_last = acts_all[:, -1, :].astype(np.float64)     # final layer (positive control)
    D = X.shape[1]
    n_layers = acts_all.shape[1]

    rng_shuf = np.random.default_rng(2027)
    shuffled_labels = concept_ids[rng_shuf.permutation(len(concept_ids))]

    subspaces = {
        # Published concept-subspace estimators
        "FARS":      B_fars,
        "LDA":       lda_subspace(X, concept_ids, k),
        "DiffMeans": diff_means_subspace(X, concept_ids, k),
        "ProbeW":    probe_weight_subspace(X, concept_ids, k),
        # Structure / negative controls
        "FullPCA":   _top_k_pca(X, k),
        "FormPCA":   centroid_pca(X, form_ids, k),
        "ShuffFARS": centroid_pca(X, shuffled_labels, k),
        "Random":    random_subspace(D, k, seed=42),
        # POSITIVE control: final-layer PCA feeds W_U directly
        "LastPCA":   _top_k_pca(X_last, k),
    }

    ceiling = float(np.sqrt(k) * np.pi / 2)
    result = {
        "model": args.slug, "layer": layer, "n_layers": n_layers,
        "K": k, "D": D, "ceiling": ceiling,
    }
    print(f"[{args.slug}] L={layer}/{n_layers-1}  D={D}  ceiling={ceiling:.3f}")
    for name, B in subspaces.items():
        dG, angles = grassmann(B, B_readout)
        e = energy_in(B, B_readout)
        result[name] = {
            "d_G_vs_readout": dG,
            "energy_in_topk_readout_pct": e * 100,
            "min_principal_angle_deg": float(angles.min()),
            "frac_of_ceiling": dG / ceiling,
        }
        print(f"  {name:10s} d_G={dG:.3f} ({dG/ceiling*100:5.1f}% of ceiling)"
              f"  energy={e*100:5.2f}%  min_theta={angles.min():5.1f} deg")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
