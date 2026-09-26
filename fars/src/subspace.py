"""
Concept-Conditioned Subspace Extraction for FARS.

Turns the Format-Agnostic Reasoning Subspace from a narrative into a
concrete, extractable mathematical object by:

1. Computing concept centroids (averaged across all forms and instances)
2. Running PCA on centroids to find "concept directions" — the principal
   axes of inter-concept variance with form-specific variance averaged out
3. Projecting all activations onto this subspace
4. Evaluating the subspace with RSA, silhouette, and cross-form probe metrics

A complementary **Form Subspace** (PCA on form centroids) serves as a
control: it should capture format-specific information and show the
opposite pattern (high form RSA, low concept RSA).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_rdm(X: np.ndarray) -> np.ndarray:
    """Build an N x N cosine dissimilarity matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    return cdist(X / norms, X / norms, metric="cosine")


def _upper_triangle(M: np.ndarray) -> np.ndarray:
    """Return strictly upper-triangular entries."""
    return M[np.triu_indices(M.shape[0], k=1)]


def _theoretical_rdm(labels: List[int]) -> np.ndarray:
    """Binary RDM: 0 if same label, 1 if different."""
    arr = np.array(labels)
    return (arr[:, None] != arr[None, :]).astype(float)


def _compute_centroids(
    X: np.ndarray,
    group_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean activation for each unique group label.

    Returns
    -------
    centroids : (n_groups, D)
    unique_labels : (n_groups,)
    """
    unique = np.unique(group_labels)
    centroids = np.stack([X[group_labels == g].mean(axis=0) for g in unique])
    return centroids, unique


# ---------------------------------------------------------------------------
# Subspace extraction
# ---------------------------------------------------------------------------

def extract_fars(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    n_components: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Extract the Format-Agnostic Reasoning Subspace at each layer.

    For each layer:
      1. Compute the mean activation per concept (averaged over forms and
         instances) -> 18 centroids of shape (D,)
      2. Run PCA on these centroids -> top-k concept directions
      3. Project all 324 activations onto the k-dimensional subspace

    Parameters
    ----------
    activations : (N, L, D) -- N stimuli, L layers, D hidden dim
    concept_labels : length-N list of concept ids
    form_labels : length-N list of form ids
    n_components : number of principal components to retain

    Returns
    -------
    dict:
        basis : (L, k, D) -- subspace basis vectors per layer
        projected : (L, N, k) -- projected activations per layer
        variance_explained : (L, k) -- fraction of centroid variance per PC
    """
    N, L, D = activations.shape
    c_arr = np.array(concept_labels)

    n_unique = len(np.unique(c_arr))
    k = min(n_components, n_unique - 1)  # PCA on n centroids -> at most n-1 PCs

    basis = np.zeros((L, k, D))
    projected = np.zeros((L, N, k))
    var_explained = np.zeros((L, k))

    for layer in range(L):
        X = activations[:, layer, :]

        # Step 1: concept centroids (averaged across forms + instances)
        centroids, _ = _compute_centroids(X, c_arr)

        # Step 2: PCA on centroids
        pca = PCA(n_components=k)
        pca.fit(centroids)

        basis[layer] = pca.components_           # (k, D)
        var_explained[layer] = pca.explained_variance_ratio_

        # Step 3: project all activations (center by centroid mean)
        X_centered = X - pca.mean_
        projected[layer] = X_centered @ pca.components_.T  # (N, k)

    return {
        "basis": basis,
        "projected": projected,
        "variance_explained": var_explained,
    }


def extract_form_subspace(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    n_components: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Extract the Form Subspace (control) at each layer.

    Same procedure as FARS but using form centroids instead of concept
    centroids.  This subspace should capture format-specific variance and
    serve as a negative control.
    """
    N, L, D = activations.shape
    f_arr = np.array(form_labels)

    n_unique = len(np.unique(f_arr))
    k = min(n_components, n_unique - 1)

    basis = np.zeros((L, k, D))
    projected = np.zeros((L, N, k))
    var_explained = np.zeros((L, k))

    for layer in range(L):
        X = activations[:, layer, :]

        centroids, _ = _compute_centroids(X, f_arr)

        pca = PCA(n_components=k)
        pca.fit(centroids)

        basis[layer] = pca.components_
        var_explained[layer] = pca.explained_variance_ratio_

        X_centered = X - pca.mean_
        projected[layer] = X_centered @ pca.components_.T

    return {
        "basis": basis,
        "projected": projected,
        "variance_explained": var_explained,
    }


# ---------------------------------------------------------------------------
# Subspace evaluation
# ---------------------------------------------------------------------------

def evaluate_subspace(
    projected: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    n_forms: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Evaluate a projected activation tensor on four metrics.

    Returns dict with per-layer arrays:
        rsa_concept, rsa_form, silhouette_concept, silhouette_form,
        probe_cross_form, probe_within_form
    """
    L, N, k = projected.shape
    c_arr = np.array(concept_labels)
    f_arr = np.array(form_labels)

    concept_rdm = _theoretical_rdm(concept_labels)
    form_rdm = _theoretical_rdm(form_labels)
    concept_vec = _upper_triangle(concept_rdm)
    form_vec = _upper_triangle(form_rdm)

    rsa_concept = np.zeros(L)
    rsa_form = np.zeros(L)
    sil_concept = np.zeros(L)
    sil_form = np.zeros(L)
    probe_cross = np.zeros(L)
    probe_within = np.zeros(L)

    for layer in range(L):
        X = projected[layer]  # (N, k)

        # (a) & (b) RSA
        emp_rdm = _cosine_rdm(X)
        emp_vec = _upper_triangle(emp_rdm)
        rsa_concept[layer], _ = spearmanr(emp_vec, concept_vec)
        rsa_form[layer], _ = spearmanr(emp_vec, form_vec)

        # (d) Silhouette
        try:
            sil_concept[layer] = silhouette_score(X, c_arr, metric="cosine")
        except ValueError:
            sil_concept[layer] = 0.0
        try:
            sil_form[layer] = silhouette_score(X, f_arr, metric="cosine")
        except ValueError:
            sil_form[layer] = 0.0

        # (c) Cross-form probe
        acc_matrix = np.zeros((n_forms, n_forms))
        for src in range(n_forms):
            src_mask = f_arr == src
            X_train = X[src_mask]
            y_train = c_arr[src_mask]

            if len(np.unique(y_train)) < 2:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            clf = RidgeClassifier(alpha=0.1)
            clf.fit(X_train_s, y_train)

            for tgt in range(n_forms):
                tgt_mask = f_arr == tgt
                X_test = X[tgt_mask]
                y_test = c_arr[tgt_mask]
                X_test_s = scaler.transform(X_test)
                preds = clf.predict(X_test_s)
                acc_matrix[src, tgt] = float(np.mean(preds == y_test))

        diag_mask = np.eye(n_forms, dtype=bool)
        off_mask = ~diag_mask
        probe_cross[layer] = np.mean(acc_matrix[off_mask])
        probe_within[layer] = np.mean(acc_matrix[diag_mask])

    return {
        "rsa_concept": rsa_concept,
        "rsa_form": rsa_form,
        "silhouette_concept": sil_concept,
        "silhouette_form": sil_form,
        "probe_cross_form": probe_cross,
        "probe_within_form": probe_within,
    }


# ---------------------------------------------------------------------------
# Full subspace analysis pipeline (for run_experiment.py)
# ---------------------------------------------------------------------------

def subspace_analysis(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    n_forms: int = 6,
    fars_components: int = 10,
    form_components: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run the full subspace extraction and evaluation pipeline.

    Returns
    -------
    dict:
        fars : {basis, projected, variance_explained, eval}
        form : {basis, projected, variance_explained, eval}
    """
    fars = extract_fars(activations, concept_labels, form_labels, fars_components)
    form = extract_form_subspace(activations, concept_labels, form_labels, form_components)

    fars_eval = evaluate_subspace(fars["projected"], concept_labels, form_labels, n_forms)
    form_eval = evaluate_subspace(form["projected"], concept_labels, form_labels, n_forms)

    fars["eval"] = fars_eval
    form["eval"] = form_eval

    return {"fars": fars, "form": form}


# ---------------------------------------------------------------------------
# Enhancement baselines and analyses
# ---------------------------------------------------------------------------

def extract_random_subspace(
    activations: np.ndarray,
    n_components: int = 10,
    n_draws: int = 10,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate random orthonormal subspaces as a control baseline.

    Returns a list of n_draws dicts, each with {basis, projected}.
    """
    N, L, D = activations.shape
    rng = np.random.RandomState(seed)
    results = []

    for _ in range(n_draws):
        basis = np.zeros((L, n_components, D))
        projected = np.zeros((L, N, n_components))

        for layer in range(L):
            raw = rng.randn(n_components, D)
            q, _ = np.linalg.qr(raw.T)
            basis[layer] = q[:, :n_components].T  # (k, D)

            X = activations[:, layer, :]
            X_centered = X - X.mean(axis=0)
            projected[layer] = X_centered @ basis[layer].T

        results.append({"basis": basis, "projected": projected})

    return results


def extract_fullpca_subspace(
    activations: np.ndarray,
    n_components: int = 10,
) -> Dict[str, np.ndarray]:
    """
    PCA on all activations (not concept centroids) as a denoising control.

    This captures the top variance directions of the full data, mixing
    concept and form variance. FARS should outperform this because it
    specifically isolates concept variance.
    """
    N, L, D = activations.shape
    k = n_components
    basis = np.zeros((L, k, D))
    projected = np.zeros((L, N, k))
    var_explained = np.zeros((L, k))

    for layer in range(L):
        X = activations[:, layer, :]
        pca = PCA(n_components=k)
        projected[layer] = pca.fit_transform(X)
        basis[layer] = pca.components_
        var_explained[layer] = pca.explained_variance_ratio_

    return {"basis": basis, "projected": projected, "variance_explained": var_explained}


def extract_fars_leave_k_out(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    holdout_concepts: List[int],
    n_components: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Extract FARS using only non-held-out concepts, then evaluate on held-out.

    Returns dict with:
        basis, train_projected, holdout_projected,
        holdout_concept_labels, holdout_form_labels
    """
    N, L, D = activations.shape
    c_arr = np.array(concept_labels)
    f_arr = np.array(form_labels)
    holdout_set = set(holdout_concepts)

    train_mask = np.array([c not in holdout_set for c in concept_labels])
    test_mask = ~train_mask

    n_train_concepts = len(set(c_arr[train_mask]))
    k = min(n_components, n_train_concepts - 1)

    basis = np.zeros((L, k, D))
    train_projected = np.zeros((L, int(train_mask.sum()), k))
    test_projected = np.zeros((L, int(test_mask.sum()), k))

    for layer in range(L):
        X_train = activations[train_mask, layer, :]
        X_test = activations[test_mask, layer, :]

        # Concept centroids from training concepts only
        centroids, _ = _compute_centroids(X_train, c_arr[train_mask])

        pca = PCA(n_components=k)
        pca.fit(centroids)
        basis[layer] = pca.components_

        train_projected[layer] = (X_train - pca.mean_) @ pca.components_.T
        test_projected[layer] = (X_test - pca.mean_) @ pca.components_.T

    return {
        "basis": basis,
        "train_projected": train_projected,
        "holdout_projected": test_projected,
        "train_concept_labels": c_arr[train_mask].tolist(),
        "train_form_labels": f_arr[train_mask].tolist(),
        "holdout_concept_labels": c_arr[test_mask].tolist(),
        "holdout_form_labels": f_arr[test_mask].tolist(),
    }


def cross_model_fars_alignment(
    projections_A: np.ndarray,
    projections_B: np.ndarray,
    concept_labels: List[int],
) -> Dict[str, float]:
    """
    Measure alignment between two models' FARS projections.

    Uses two complementary metrics:
    1. CCA: Canonical Correlation Analysis between projected spaces
    2. RSA: Spearman correlation between concept RDMs in projected spaces

    Parameters
    ----------
    projections_A : (N, k) projected activations from model A at best layer
    projections_B : (N, k) projected activations from model B at best layer
    concept_labels : length-N concept ids

    Returns dict with cca_mean, cca_correlations, rsa_alignment
    """
    from sklearn.cross_decomposition import CCA as SkCCA

    k = min(projections_A.shape[1], projections_B.shape[1], 10)

    # CCA
    cca = SkCCA(n_components=k)
    X_c, Y_c = cca.fit_transform(projections_A, projections_B)
    # Canonical correlations
    cca_corrs = [float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(k)]

    # RSA alignment: compare concept RDMs
    rdm_A = _cosine_rdm(projections_A)
    rdm_B = _cosine_rdm(projections_B)
    vec_A = _upper_triangle(rdm_A)
    vec_B = _upper_triangle(rdm_B)
    rsa_align, _ = spearmanr(vec_A, vec_B)

    # Also compare concept-centroid RDMs (more stable)
    c_arr = np.array(concept_labels)
    centroids_A, _ = _compute_centroids(projections_A, c_arr)
    centroids_B, _ = _compute_centroids(projections_B, c_arr)
    crdm_A = _cosine_rdm(centroids_A)
    crdm_B = _cosine_rdm(centroids_B)
    centroid_rsa, _ = spearmanr(_upper_triangle(crdm_A), _upper_triangle(crdm_B))

    return {
        "cca_correlations": cca_corrs,
        "cca_mean": float(np.mean(cca_corrs)),
        "rsa_alignment": float(rsa_align),
        "centroid_rsa_alignment": float(centroid_rsa),
    }
