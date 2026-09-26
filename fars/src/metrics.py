"""
Analysis metrics for the representation-agnostic pilot.

Three complementary measurements are implemented:

M1  RSA  (Representational Similarity Analysis)
    ─ Build an empirical RDM (pairwise cosine distance matrix) for each layer.
    ─ Build two theoretical RDMs: conceptual structure and form structure.
    ─ Measure Spearman correlation of the empirical RDM with each theory.
    ─ If deep layers yield higher corr(conceptual) than corr(form), the
      model abstracts over symbolic form.

M2  Cross-representation linear probe transfer
    ─ For each pair of source/target representation forms, train a
      multinomial logistic regression on the source activations and evaluate
      on the target activations.
    ─ 5-way classification (5 concepts).  Chance = 20 %.
    ─ High cross-form accuracy → representation-agnostic internal coding.

M3  Layer-wise silhouette analysis
    ─ At every layer, compute the silhouette coefficient when samples are
      grouped by concept label vs. when grouped by form label.
    ─ A crossing of the two curves indicates the depth at which conceptual
      structure dominates over surface form.

Bonus: CKA (Centered Kernel Alignment) between form-specific sub-spaces,
       per layer.  CKA ≈ 1 → the two sub-spaces are linearly equivalent.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_rdm(X: np.ndarray) -> np.ndarray:
    """
    Build an N×N Representational Dissimilarity Matrix using cosine distance.
    X: (N, D) float array.
    """
    return cdist(X, X, metric="cosine")


def _upper_triangle(M: np.ndarray) -> np.ndarray:
    """Return the strictly upper-triangular entries of a square matrix."""
    n = M.shape[0]
    idx = np.triu_indices(n, k=1)
    return M[idx]


def _theoretical_rdm(labels: List[int]) -> np.ndarray:
    """
    Build a binary RDM: 0 if same label, 1 if different label.
    """
    n = len(labels)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = 0.0 if labels[i] == labels[j] else 1.0
    return rdm


# ---------------------------------------------------------------------------
# M1 — RSA
# ---------------------------------------------------------------------------

def rsa_layer(
    X: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
) -> Tuple[float, float, float, float]:
    """
    Run RSA for a single layer.

    Parameters
    ----------
    X : (N, D) array of hidden states for this layer.
    concept_labels : list of int (length N)
    form_labels    : list of int (length N)

    Returns
    -------
    (corr_concept, p_concept, corr_form, p_form)
        Spearman correlations of the empirical RDM with the conceptual and
        form-based theoretical RDMs.
    """
    # Normalise (unit-norm rows) before cosine distance to get stable values
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    X_norm = X / norms

    empirical_rdm = _cosine_rdm(X_norm)
    concept_rdm   = _theoretical_rdm(concept_labels)
    form_rdm      = _theoretical_rdm(form_labels)

    emp_vec     = _upper_triangle(empirical_rdm)
    concept_vec = _upper_triangle(concept_rdm)
    form_vec    = _upper_triangle(form_rdm)

    corr_c, p_c = spearmanr(emp_vec, concept_vec)
    corr_f, p_f = spearmanr(emp_vec, form_vec)

    return float(corr_c), float(p_c), float(corr_f), float(p_f)


def rsa_all_layers(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
) -> Dict[str, np.ndarray]:
    """
    Run RSA across all layers.

    Parameters
    ----------
    activations : (N, L, D) — N stimuli, L layers, D hidden dim.

    Returns
    -------
    dict with keys: corr_concept, p_concept, corr_form, p_form
    Each value is a 1-D array of length L.
    """
    N, L, D = activations.shape
    out = {k: np.zeros(L) for k in ("corr_concept", "p_concept", "corr_form", "p_form")}
    for layer in range(L):
        X = activations[:, layer, :]
        cc, pc, cf, pf = rsa_layer(X, concept_labels, form_labels)
        out["corr_concept"][layer] = cc
        out["p_concept"][layer]    = pc
        out["corr_form"][layer]    = cf
        out["p_form"][layer]       = pf
    return out


# ---------------------------------------------------------------------------
# M2 — Cross-form linear probe transfer
# ---------------------------------------------------------------------------

def cross_probe_layer(
    X: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    n_forms: int = 4,
) -> np.ndarray:
    """
    For a single layer, train a probe on each source form and evaluate on
    every other target form.

    Returns
    -------
    transfer_matrix : (n_forms, n_forms) float array
        transfer_matrix[src, tgt] = accuracy when trained on src, tested on tgt.
        Diagonal = within-form leave-one-out accuracy (same form).
    """
    concept_labels_arr = np.array(concept_labels)
    form_labels_arr    = np.array(form_labels)
    matrix = np.zeros((n_forms, n_forms))

    for src in range(n_forms):
        src_mask = form_labels_arr == src
        X_train  = X[src_mask]
        y_train  = concept_labels_arr[src_mask]

        scaler  = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        clf = LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, solver="lbfgs"
        )
        # With only 5 samples per form we can't split further; evaluate train acc
        # for diagonal and direct transfer for off-diagonal.
        clf.fit(X_train_s, y_train)

        for tgt in range(n_forms):
            tgt_mask = form_labels_arr == tgt
            X_test   = X[tgt_mask]
            y_test   = concept_labels_arr[tgt_mask]
            X_test_s = scaler.transform(X_test)
            preds    = clf.predict(X_test_s)
            matrix[src, tgt] = np.mean(preds == y_test)

    return matrix


def cross_probe_all_layers(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    n_forms: int = 4,
) -> np.ndarray:
    """
    Run cross-probe for all layers.

    Returns
    -------
    np.ndarray of shape (L, n_forms, n_forms)
    """
    N, L, D = activations.shape
    results = np.zeros((L, n_forms, n_forms))
    for layer in range(L):
        results[layer] = cross_probe_layer(
            activations[:, layer, :], concept_labels, form_labels, n_forms
        )
    return results


def cross_probe_summary(transfer_tensor: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Aggregate cross-probe results.

    Parameters
    ----------
    transfer_tensor : (L, F, F)

    Returns
    -------
    dict:
        diagonal_mean  : (L,) mean within-form accuracy
        off_diag_mean  : (L,) mean cross-form transfer accuracy
        cross_vs_chance: (L,) (off_diag_mean - 0.2) / 0.2  —  relative gain over chance
    """
    L, F, _ = transfer_tensor.shape
    diagonal_mean = np.array([np.mean(np.diag(transfer_tensor[l])) for l in range(L)])
    # Mask diagonal
    mask = ~np.eye(F, dtype=bool)
    off_diag_mean = np.array([
        np.mean(transfer_tensor[l][mask]) for l in range(L)
    ])
    cross_vs_chance = (off_diag_mean - 1.0 / F) / (1.0 / F)
    return {
        "diagonal_mean":   diagonal_mean,
        "off_diag_mean":   off_diag_mean,
        "cross_vs_chance": cross_vs_chance,
    }


# ---------------------------------------------------------------------------
# M3 — Silhouette analysis
# ---------------------------------------------------------------------------

def silhouette_all_layers(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
) -> Dict[str, np.ndarray]:
    """
    Compute silhouette scores (with cosine metric) at every layer.

    Returns
    -------
    dict:
        sil_concept : (L,)  silhouette when grouped by concept
        sil_form    : (L,)  silhouette when grouped by form
    """
    N, L, D = activations.shape
    concept_arr = np.array(concept_labels)
    form_arr    = np.array(form_labels)
    sil_concept = np.zeros(L)
    sil_form    = np.zeros(L)

    for layer in range(L):
        X = activations[:, layer, :]
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        X_norm = X / norms

        # silhouette_score requires >= 2 unique labels and N > n_labels
        try:
            sil_concept[layer] = silhouette_score(X_norm, concept_arr, metric="cosine")
        except ValueError:
            sil_concept[layer] = 0.0
        try:
            sil_form[layer] = silhouette_score(X_norm, form_arr, metric="cosine")
        except ValueError:
            sil_form[layer] = 0.0

    return {"sil_concept": sil_concept, "sil_form": sil_form}


def find_crossover_layer(sil_concept: np.ndarray, sil_form: np.ndarray) -> int:
    """
    Return the first layer where sil_concept > sil_form.
    Returns -1 if no crossover is found.
    """
    for l in range(len(sil_concept)):
        if sil_concept[l] > sil_form[l]:
            return l
    return -1


# ---------------------------------------------------------------------------
# Bonus: CKA (Linear Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two representation matrices.

    Parameters
    ----------
    X, Y : (N, D) float arrays (same N, possibly different D).

    Returns
    -------
    CKA similarity in [0, 1].
    """
    # Centre columns
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices (N×N)
    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = np.sum(K * L)
    hsic_kk = np.sqrt(np.sum(K * K))
    hsic_ll = np.sqrt(np.sum(L * L))

    if hsic_kk < 1e-10 or hsic_ll < 1e-10:
        return 0.0
    return float(hsic_kl / (hsic_kk * hsic_ll))


def cka_cross_form_layer(
    X: np.ndarray,
    form_labels: List[int],
    n_forms: int = 4,
) -> np.ndarray:
    """
    For a single layer, compute CKA between every pair of form sub-spaces.

    Returns
    -------
    (n_forms, n_forms) symmetric CKA matrix (diagonal = 1).
    """
    form_arr = np.array(form_labels)
    cka_mat  = np.eye(n_forms)
    for f1, f2 in combinations(range(n_forms), 2):
        X1 = X[form_arr == f1]
        X2 = X[form_arr == f2]
        # CKA needs same N; zip to min length (all forms have same n here)
        min_n = min(len(X1), len(X2))
        cka_val = linear_cka(X1[:min_n], X2[:min_n])
        cka_mat[f1, f2] = cka_mat[f2, f1] = cka_val
    return cka_mat


def cka_all_layers(
    activations: np.ndarray,
    form_labels: List[int],
    n_forms: int = 4,
) -> np.ndarray:
    """
    CKA matrices for all layers.

    Returns
    -------
    (L, n_forms, n_forms)
    """
    N, L, D = activations.shape
    results = np.zeros((L, n_forms, n_forms))
    for layer in range(L):
        results[layer] = cka_cross_form_layer(
            activations[:, layer, :], form_labels, n_forms
        )
    return results


def cka_summary(cka_tensor: np.ndarray) -> np.ndarray:
    """
    Return mean off-diagonal CKA (cross-form alignment) per layer.

    Parameters
    ----------
    cka_tensor : (L, F, F)

    Returns
    -------
    (L,) array of mean off-diagonal CKA per layer.
    """
    L, F, _ = cka_tensor.shape
    mask = ~np.eye(F, dtype=bool)
    return np.array([np.mean(cka_tensor[l][mask]) for l in range(L)])


# ---------------------------------------------------------------------------
# Extended RSA — four theoretical RDMs (for RMR hypothesis)
# ---------------------------------------------------------------------------

def rsa_all_layers_v2(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    bias_rdm: np.ndarray,
    language_type_rdm: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Extended RSA with four theoretical RDMs:
      1. conceptual_RDM  — same concept = 0, different = 1
      2. form_RDM        — same surface form = 0, different = 1
      3. bias_RDM        — continuous structural-property distance
      4. language_type_RDM — same language type = 0, cross-type = 1

    RMR prediction:
      At deep layers, corr(empirical, conceptual) should be the strongest
      positive predictor, while corr(empirical, language_type) should NOT
      dominate after corr(empirical, bias) is accounted for.

    Returns
    -------
    dict with keys: corr_{concept,form,bias,language_type} and p_{...}
    Each value is a 1-D array of length L.
    """
    N, L, D = activations.shape

    concept_rdm = _theoretical_rdm(concept_labels)
    form_rdm    = _theoretical_rdm(form_labels)

    concept_vec = _upper_triangle(concept_rdm)
    form_vec    = _upper_triangle(form_rdm)
    bias_vec    = _upper_triangle(bias_rdm)
    lang_vec    = _upper_triangle(language_type_rdm)

    keys = ["corr_concept", "p_concept",
            "corr_form",    "p_form",
            "corr_bias",    "p_bias",
            "corr_language_type", "p_language_type"]
    out = {k: np.zeros(L) for k in keys}

    for layer in range(L):
        X = activations[:, layer, :]
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        emp_rdm = _cosine_rdm(X / norms)
        emp_vec = _upper_triangle(emp_rdm)

        cc, pc = spearmanr(emp_vec, concept_vec)
        cf, pf = spearmanr(emp_vec, form_vec)
        cb, pb = spearmanr(emp_vec, bias_vec)
        cl, pl = spearmanr(emp_vec, lang_vec)

        out["corr_concept"][layer]       = float(cc)
        out["p_concept"][layer]          = float(pc)
        out["corr_form"][layer]          = float(cf)
        out["p_form"][layer]             = float(pf)
        out["corr_bias"][layer]          = float(cb)
        out["p_bias"][layer]             = float(pb)
        out["corr_language_type"][layer] = float(cl)
        out["p_language_type"][layer]    = float(pl)

    return out


# ---------------------------------------------------------------------------
# Bias regression — OLS per layer (for RMR hypothesis test)
# ---------------------------------------------------------------------------

def bias_regression_layer(
    X_rep: np.ndarray,
    concept_labels: List[int],
    language_type_labels: List[int],
    bias_matrix: np.ndarray,
    bias_feature_names: List[str],
) -> Dict:
    """
    OLS regression of pairwise cosine distance on:
      same_concept        (binary)  — captures conceptual alignment
      same_language_type  (binary)  — the 'language essentialism' predictor
      |bias_i - bias_j|   (continuous, one per feature) — structural explanation

    All predictors are standardised (z-scored) for comparable β values.

    RMR predicts:
      β(same_concept) dominates  →  concept drives similarity
      β(same_language_type) is small after controlling for bias_features
    """
    from sklearn.linear_model import LinearRegression

    N = len(concept_labels)
    norms = np.linalg.norm(X_rep, axis=1, keepdims=True) + 1e-10
    dist_mat = _cosine_rdm(X_rep / norms)

    concept_arr = np.array(concept_labels)
    lang_arr    = np.array(language_type_labels)

    y_list: List[float] = []
    X_list: List[List[float]] = []

    for i in range(N):
        for j in range(i + 1, N):
            y_list.append(float(dist_mat[i, j]))
            bias_diff = np.abs(bias_matrix[i] - bias_matrix[j]).tolist()
            X_list.append(
                [float(concept_arr[i] == concept_arr[j]),
                 float(lang_arr[i] == lang_arr[j])]
                + bias_diff
            )

    y = np.array(y_list)
    X = np.array(X_list)

    # Standardise predictors for comparable β magnitudes
    means = X.mean(0)
    stds  = X.std(0)
    stds[stds < 1e-10] = 1.0
    X_std = (X - means) / stds

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_std, y)
    y_pred = reg.predict(X_std)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    pred_names = ["same_concept", "same_language_type"] + bias_feature_names
    betas = {n: float(b) for n, b in zip(pred_names, reg.coef_)}

    return {"betas": betas, "r2": r2, "n_pairs": len(y)}


def bias_regression_all_layers(
    activations: np.ndarray,
    concept_labels: List[int],
    language_type_labels: List[int],
    bias_matrix: np.ndarray,
    bias_feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Run bias regression at every transformer layer.

    Returns
    -------
    dict mapping each predictor name (and 'r2') to a 1-D array of length L.
    """
    N, L, D = activations.shape
    pred_names = ["same_concept", "same_language_type"] + bias_feature_names
    results: Dict[str, np.ndarray] = {n: np.zeros(L) for n in pred_names}
    results["r2"] = np.zeros(L)

    for layer in range(L):
        res = bias_regression_layer(
            activations[:, layer, :],
            concept_labels, language_type_labels,
            bias_matrix, bias_feature_names,
        )
        for n in pred_names:
            results[n][layer] = res["betas"][n]
        results["r2"][layer] = res["r2"]

    return results
