"""
Robust statistical methods for representation analysis.

Fixes the key issues in the original pilot:
1. Permutation tests for RSA (respects non-independence of pairwise distances)
2. Block bootstrap for regression (accounts for shared stimuli)
3. Ridge regression probes with cross-validation (prevents overfitting)
4. FDR correction for multiple comparisons
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut


# ---------------------------------------------------------------------------
# Permutation-based RSA
# ---------------------------------------------------------------------------

def permutation_rsa(
    empirical_rdm: np.ndarray,
    theoretical_rdm: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Permutation test for RSA that respects non-independence of pairwise distances.

    Instead of permuting individual distance pairs (which assumes independence),
    we permute the stimulus labels and recompute the full RDM, which preserves
    the dependency structure.

    Parameters
    ----------
    empirical_rdm : (N, N) symmetric distance matrix
    theoretical_rdm : (N, N) symmetric theoretical RDM
    n_permutations : number of permutations

    Returns
    -------
    (observed_rho, p_value)
    """
    N = empirical_rdm.shape[0]
    idx_upper = np.triu_indices(N, k=1)

    emp_vec = empirical_rdm[idx_upper]
    theo_vec = theoretical_rdm[idx_upper]
    observed_rho, _ = spearmanr(emp_vec, theo_vec)

    rng = np.random.RandomState(seed)
    count_ge = 0

    for _ in range(n_permutations):
        # Permute rows AND columns of theoretical RDM (equivalent to permuting labels)
        perm = rng.permutation(N)
        perm_rdm = theoretical_rdm[np.ix_(perm, perm)]
        perm_vec = perm_rdm[idx_upper]
        perm_rho, _ = spearmanr(emp_vec, perm_vec)
        if perm_rho >= observed_rho:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return float(observed_rho), float(p_value)


def permutation_rsa_all_layers(
    activations: np.ndarray,
    theoretical_rdms: Dict[str, np.ndarray],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Run permutation RSA for all layers against multiple theoretical RDMs.

    Parameters
    ----------
    activations : (N, L, D)
    theoretical_rdms : dict mapping name -> (N, N) RDM
    n_permutations : number of permutations per test

    Returns
    -------
    dict with keys: corr_{name}, p_{name} for each theoretical RDM
    """
    N, L, D = activations.shape
    results = {}
    for name in theoretical_rdms:
        results[f"corr_{name}"] = np.zeros(L)
        results[f"p_{name}"] = np.zeros(L)

    for layer in range(L):
        X = activations[:, layer, :]
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        emp_rdm = cdist(X / norms, X / norms, metric="cosine")

        for name, theo_rdm in theoretical_rdms.items():
            rho, p = permutation_rsa(emp_rdm, theo_rdm, n_permutations, seed + layer)
            results[f"corr_{name}"][layer] = rho
            results[f"p_{name}"][layer] = p

    return results


# ---------------------------------------------------------------------------
# Ridge probe with leave-one-concept-out CV
# ---------------------------------------------------------------------------

def ridge_probe_cv(
    X: np.ndarray,
    concept_labels: np.ndarray,
    form_labels: np.ndarray,
    source_form: int,
    target_form: int,
    alpha: float = 0.1,
) -> Tuple[float, float]:
    """
    Train ridge classifier on source_form, test on target_form,
    using leave-one-concept-out cross-validation.

    Returns (mean_accuracy, std_accuracy) across CV folds.
    """
    src_mask = form_labels == source_form
    tgt_mask = form_labels == target_form

    X_src = X[src_mask]
    y_src = concept_labels[src_mask]
    X_tgt = X[tgt_mask]
    y_tgt = concept_labels[tgt_mask]

    if source_form == target_form:
        # Within-form: use leave-one-out
        accuracies = []
        for i in range(len(X_src)):
            mask = np.ones(len(X_src), dtype=bool)
            mask[i] = False
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_src[mask])
            X_test = scaler.transform(X_src[~mask])
            clf = RidgeClassifier(alpha=alpha)
            clf.fit(X_train, y_src[mask])
            pred = clf.predict(X_test)
            accuracies.append(float(pred[0] == y_src[i]))
        return float(np.mean(accuracies)), float(np.std(accuracies))
    else:
        # Cross-form: train on all source, test on all target
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_src)
        X_test = scaler.transform(X_tgt)
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(X_train, y_src)
        preds = clf.predict(X_test)
        acc = float(np.mean(preds == y_tgt))
        return acc, 0.0


def ridge_probe_all_layers(
    activations: np.ndarray,
    concept_labels: List[int],
    form_labels: List[int],
    n_forms: int = 6,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ridge probe transfer matrix for all layers.

    Returns
    -------
    (accuracy_tensor, std_tensor) each of shape (L, n_forms, n_forms)
    """
    N, L, D = activations.shape
    c_arr = np.array(concept_labels)
    f_arr = np.array(form_labels)

    acc_tensor = np.zeros((L, n_forms, n_forms))
    std_tensor = np.zeros((L, n_forms, n_forms))

    for layer in range(L):
        X = activations[:, layer, :]
        for src in range(n_forms):
            for tgt in range(n_forms):
                acc, std = ridge_probe_cv(X, c_arr, f_arr, src, tgt, alpha)
                acc_tensor[layer, src, tgt] = acc
                std_tensor[layer, src, tgt] = std

    return acc_tensor, std_tensor


# ---------------------------------------------------------------------------
# Block bootstrap regression
# ---------------------------------------------------------------------------

def block_bootstrap_regression(
    activations_layer: np.ndarray,
    concept_labels: List[int],
    language_type_labels: List[int],
    bias_matrix: np.ndarray,
    bias_feature_names: List[str],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Block bootstrap for pairwise distance regression.

    Blocks are defined by stimuli: when resampling, we resample entire stimuli
    (rows of the activation matrix), which preserves the dependency structure
    of pairwise distances.

    Returns
    -------
    dict mapping predictor_name -> (beta, ci_lower, ci_upper)
    """
    from sklearn.linear_model import LinearRegression

    N = len(concept_labels)
    concept_arr = np.array(concept_labels)
    lang_arr = np.array(language_type_labels)
    rng = np.random.RandomState(seed)

    pred_names = ["same_concept", "same_language_type"] + bias_feature_names

    def _fit_one(X_rep, c_arr, l_arr, b_mat):
        norms = np.linalg.norm(X_rep, axis=1, keepdims=True) + 1e-10
        dist_mat = cdist(X_rep / norms, X_rep / norms, metric="cosine")
        n = len(c_arr)

        y_list, feat_list = [], []
        for i in range(n):
            for j in range(i + 1, n):
                y_list.append(dist_mat[i, j])
                bias_diff = np.abs(b_mat[i] - b_mat[j]).tolist()
                feat_list.append(
                    [float(c_arr[i] == c_arr[j]), float(l_arr[i] == l_arr[j])]
                    + bias_diff
                )

        y = np.array(y_list)
        X = np.array(feat_list)
        means, stds = X.mean(0), X.std(0)
        stds[stds < 1e-10] = 1.0
        X_std = (X - means) / stds

        reg = LinearRegression(fit_intercept=True)
        reg.fit(X_std, y)
        return dict(zip(pred_names, reg.coef_))

    # Observed betas
    observed = _fit_one(activations_layer, concept_arr, lang_arr, bias_matrix)

    # Bootstrap
    boot_betas = {name: [] for name in pred_names}
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        betas = _fit_one(
            activations_layer[idx],
            concept_arr[idx],
            lang_arr[idx],
            bias_matrix[idx],
        )
        for name in pred_names:
            boot_betas[name].append(betas[name])

    results = {}
    for name in pred_names:
        boot_arr = np.array(boot_betas[name])
        ci_lo = float(np.percentile(boot_arr, 2.5))
        ci_hi = float(np.percentile(boot_arr, 97.5))
        results[name] = (observed[name], ci_lo, ci_hi)

    return results


# ---------------------------------------------------------------------------
# FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Returns
    -------
    (rejected, adjusted_p_values)
        rejected: boolean array, True where null is rejected
        adjusted_p_values: FDR-adjusted p-values
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH adjusted p-values
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            sorted_p[i] * n / (i + 1),
            adjusted[sorted_idx[i + 1]]
        )

    rejected = adjusted <= alpha
    return rejected, adjusted


# ---------------------------------------------------------------------------
# Dimension-wise analysis
# ---------------------------------------------------------------------------

def analyze_by_dimension(
    activations: np.ndarray,
    form_labels: List[int],
    form_names: List[str],
    dimension_pairs: Dict[str, List[Tuple[str, str]]],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Analyze representational alignment along each surface form dimension
    (linguistic, symbolic, structural) separately.

    For each dimension, compute mean CKA across the relevant form pairs.

    Returns
    -------
    dict: dimension_name -> {"cka_mean": (L,), "pairs": list of pair results}
    """
    from src.metrics import linear_cka

    N, L, D = activations.shape
    f_arr = np.array(form_labels)
    form_to_idx = {name: i for i, name in enumerate(form_names)}

    results = {}
    for dim_name, pairs in dimension_pairs.items():
        pair_ckas = []
        for form_a, form_b in pairs:
            idx_a = form_to_idx.get(form_a)
            idx_b = form_to_idx.get(form_b)
            if idx_a is None or idx_b is None:
                continue

            cka_per_layer = np.zeros(L)
            for layer in range(L):
                X_a = activations[f_arr == idx_a, layer, :]
                X_b = activations[f_arr == idx_b, layer, :]
                min_n = min(len(X_a), len(X_b))
                if min_n < 2:
                    continue
                cka_per_layer[layer] = linear_cka(X_a[:min_n], X_b[:min_n])
            pair_ckas.append(cka_per_layer)

        if pair_ckas:
            results[dim_name] = {
                "cka_mean": np.mean(pair_ckas, axis=0),
                "cka_std": np.std(pair_ckas, axis=0),
                "n_pairs": len(pair_ckas),
            }

    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metric_ci(
    values: np.ndarray,
    group_labels: np.ndarray,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Block-bootstrap CI by resampling groups (e.g., concepts).

    Parameters
    ----------
    values : 1D array of metric values (one per observation)
    group_labels : group id for each observation (bootstrap unit)
    n_bootstrap : number of resamples
    alpha : significance level (0.05 = 95% CI)

    Returns dict with mean, ci_low, ci_high, std
    """
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(group_labels)
    n_groups = len(unique_groups)

    boot_means = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sampled_vals = np.concatenate([values[group_labels == g] for g in sampled_groups])
        boot_means[b] = np.mean(sampled_vals)

    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "std": float(np.std(boot_means)),
    }
