"""The REAL R3 test: does the FARS method reproduce its key property
(cross-format concept retrieval) on a COMPLETELY DIFFERENT concept inventory?

If Novelty-FARS achieves cross-format Agn% comparable to TriForm-FARS's on
the same models, the method is inventory-robust and the geometry is NOT
merely benchmark-induced. If Novelty-FARS collapses (e.g., Agn% near
chance = 1/10 = 10%), R3's concern is confirmed.

For each model with Novelty activations:
  1. Extract Novelty-FARS at best layer (concept-centroid PCA on 10 concepts)
  2. Project all 180 stimuli into the 10-d FARS
  3. Compute concept centroids in FARS
  4. For each stimulus, retrieve nearest concept centroid
  5. Report: top-1 accuracy overall (Agn%), cross-format only (X), within-format only (W)
"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, ".")
from benchmark.generate import generate_all_stimuli as triform_stim
from benchmark.novelty_concepts import generate_novelty_stimuli

K = 10


def fars_at(acts_L, concept_ids, k=K):
    n = int(concept_ids.max()) + 1
    C = np.zeros((n, acts_L.shape[-1]), dtype=np.float64)
    for c in range(n):
        C[c] = acts_L[concept_ids == c].mean(0)
    C = C - C.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt[:k]


def best_layer_by_ve(acts, ids, k=K, lo=0.30, hi=0.75):
    L = acts.shape[1]
    lo_i, hi_i = int(L * lo), int(L * hi)
    best = (-1, None, -1.0)
    for L_i in range(lo_i, hi_i + 1):
        acts_L = acts[:, L_i, :].astype(np.float32)
        # concept centroids
        n = int(ids.max()) + 1
        C = np.zeros((n, acts_L.shape[-1]), dtype=np.float64)
        for c in range(n):
            C[c] = acts_L[ids == c].mean(0)
        C = C - C.mean(0, keepdims=True)
        _, s, _ = np.linalg.svd(C, full_matrices=False)
        total = float((s ** 2).sum())
        ve = float((s[:k] ** 2).sum() / total) if total else 0.0
        if ve > best[2]:
            best = (L_i, C, ve)
    return best


def retrieval_metrics(acts_L, basis, concept_ids, form_ids):
    """Given projected acts, compute:
       - Agn% : overall top-1 concept retrieval accuracy
       - X    : cross-format top-1 (query and centroid from different forms)
       - W    : within-format top-1 (query and centroid from same form)
    """
    proj = acts_L @ basis.T  # (N, K)
    proj = proj - proj.mean(0, keepdims=True)
    n_concepts = int(concept_ids.max()) + 1
    n_forms = int(form_ids.max()) + 1

    # overall centroids (Agn%)
    C_all = np.zeros((n_concepts, K), dtype=np.float64)
    for c in range(n_concepts):
        C_all[c] = proj[concept_ids == c].mean(0)
    d = np.linalg.norm(proj[:, None, :] - C_all[None, :, :], axis=-1)
    pred = d.argmin(1)
    agn = float((pred == concept_ids).mean())

    # per-form centroids for X/W
    C_by_form = np.zeros((n_forms, n_concepts, K), dtype=np.float64)
    for f in range(n_forms):
        for c in range(n_concepts):
            m = (form_ids == f) & (concept_ids == c)
            if m.sum() > 0:
                C_by_form[f, c] = proj[m].mean(0)

    # cross-format: for each stimulus, use centroids from OTHER forms only
    X_correct = 0
    W_correct = 0
    for i in range(len(proj)):
        f_i = form_ids[i]
        # centroids averaged over other forms
        other_forms = [f for f in range(n_forms) if f != f_i]
        C_other = C_by_form[other_forms].mean(axis=0)  # (n_concepts, K)
        d_x = np.linalg.norm(proj[i] - C_other, axis=1)
        X_correct += int(d_x.argmin() == concept_ids[i])
        # within-format
        C_same = C_by_form[f_i]
        d_w = np.linalg.norm(proj[i] - C_same, axis=1)
        W_correct += int(d_w.argmin() == concept_ids[i])
    X = X_correct / len(proj)
    W = W_correct / len(proj)
    return {"agn": agn, "X": X, "W": W, "X_over_W": (X / W) if W > 0 else 0.0,
            "n_concepts": n_concepts, "chance": 1.0 / n_concepts}


# Model pairs
PAIRS = [
    ("qwen2.5-7b",
     "results/qwen2.5-7b/activations_cache.npz",
     "results_novelty/qwen2.5-7b/activations_cache.npz"),
    ("qwen2.5-3b-instruct",
     "results/qwen2.5-3b-instruct/activations_cache.npz",
     "results_novelty/qwen2.5-3b/activations_cache.npz"),
    ("phi-3.5-mini-instruct",
     "results/phi-3.5-mini-instruct/activations_cache.npz",
     "results_novelty/phi-3.5-mini-instruct/activations_cache.npz"),
    ("mistral-7b-v0.3",
     "results/mistral-7b-v0.3/activations_cache.npz",
     "results_novelty/mistral-7b-v0.3/activations_cache.npz"),
    ("gpt2-xl",
     "results/gpt2-xl/activations_cache.npz",
     "results_novelty/gpt2-xl/activations_cache.npz"),
    ("olmo-2-7b-instruct",
     "results/olmo-2-7b-instruct/activations_cache.npz",
     "results_novelty/olmo-2-7b-instruct/activations_cache.npz"),
    ("mamba-2.8b",
     "results/mamba-2.8b/activations_cache.npz",
     "results_novelty/mamba-2.8b/activations_cache.npz"),
    ("falcon-mamba-7b",
     "results/falcon-mamba-7b/activations_cache.npz",
     "results_novelty/falcon-mamba-7b/activations_cache.npz"),
    ("deberta-v3-large",
     "results/deberta-v3-large/activations_cache.npz",
     "results_novelty/deberta-v3-large/activations_cache.npz"),
    ("r1-distill-llama-8b",
     "results/r1-distill-llama-8b/activations_cache.npz",
     "results_novelty/r1-distill-llama-8b/activations_cache.npz"),
    ("r1-distill-qwen-7b",
     "results/r1-distill-qwen-7b/activations_cache.npz",
     "results_novelty/r1-distill-qwen-7b/activations_cache.npz"),
] + [  # completion batch (Delta): skipped automatically until the activations exist
    (s, f"results/{s}/activations_cache.npz", f"results_novelty/{s}/activations_cache.npz")
    for s in ["mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct", "deepseek-v2-lite-chat",
              "mixtral-8x7b-instruct", "llama-3.1-70b-instruct", "r1-distill-qwen-14b", "qwq-32b",
              "qwen3-8b", "phi-4", "gpt-oss-20b", "nemotron-h-8b", "granite-3.3-8b-instruct",
              "falcon3-7b-instruct", "yi-1.5-9b-chat", "smollm3-3b", "qwen3-4b"]
]

tri_stim = triform_stim()
nov_stim = generate_novelty_stimuli()
tri_ids = np.array([s.concept_id for s in tri_stim])
tri_forms = np.array([s.form_id for s in tri_stim])
nov_ids = np.array([s.concept_id for s in nov_stim])
nov_forms = np.array([s.form_id for s in nov_stim])

print(f"TriForm: {int(tri_ids.max())+1} concepts × {int(tri_forms.max())+1} forms, N={len(tri_ids)}")
print(f"Novelty: {int(nov_ids.max())+1} concepts × {int(nov_forms.max())+1} forms, N={len(nov_ids)}")

results = []
import os
for name, tri_path, nov_path in PAIRS:
    if not (os.path.exists(tri_path) and os.path.exists(nov_path)):
        print(f"\n=== {name} === [skip: activations missing]"); continue
    print(f"\n=== {name} ===")
    tri_acts = np.load(tri_path, allow_pickle=True)["activations"]
    nov_acts = np.load(nov_path, allow_pickle=True)["activations"]

    L_tri, _, ve_tri = best_layer_by_ve(tri_acts, tri_ids)
    L_nov, _, ve_nov = best_layer_by_ve(nov_acts, nov_ids)
    print(f"  Best layers: TriForm L={L_tri} (ve={ve_tri:.3f}), Novelty L={L_nov} (ve={ve_nov:.3f})")

    # TriForm-FARS at TriForm best layer, evaluated on TriForm
    B_tri = fars_at(tri_acts[:, L_tri, :].astype(np.float32), tri_ids)
    m_tri = retrieval_metrics(tri_acts[:, L_tri, :].astype(np.float32),
                              B_tri, tri_ids, tri_forms)
    print(f"  TriForm-FARS on TriForm: Agn%={m_tri['agn']:.3f}, X={m_tri['X']:.3f}, W={m_tri['W']:.3f}, X/W={m_tri['X_over_W']:.2f}  (chance {m_tri['chance']:.3f})")

    # Novelty-FARS at Novelty best layer, evaluated on Novelty
    B_nov = fars_at(nov_acts[:, L_nov, :].astype(np.float32), nov_ids)
    m_nov = retrieval_metrics(nov_acts[:, L_nov, :].astype(np.float32),
                              B_nov, nov_ids, nov_forms)
    print(f"  Novelty-FARS on Novelty: Agn%={m_nov['agn']:.3f}, X={m_nov['X']:.3f}, W={m_nov['W']:.3f}, X/W={m_nov['X_over_W']:.2f}  (chance {m_nov['chance']:.3f})")

    # Cross-transfer: TriForm-FARS on Novelty (project Novelty into TriForm basis)
    m_cross = retrieval_metrics(nov_acts[:, L_tri, :].astype(np.float32),
                                B_tri, nov_ids, nov_forms)
    print(f"  TriForm-FARS on Novelty (cross-transfer): Agn%={m_cross['agn']:.3f}, X={m_cross['X']:.3f}  (chance {m_cross['chance']:.3f})")

    # Baseline: random rank-10 orthonormal, evaluated on Novelty
    rng = np.random.default_rng(2027)
    Q, _ = np.linalg.qr(rng.standard_normal((nov_acts.shape[-1], K)))
    B_rand = Q.T
    m_rand = retrieval_metrics(nov_acts[:, L_nov, :].astype(np.float32),
                               B_rand, nov_ids, nov_forms)
    print(f"  Random rank-10 on Novelty: Agn%={m_rand['agn']:.3f}, X={m_rand['X']:.3f}  (chance {m_rand['chance']:.3f})")

    results.append({
        "model": name,
        "L_triform": L_tri, "L_novelty": L_nov,
        "triform_agn": m_tri["agn"], "triform_X": m_tri["X"], "triform_W": m_tri["W"], "triform_X_over_W": m_tri["X_over_W"],
        "novelty_agn": m_nov["agn"], "novelty_X": m_nov["X"], "novelty_W": m_nov["W"], "novelty_X_over_W": m_nov["X_over_W"],
        "triform_fars_on_novelty_agn": m_cross["agn"],
        "random_fars_on_novelty_agn": m_rand["agn"],
        "chance": m_nov["chance"],
    })

with open("./analysis/novelty_retrieval.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=========== SUMMARY: does the method transfer to novel concepts? ===========")
print(f"{'Model':30s} {'TriForm Agn%':>13s} {'Novelty Agn%':>13s} {'Novelty X/W':>12s} {'Random Agn%':>12s}")
for r in results:
    print(f"{r['model']:30s} {r['triform_agn']*100:>12.1f}% {r['novelty_agn']*100:>12.1f}% "
          f"{r['novelty_X_over_W']:>11.2f}x {r['random_fars_on_novelty_agn']*100:>11.1f}%")
