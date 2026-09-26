"""Does the end-of-CoT subspace carry the SAME cross-format concept identity
that the last-input-token subspace does?

Section 7's Q2 result showed that in R1-Distill models the CoT-tail FARS
DEPARTS from the last-token FARS (d_G ~4.3-4.5 out of ceiling 4.97).
That is a "boundary" finding for the paper's last-token pipeline. This
script asks the follow-up question: extract FARS from the CoT-tail
activations and measure cross-format concept retrieval on it. If the
number is comparable to last-token FARS retrieval, the CoT is depositing
reasoning in a DIFFERENT subspace that ALSO supports cross-format
identity -- turning Q2 from a boundary into a positive discovery.

Uses the R1-Distill activations extracted earlier (both `activations` and
`activations_cottail` keys in results/r1-distill-*/activations_cache.npz).
"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, ".")
from benchmark.generate import generate_all_stimuli

K = 10


def fars_at(acts_L, ids, k=K):
    n = int(ids.max()) + 1
    C = np.zeros((n, acts_L.shape[-1]), dtype=np.float64)
    for c in range(n):
        C[c] = acts_L[ids == c].mean(0)
    C = C - C.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt[:k]


def best_layer(acts, ids, k=K, lo=0.30, hi=0.75):
    L = acts.shape[1]
    lo_i, hi_i = int(L * lo), int(L * hi)
    best = (-1, None, -1.0)
    for L_i in range(lo_i, hi_i + 1):
        acts_L = acts[:, L_i, :].astype(np.float32)
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
    proj = acts_L @ basis.T
    proj = proj - proj.mean(0, keepdims=True)
    n_concepts = int(concept_ids.max()) + 1
    n_forms = int(form_ids.max()) + 1

    C_all = np.zeros((n_concepts, K), dtype=np.float64)
    for c in range(n_concepts):
        C_all[c] = proj[concept_ids == c].mean(0)
    d = np.linalg.norm(proj[:, None, :] - C_all[None, :, :], axis=-1)
    agn = float((d.argmin(1) == concept_ids).mean())

    C_by_form = np.zeros((n_forms, n_concepts, K), dtype=np.float64)
    for f in range(n_forms):
        for c in range(n_concepts):
            m = (form_ids == f) & (concept_ids == c)
            if m.sum() > 0:
                C_by_form[f, c] = proj[m].mean(0)

    X_c, W_c = 0, 0
    for i in range(len(proj)):
        f_i = form_ids[i]
        other = [f for f in range(n_forms) if f != f_i]
        C_other = C_by_form[other].mean(0)
        X_c += int(np.linalg.norm(proj[i] - C_other, axis=1).argmin() == concept_ids[i])
        W_c += int(np.linalg.norm(proj[i] - C_by_form[f_i], axis=1).argmin() == concept_ids[i])
    return {"agn": agn, "X": X_c / len(proj), "W": W_c / len(proj)}


def grassmann(B1, B2):
    B1 = B1 / np.linalg.norm(B1, axis=1, keepdims=True)
    B2 = B2 / np.linalg.norm(B2, axis=1, keepdims=True)
    _, s, _ = np.linalg.svd(B1 @ B2.T, full_matrices=False)
    ang = np.arccos(np.clip(s, -1, 1))
    return float(np.sqrt((ang ** 2).sum()))


MODELS = [
    ("r1-distill-llama-8b",
     "results/r1-distill-llama-8b/activations_cache.npz"),
    ("r1-distill-qwen-7b",
     "results/r1-distill-qwen-7b/activations_cache.npz"),
    ("qwq-32b",
     "results/qwq-32b/activations_cache.npz"),   # end-of-CoT states live in results_cot/
]
import os

stim = generate_all_stimuli()
concept_ids = np.array([s.concept_id for s in stim])
form_ids = np.array([s.form_id for s in stim])
n_concepts = int(concept_ids.max()) + 1
print(f"TriForm: {n_concepts} concepts × {int(form_ids.max())+1} forms, N={len(concept_ids)}")

results = []
for name, path in MODELS:
    print(f"\n=== {name} ===")
    z = np.load(path, allow_pickle=True)
    last = z["activations"]           # (N, L, D) at last-input-token
    cot_path = f"results_cot/{name}/activations_cache.npz"
    if "activations_cottail" in z.files:
        cot = z["activations_cottail"]    # (N, L, D) at end-of-CoT
    elif os.path.exists(cot_path):
        cot = np.load(cot_path, allow_pickle=True)["activations"]  # extract_cot_activations.py layout
    else:
        print(f"  no end-of-CoT activations, skipping")
        continue

    print(f"  Shapes: last={last.shape}, cot={cot.shape}")

    L_last, _, ve_last = best_layer(last, concept_ids)
    L_cot,  _, ve_cot  = best_layer(cot,  concept_ids)
    print(f"  Best layers: last L={L_last} (ve={ve_last:.3f}), cot L={L_cot} (ve={ve_cot:.3f})")

    B_last = fars_at(last[:, L_last, :].astype(np.float32), concept_ids)
    B_cot  = fars_at(cot [:, L_cot,  :].astype(np.float32), concept_ids)

    m_last = retrieval_metrics(last[:, L_last, :].astype(np.float32), B_last, concept_ids, form_ids)
    m_cot  = retrieval_metrics(cot [:, L_cot,  :].astype(np.float32), B_cot,  concept_ids, form_ids)
    m_cross = retrieval_metrics(cot[:, L_last, :].astype(np.float32), B_last, concept_ids, form_ids)  # last-FARS on CoT position

    print(f"  Last-token FARS on last-token: Agn%={m_last['agn']:.3f}, X={m_last['X']:.3f}, W={m_last['W']:.3f}")
    print(f"  CoT-tail   FARS on CoT-tail:   Agn%={m_cot['agn']:.3f}, X={m_cot['X']:.3f}, W={m_cot['W']:.3f}")
    print(f"  Last-token FARS on CoT-tail:   Agn%={m_cross['agn']:.3f}, X={m_cross['X']:.3f}")

    dG = grassmann(B_last, B_cot)
    print(f"  Grassmann d_G(FARS_last, FARS_cot) = {dG:.3f}")

    # Baseline: random rank-10 on CoT-tail
    rng = np.random.default_rng(2027)
    Q, _ = np.linalg.qr(rng.standard_normal((cot.shape[-1], K)))
    B_rand = Q.T
    m_rand = retrieval_metrics(cot[:, L_cot, :].astype(np.float32), B_rand, concept_ids, form_ids)
    print(f"  Random rank-10 on CoT-tail:    Agn%={m_rand['agn']:.3f} (chance {1/n_concepts:.3f})")

    results.append({
        "model": name,
        "L_last": L_last, "L_cot": L_cot,
        "last_agn": m_last["agn"], "last_X": m_last["X"], "last_W": m_last["W"],
        "cot_agn":  m_cot["agn"],  "cot_X":  m_cot["X"],  "cot_W":  m_cot["W"],
        "last_fars_on_cot_position_agn": m_cross["agn"],
        "random_on_cot_position_agn": m_rand["agn"],
        "d_G_last_vs_cot": dG,
    })

with open("./analysis/cot_fars_retrieval.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=========== CoT-FARS RETRIEVAL SUMMARY ===========")
print(f"{'Model':25s} {'Last Agn%':>10s} {'CoT Agn%':>9s} {'CrossFARS':>10s} {'Random':>8s} {'d_G':>6s}")
for r in results:
    print(f"{r['model']:25s} {r['last_agn']*100:>9.1f}% {r['cot_agn']*100:>8.1f}% "
          f"{r['last_fars_on_cot_position_agn']*100:>9.1f}% {r['random_on_cot_position_agn']*100:>7.1f}% "
          f"{r['d_G_last_vs_cot']:>6.2f}")
