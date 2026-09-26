"""Layer-localization sweep for FARS.

For each of L layers in a model:
    1. Extract layer-L-FARS: concept-centroid PCA on 18 concept centroids
       computed from activations[:, L, :]
    2. Compute:
        - VE_top_10 : fraction of concept-centroid variance captured by top-10
        - Grassmann distance d_G(layer-L-FARS, top-10 readout channels of W_U)
        - Agn% : cross-format concept retrieval top-1 using layer-L-FARS
    3. Also record L_best from the FARS extraction pipeline (loaded metadata)

Output: one JSON per model with the per-layer profile.

The paper's claim: FARS is layer-LOCAL, i.e. the concept-invariance signal is
concentrated in a middle-layer band, not distributed. The plot should show a
single unimodal peak in mid layers.

Usage on <cluster>:
    python3 analysis/layer_localization.py \\
        --slug qwen2.5-7b \\
        --acts results/qwen2.5-7b/activations_cache.npz \\
        --top10-readout analysis/readout_top10_qwen2.5-7b.npz \\
        --out analysis/layer_loc_qwen2.5-7b.json
"""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np

K = 10


def _concept_form_ids(n_stim: int) -> tuple[np.ndarray, np.ndarray]:
    # TriForm: 18 concepts x 6 forms x 3 instances = 324, order determined by
    # benchmark/generate.generate_all_stimuli. Reconstruct via that iterator.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from benchmark.generate import generate_all_stimuli
    stim = generate_all_stimuli()
    assert len(stim) == n_stim, (len(stim), n_stim)
    return (np.array([s.concept_id for s in stim]),
            np.array([s.form_id for s in stim]))


def fars_from_layer(acts_L: np.ndarray, concept_ids: np.ndarray, k: int = K):
    n = int(concept_ids.max()) + 1
    C = np.zeros((n, acts_L.shape[-1]), dtype=np.float64)
    for c in range(n):
        C[c] = acts_L[concept_ids == c].mean(0)
    C = C - C.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(C, full_matrices=False)
    total = float((s ** 2).sum())
    ve_top_k = float((s[:k] ** 2).sum() / total) if total > 0 else 0.0
    return Vt[:k], ve_top_k


def grassmann(B1: np.ndarray, B2: np.ndarray) -> float:
    B1 = B1 / np.linalg.norm(B1, axis=1, keepdims=True)
    B2 = B2 / np.linalg.norm(B2, axis=1, keepdims=True)
    _, s, _ = np.linalg.svd(B1 @ B2.T, full_matrices=False)
    return float(np.sqrt((np.arccos(np.clip(s, -1, 1)) ** 2).sum()))


def energy_in(B_test: np.ndarray, B_span: np.ndarray) -> float:
    coeffs = B_test.astype(np.float64) @ B_span.astype(np.float64).T
    e_in = (coeffs ** 2).sum(axis=1)
    e_all = (B_test.astype(np.float64) ** 2).sum(axis=1)
    return float((e_in / e_all).mean())


def cross_format_agn(acts_L: np.ndarray, basis: np.ndarray,
                     concept_ids: np.ndarray, form_ids: np.ndarray) -> float:
    proj = acts_L.astype(np.float64) @ basis.T
    proj = proj - proj.mean(0, keepdims=True)
    n_concepts = int(concept_ids.max()) + 1
    n_forms = int(form_ids.max()) + 1
    C_by_form = np.zeros((n_forms, n_concepts, proj.shape[1]), dtype=np.float64)
    for f in range(n_forms):
        for c in range(n_concepts):
            m = (form_ids == f) & (concept_ids == c)
            if m.sum() > 0:
                C_by_form[f, c] = proj[m].mean(0)
    hits = 0
    for i in range(len(proj)):
        f_i = form_ids[i]
        others = [f for f in range(n_forms) if f != f_i]
        C = C_by_form[others].mean(axis=0)
        d = np.linalg.norm(proj[i] - C, axis=1)
        hits += int(d.argmin() == concept_ids[i])
    return hits / len(proj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--top10-readout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="Skip factor to reduce compute on huge models")
    args = ap.parse_args()

    print(f"[load] acts {args.acts}")
    acts = np.load(args.acts, allow_pickle=True)["activations"]
    n_stim, n_layers, D = acts.shape
    print(f"[shape] N={n_stim}, L={n_layers}, D={D}")
    concept_ids, form_ids = _concept_form_ids(n_stim)

    readout = np.load(args.top10_readout, allow_pickle=True)
    B_readout = (readout["basis"] if "basis" in readout.files
                 else readout["top_k_readout"]).astype(np.float64)

    per_layer = []
    for L in range(0, n_layers, args.layer_stride):
        acts_L = acts[:, L, :].astype(np.float64)
        basis, ve = fars_from_layer(acts_L, concept_ids, args.k)
        dG = grassmann(basis, B_readout)
        e = energy_in(basis, B_readout)
        agn = cross_format_agn(acts_L, basis, concept_ids, form_ids)
        per_layer.append({
            "layer": L, "ve_top_k": ve, "d_G_vs_readout": dG,
            "energy_in_top10_readout_pct": e * 100, "agn": agn,
        })
        print(f"  L={L:3d}  ve={ve:.3f}  d_G={dG:.3f}  e%={e*100:5.2f}  Agn%={agn*100:5.1f}")

    best = max(per_layer, key=lambda r: r["agn"])
    result = {
        "model": args.slug, "K": args.k, "n_layers": n_layers, "D": D,
        "layer_stride": args.layer_stride,
        "best_by_agn": best,
        "per_layer": per_layer,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] best Agn% at L={best['layer']} = {best['agn']*100:.1f}%; wrote {args.out}")


if __name__ == "__main__":
    main()
