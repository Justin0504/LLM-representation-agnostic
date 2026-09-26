#!/usr/bin/env python3
"""Run the readout-orthogonality test on published concept directions.

Inputs (per model):
    analysis/published_dirs/<slug>.npz   refusal / truth difference-of-means
                                          directions at the FARS block (published_subspaces.py)
    analysis/depth_matched/<slug>.npz    W_U top-100 right singular vectors and the
                                          readout channels pulled back to the same block,
                                          plus the same-layer next-token centroid PCA
    results/<slug>/fars_basis_L*.npz   FARS basis (its first PC is the rank-1 comparison)

For a unit direction v and a readout basis R (k x D, orthonormal rows) the
readout energy is ||R v||^2, whose Haar expectation is k/D. We report it for
    refusal, truth, FARS-PC1, next-token-PC1 (same block), Haar random (mean of 200)
against both W_U (k=10) and the pulled-back readout (k=10), together with each
direction's held-out separability AUC. Output: analysis/published_<slug>.json.
"""
from __future__ import annotations
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 10


def energy(v: np.ndarray, R: np.ndarray) -> float:
    Q, _ = np.linalg.qr(R.T); R = Q.T
    v = v / np.linalg.norm(v)
    return float(((R @ v) ** 2).sum())


def main() -> None:
    for f in sorted(glob.glob(os.path.join(ROOT, "analysis/published_dirs/*.npz"))):
        slug = os.path.basename(f)[:-4]
        dm_f = os.path.join(ROOT, f"analysis/depth_matched/{slug}.npz")
        fars_f = glob.glob(os.path.join(ROOT, f"results/{slug}/fars_basis_L*.npz"))
        if not fars_f:
            print(f"[skip] {slug}: no FARS basis"); continue
        # The depth-matched control needs a next-token positive control, which a masked
        # LM does not have, so one model legitimately has no depth_matched run. Its
        # direction can still be measured against W_U itself -- that is what the depth
        # contrast uses -- so report the W_U half rather than dropping the model.
        has_dm = os.path.exists(dm_f)
        p = np.load(f)
        dm = np.load(dm_f) if has_dm else None
        block_mismatch = has_dm and int(p["layer"]) != int(dm["layer"])
        if block_mismatch:
            # The pulled-back readout is defined at the depth-matched run's own block,
            # which for a few models differs by one from the block the direction was
            # built at. Record it so the table can mark those rows rather than present
            # them as matched-depth comparisons.
            print(f"[warn] {slug}: published block {int(p['layer'])} != depth-matched block {int(dm['layer'])}")
        fars = np.load(fars_f[0], allow_pickle=True)["basis"].astype(np.float64)
        D = int(p["D"])
        if has_dm:
            bases = {"wu": dm["wu_top100"][:K].astype(np.float64),
                     "readout_L": dm["readout_L_top100"][:K].astype(np.float64)}
        else:
            import glob as _g
            rz = _g.glob(os.path.join(ROOT, f"analysis/readout_top10_{slug}.npz"))
            if not rz:
                print(f"[skip] {slug}: no depth_matched and no readout basis"); continue
            z = np.load(rz[0], allow_pickle=True)
            bases = {"wu": z["basis" if "basis" in z.files else "top_k_readout"][:K].astype(np.float64)}
        dirs = {"refusal": p["refusal_dir"].astype(np.float64), "truth": p["truth_dir"].astype(np.float64),
                "FARS_pc1": fars[0],
                "refusal_last": p["refusal_dir_last"].astype(np.float64),
                "truth_last": p["truth_dir_last"].astype(np.float64)}
        if has_dm:
            dirs["NextTok_pc1"] = dm["nexttok_pca_L"][0].astype(np.float64)
        rng = np.random.default_rng(0)
        rand = rng.standard_normal((200, D))
        res = {"model": slug, "block": int(p["layer"]), "D": D, "K": K, "haar_pct": 100.0 * K / D,
               "refusal_auc": float(p["refusal_auc"]), "truth_auc": float(p["truth_auc"]),
               "block_mismatch": block_mismatch, "has_depth_matched": bool(has_dm),
               "dm_block": int(dm["layer"]) if has_dm else None}
        # Final-block separability, recorded only for runs made after the depth
        # contrast was added: it checks that the final-block direction is still a
        # working classifier, not a degenerate vector near the readout span.
        for k_ in ("refusal_auc_last", "truth_auc_last"):
            if k_ in p.files:
                res[k_] = float(p[k_])
        for b, R in bases.items():
            res[b] = {n: 100 * energy(v, R) for n, v in dirs.items()}
            res[b]["Random"] = float(np.mean([100 * energy(r, R) for r in rand]))
        e = lambda b, n: res[b][n]
        if not has_dm:
            print(f"[{slug}] blk {res['block']} (no depth-matched run; W_U columns only) "
                  f"refusal {e('wu','refusal'):.2f}% truth {e('wu','truth'):.2f}% "
                  f"| last-block {e('wu','refusal_last'):.2f}% / {e('wu','truth_last'):.2f}%")
            json.dump(res, open(os.path.join(ROOT, f"analysis/published_{slug}.json"), "w"), indent=2)
            continue
        print(f"[{slug}] blk {res['block']} AUC ref {res['refusal_auc']:.2f} truth {res['truth_auc']:.2f} | "
              f"vs W_U: refusal {e('wu','refusal'):.2f}% truth {e('wu','truth'):.2f}% FARS1 {e('wu','FARS_pc1'):.2f}% "
              f"NextTok1 {e('wu','NextTok_pc1'):.2f}% rand {e('wu','Random'):.2f}% | vs readout_L: refusal {e('readout_L','refusal'):.2f}% "
              f"truth {e('readout_L','truth'):.2f}% FARS1 {e('readout_L','FARS_pc1'):.2f}% NextTok1 {e('readout_L','NextTok_pc1'):.2f}% "
              f"rand {e('readout_L','Random'):.2f}% | last-block refusal {e('wu','refusal_last'):.2f}% truth {e('wu','truth_last'):.2f}%")
        json.dump(res, open(os.path.join(ROOT, f"analysis/published_{slug}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
