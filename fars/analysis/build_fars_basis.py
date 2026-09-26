"""Generate FARS basis npz file for R1-Distill-Llama-8B at the best layer.

Reads activations_cache.npz and writes fars_basis_L{L}.npz in the same
format as other bases in results/ (keys: basis, layer).
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, ".")
from benchmark.generate import generate_all_stimuli

K = 10

p = argparse.ArgumentParser()
p.add_argument("--acts", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--layer-range-lo", type=float, default=0.30)
p.add_argument("--layer-range-hi", type=float, default=0.75)
args = p.parse_args()

acts = np.load(args.acts, allow_pickle=True)["activations"]  # (N, L, D)
stim = generate_all_stimuli()
ids = np.array([s.concept_id for s in stim])

def fars(acts_L, ids, k=K):
    n = int(ids.max()) + 1
    C = np.zeros((n, acts_L.shape[-1]), dtype=np.float64)
    for c in range(n):
        C[c] = acts_L[ids == c].mean(0)
    C = C - C.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(C, full_matrices=False)
    total = float((s ** 2).sum())
    return Vt[:k], (float((s[:k] ** 2).sum() / total) if total else 0.0)

L = acts.shape[1]
lo, hi = int(L * args.layer_range_lo), int(L * args.layer_range_hi)
best = (-1, None, -1.0)
for L_i in range(lo, hi + 1):
    B, ve = fars(acts[:, L_i, :].astype(np.float32), ids)
    if ve > best[2]:
        best = (L_i, B, ve)

layer, basis, ve = best
print(f"[best] L={layer} ve={ve:.4f}")

os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, f"fars_basis_L{layer}.npz")
np.savez(out_path, basis=basis, layer=layer)
print(f"[write] {out_path}")
