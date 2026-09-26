"""Reference implementation of the raw equal-rank readout diagnostic.

This is newly written verification code, not the missing original experiment
pipeline. It does not extract FARS or fit the depth-matched translator.
Inputs are .npy arrays: B is (k, D); W_U is (vocabulary, D).
Alternatively provide an already computed (k, D) readout row basis.
Computing a full SVD of a large unembedding can be expensive; nothing is fetched.
"""
import argparse
import json
import numpy as np


def _basis(a, name):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not 0 < a.shape[0] <= a.shape[1]:
        raise ValueError(f'{name} must be a nonempty k-by-D row basis, k <= D')
    if not np.isfinite(a).all():
        raise ValueError(f'{name} contains nonfinite values')
    if not np.allclose(a @ a.T, np.eye(a.shape[0]), atol=1e-6, rtol=1e-6):
        raise ValueError(f'{name} rows must be orthonormal; no silent re-estimation')
    return a


def raw_readout_basis(unembedding, rank):
    w = np.asarray(unembedding, dtype=np.float64)
    if w.ndim != 2 or not np.isfinite(w).all():
        raise ValueError('W_U must be a finite vocabulary-by-D matrix')
    if not 0 < rank <= min(w.shape):
        raise ValueError('rank exceeds an input dimension')
    # NumPy returns right singular directions as rows, in descending order.
    _, singular, vt = np.linalg.svd(w, full_matrices=False)
    if singular[rank - 1] <= np.finfo(float).eps * max(w.shape) * singular[0]:
        raise ValueError('W_U has fewer than k numerically nonzero directions')
    return vt[:rank]


def compare(basis, readout_basis):
    b = _basis(basis, 'B')
    v = _basis(readout_basis, 'V_k')
    if b.shape != v.shape:
        raise ValueError('This diagnostic requires equal rank and hidden dimension')
    k, d = b.shape
    singular = np.linalg.svd(b @ v.T, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    singular[np.isclose(singular, 1.0, atol=1e-14, rtol=0)] = 1.0
    angles = np.arccos(singular)
    energy = float(np.linalg.norm(b @ v.T, 'fro') ** 2 / k)
    return {
        'rank': k, 'hidden_dimension': d,
        'principal_angles_radians': angles.tolist(),
        'grassmann_distance': float(np.linalg.norm(angles)),
        'rank_ceiling': float(np.sqrt(k) * np.pi / 2),
        'energy_fraction': energy, 'energy_percent': 100 * energy,
        'haar_expected_energy_fraction': k / d,
        'enrichment_over_haar_expectation': energy / (k / d),
        'scope': 'Raw unembedding reference; no depth matching or causal verdict',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basis', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--readout-basis')
    g.add_argument('--unembedding')
    args = p.parse_args()
    b = _basis(np.load(args.basis, allow_pickle=False), 'B')
    v = (np.load(args.readout_basis, allow_pickle=False) if args.readout_basis
         else raw_readout_basis(np.load(args.unembedding, allow_pickle=False), b.shape[0]))
    print(json.dumps(compare(b, v), indent=2))
