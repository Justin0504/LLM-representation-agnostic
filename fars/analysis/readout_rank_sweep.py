"""Readout-rank sweep: is the concept subspace readout-adjacent at ANY rank?

The top-k readout test in the main paper fixes k=10. The sharpest objection is
that concept information could be read out through lower-variance channels of
W_U -- say singular directions 50--200 -- which a top-10 comparison would miss.

This script answers that directly. For a model we compute the FULL right-singular
basis V of W_U (via the eigendecomposition of the D x D Gram matrix, which is
tractable for D <= 8192) and then, for every rank k, the fraction of a subspace's
energy that lies inside span(V[:, :k]):

    E_S(k) = mean over unit rows s of S  of  || V[:, :k]^T s ||^2

For a Haar-random rank-m subspace, E(k) = k/D exactly in expectation. That line
is the null. A subspace that is *preferentially* read out rises above k/D; one
that avoids the readout tracks or falls below it.

We sweep FARS, the final-layer PCA positive control, and a Haar-random subspace,
and report the full curve plus the area between each curve and the k/D null.

Usage:
    python3 analysis/readout_rank_sweep.py --model openai-community/gpt2-xl \
        --slug gpt2-xl --acts results/gpt2-xl/activations_cache.npz \
        --fars-basis results/gpt2-xl/fars_basis_L20.npz \
        --out analysis/readout_rank_sweep_gpt2-xl.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CANDIDATE_KEYS = [
    "lm_head.weight", "embed_out.weight", "output_projection.weight",
    "cls.predictions.decoder.weight", "transformer.wte.weight", "wte.weight",
    "model.embed_tokens.weight", "embed_tokens.weight",
    "model.decoder.embed_tokens.weight",
    "backbone.embeddings.weight", "backbone.embed_tokens.weight",
    "roberta.embeddings.word_embeddings.weight",   # RoBERTa MLM (tied)
    "deberta.embeddings.word_embeddings.weight",   # DeBERTa-v3 MLM (tied)
    "bert.embeddings.word_embeddings.weight",      # BERT MLM (tied)
]


def load_W_U(model_id: str) -> np.ndarray:
    """Download only the shard holding the unembedding weight and return it."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import json as _json

    shard = key = None
    try:
        idx = _json.load(open(hf_hub_download(model_id, "model.safetensors.index.json")))
        wmap = idx["weight_map"]
        for k in CANDIDATE_KEYS:
            if k in wmap:
                shard, key = wmap[k], k
                break
    except Exception:
        for name in ["model.safetensors", "pytorch_model.safetensors"]:
            try:
                p = hf_hub_download(model_id, name)
                with safe_open(p, framework="np") as f:
                    keys = list(f.keys())
                for k in CANDIDATE_KEYS:
                    if k in keys:
                        shard, key = name, k
                        break
                if key:
                    break
            except Exception:
                continue
    if key is None:
        # Some checkpoints (e.g. DeBERTa-v3) ship only pytorch_model.bin.
        return _load_W_U_from_bin(model_id)

    print(f"[locate] {model_id}: shard={shard} key={key}")
    local = hf_hub_download(model_id, shard)
    try:
        with safe_open(local, framework="np") as f:
            W = f.get_tensor(key)
        return np.asarray(W, dtype=np.float64)
    except Exception as e:
        # numpy has no bfloat16 dtype; read the raw safetensors bytes instead.
        print(f"  [np path failed: {e}] falling back to raw bf16 reader")
        return _read_tensor_raw(local, key)


def _load_W_U_from_bin(model_id: str) -> np.ndarray:
    """Fallback for checkpoints published only as a PyTorch .bin state dict."""
    from huggingface_hub import hf_hub_download
    import torch
    local = hf_hub_download(model_id, "pytorch_model.bin")
    sd = torch.load(local, map_location="cpu", weights_only=True)
    for k in CANDIDATE_KEYS:
        if k in sd:
            print(f"[locate] {model_id}: pytorch_model.bin key={k}")
            return sd[k].to(dtype=torch.float32).numpy().astype(np.float64)
    raise RuntimeError(
        f"cannot locate unembedding weight in {model_id}; saw {list(sd)[:6]}")


def _read_tensor_raw(path: str, key: str) -> np.ndarray:
    """Minimal safetensors reader that also handles BF16.

    Layout: [u64 header_len][JSON header][tensor data]. BF16 is the upper 16
    bits of the corresponding float32, so widening is a shift, not a conversion
    table.
    """
    import json as _json
    NP_DTYPE = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
                "I64": np.int64, "I32": np.int32, "I16": np.int16,
                "U8": np.uint8, "I8": np.int8}
    with open(path, "rb") as fh:
        header_len = int(np.frombuffer(fh.read(8), dtype="<u8")[0])
        header = _json.loads(fh.read(header_len).decode("utf-8"))
        meta = header[key]
        start, end = meta["data_offsets"]
        fh.seek(8 + header_len + start)
        buf = fh.read(end - start)

    dtype, shape = meta["dtype"], tuple(meta["shape"])
    if dtype == "BF16":
        u16 = np.frombuffer(buf, dtype="<u2")
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape).astype(np.float64)
    if dtype in NP_DTYPE:
        return np.frombuffer(buf, dtype=NP_DTYPE[dtype]).reshape(shape).astype(np.float64)
    raise ValueError(f"unsupported safetensors dtype {dtype}")


def full_right_singular_basis(W_U: np.ndarray) -> np.ndarray:
    """Right singular vectors of W_U, columns ordered by decreasing singular value.

    W_U is (V, D) or (D, V); either way the eigenvectors of W_U^T W_U (D x D)
    are the right singular vectors we want.
    """
    Gram = W_U.T @ W_U
    evals, evecs = np.linalg.eigh(Gram)
    order = np.argsort(evals)[::-1]
    return evecs[:, order]            # (D, D)


def orthonormal_rows(B: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(B.T)
    return Q.T


def energy_curve(S: np.ndarray, V: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Fraction of S's energy inside span(V[:, :k]) for each k in ks.

    Rows of S span the subspace; V has orthonormal columns.
    Because V is a complete orthonormal basis, the per-direction coefficients
    c = V^T s satisfy ||c||^2 = ||s||^2, so the cumulative sum of c^2 gives the
    curve for all k at once.
    """
    S = orthonormal_rows(S)                    # (m, D), unit rows
    C = S @ V                                  # (m, D) coefficients
    cum = np.cumsum(C ** 2, axis=1)            # (m, D)
    cum = cum / cum[:, -1:]                    # normalise (guards fp drift)
    return cum.mean(axis=0)[ks - 1]            # mean over the m directions


def _top_k_pca(M: np.ndarray, k: int) -> np.ndarray:
    M = M - M.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--fars-basis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10, help="subspace rank")
    ap.add_argument("--n-random-draws", type=int, default=32,
                    help="Haar draws averaged into the empirical null")
    args = ap.parse_args()
    k = args.k

    W_U = load_W_U(args.model)
    print(f"[W_U] {W_U.shape}")
    V = full_right_singular_basis(W_U)
    D = V.shape[0]
    del W_U

    acts = np.load(args.acts, allow_pickle=True)["activations"]
    fz = np.load(args.fars_basis, allow_pickle=True)
    layer = int(fz["layer"])
    B_fars = fz["basis"].astype(np.float64)
    X_last = acts[:, -1, :].astype(np.float64)
    B_last = _top_k_pca(X_last, k)

    # Log-spaced ranks plus every small rank, capped at D.
    ks = np.unique(np.concatenate([
        np.arange(1, min(33, D + 1)),
        np.unique(np.logspace(np.log10(32), np.log10(D), 60).astype(int)),
    ]))
    ks = ks[ks <= D]

    # A single Haar draw has non-trivial variance at small k (E(10)=10/D is a
    # tiny number), so the empirical null is averaged over many draws.
    n_draws = args.n_random_draws
    rng = np.random.default_rng(42)
    rand_curves = []
    for _ in range(n_draws):
        Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
        rand_curves.append(energy_curve(Q.T, V, ks))
    rand_curves = np.stack(rand_curves)

    curves = {
        "FARS": energy_curve(B_fars, V, ks),
        "LastPCA": energy_curve(B_last, V, ks),
        "Random": rand_curves.mean(axis=0),
    }
    rand_sd = rand_curves.std(axis=0)
    null = ks / D                                  # Haar expectation

    # Summary statistics.
    #   enrichment(k) = E(k) / (k/D)   -- how many times chance the loading is.
    #   half_rank     = smallest k with E(k) >= 0.5, normalised by D/2 (=1 for Haar).
    probe_ks = [kk for kk in (10, 32, 100, max(1, D // 16)) if kk <= D]

    def half_rank(c: np.ndarray) -> float:
        idx = np.searchsorted(c, 0.5)
        return float(ks[min(idx, len(ks) - 1)])

    result = {"model": args.slug, "layer": layer, "D": D, "K": k,
              "ks": ks.tolist(), "null_k_over_D": null.tolist(),
              "probe_ks": probe_ks, "n_random_draws": n_draws,
              "random_sd_curve": rand_sd.tolist()}
    print(f"[{args.slug}] D={D}, FARS layer L={layer}")
    for name, c in curves.items():
        enr = {}
        for kk in probe_ks:
            i = int(np.searchsorted(ks, kk))
            enr[str(kk)] = float(c[i] / (ks[i] / D))
        hr = half_rank(c)
        result[name] = {
            "energy_curve": c.tolist(),
            "enrichment_at_k": enr,
            "half_rank": hr,
            "half_rank_over_null": hr / (D / 2.0),
        }
    print(f"  {'k':>6s} {'null k/D':>9s} " + " ".join(f"{n:>9s}" for n in curves))
    for kk in probe_ks:
        i = int(np.searchsorted(ks, kk))
        print(f"  {ks[i]:6d} {null[i]:9.4f} " +
              " ".join(f"{c[i]:9.4f}" for c in curves.values()))
    print("  enrichment over chance  E(k)/(k/D):")
    for name in curves:
        e = result[name]["enrichment_at_k"]
        print(f"    {name:9s} " + "  ".join(f"k={kk}:{e[str(kk)]:5.2f}x" for kk in probe_ks))
    print(f"  [null check] Random enrichment should be ~1.00 at every k")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
