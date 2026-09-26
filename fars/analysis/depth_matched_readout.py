#!/usr/bin/env python3
"""Depth-matched readout channels for the readout-orthogonality test.

The objection this answers: a mid-layer subspace may be far from the top-k
right singular vectors of W_U simply because many nonlinear layers separate
layer L from the unembedding, so "orthogonal to the readout" could just mean
"not at the last layer". To test at matched depth we pull the readout back to
layer L with a linear translator (tuned-lens style):

    h_last ~= A_L h_L + b            (ridge, fitted on a generic text corpus)
    logits  = W_U * gamma ⊙ LN(h_last)

so the readout channels *at layer L* are the top-k right singular vectors of
M_L = W_U diag(gamma) P A_L  (P = mean-removal for LayerNorm, identity for
RMSNorm). We also build a depth-matched POSITIVE control at the same layer:
next-token-identity centroid PCA (group layer-L states by the token the model
goes on to predict, PCA the class centroids, top-k). If that subspace is
readout-adjacent w.r.t. M_L while the concept estimators are not, the test
discriminates function at matched depth rather than depth itself.

Outputs one npz per model with:
    layer, n_layers, D, n_tokens
    readout_L_top100   (100, D)  right singular vectors of M_L (rows)
    wu_top100          (100, D)  right singular vectors of W_U (identity translator)
    nexttok_pca_L      (10, D)   depth-matched positive control at layer L
    nexttok_pca_last   (10, D)   same construction at the last block (sanity)
    ridge_r2, ridge_cos, ridge_lambda, n_classes

Usage:
    python analysis/depth_matched_readout.py --model Qwen/Qwen2.5-7B \
        --slug qwen2.5-7b --layer 13 --corpus analysis/corpus_wikitext.txt \
        --out analysis/depth_matched/qwen2.5-7b.npz
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.extractor import RepresentationExtractor  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def find_final_norm(model: torch.nn.Module):
    """Return (weight, is_layernorm) of the final normalisation before lm_head."""
    cands = []
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in {"norm", "ln_f", "norm_f", "final_layernorm", "final_norm"} and hasattr(mod, "weight"):
            cands.append((name, mod))
    if not cands:
        return None, False
    name, mod = cands[-1]
    is_ln = "rms" not in type(mod).__name__.lower()
    print(f"  final norm: {name} ({type(mod).__name__}, layernorm={is_ln})")
    return mod.weight.detach().float(), is_ln


def top_eigvecs(S: torch.Tensor, k: int) -> np.ndarray:
    """Top-k eigenvectors (rows) of a symmetric PSD matrix.

    cuSOLVER's eigh can fail to converge on ill-conditioned Gram matrices
    (seen on gpt-oss-20b); fall back to LAPACK on the CPU, then to SVD.
    """
    S = ((S + S.T) / 2).double()
    try:
        evals, evecs = torch.linalg.eigh(S)
        idx = torch.argsort(evals, descending=True)[:k]
        return evecs[:, idx].T.float().cpu().numpy()
    except Exception as e:  # noqa: BLE001
        print(f"  [eigh fallback] {type(e).__name__}: using CPU LAPACK")
        Sn = S.cpu().numpy()
        try:
            evals, evecs = np.linalg.eigh(Sn)
            idx = np.argsort(evals)[::-1][:k]
            return evecs[:, idx].T.astype(np.float32)
        except Exception:
            U, _, _ = np.linalg.svd(Sn, full_matrices=False)
            return U[:, :k].T.astype(np.float32)


def centroid_pca(H: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    classes = np.unique(labels)
    C = np.stack([H[labels == c].mean(0) for c in classes]).astype(np.float64)
    C -= C.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt[:k].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--layer", type=int, required=True, help="FARS layer (block index)")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "analysis/corpus_wikitext.txt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--k-out", type=int, default=100)
    ap.add_argument("--k-pos", type=int, default=10)
    ap.add_argument("--min-class", type=int, default=20)
    ap.add_argument("--max-classes", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()

    ex = RepresentationExtractor(args.model, device=args.device, dtype=DTYPES[args.dtype])
    ex.load()
    model, tok = ex._model, ex._tokenizer
    model.config.use_cache = False
    layers = ex._get_layer_modules()
    n_layers = len(layers)
    L, last = args.layer, n_layers - 1
    assert 0 <= L < n_layers, (L, n_layers)
    dev0 = model.device if args.device == "auto" else torch.device(args.device)
    print(f"[{args.slug}] layer {L}/{last}  device {dev0}")

    # ---- corpus -> fixed-length chunks
    text = open(args.corpus).read()
    ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    T = args.seq_len
    n_chunks = min(len(ids) // T, max(1, args.max_tokens // T))
    chunks = torch.tensor([ids[i * T:(i + 1) * T] for i in range(n_chunks)])
    print(f"  corpus: {len(ids)} tokens -> {n_chunks} chunks x {T}")

    # ---- hooks on block L and last block (same convention as the extractor)
    cap = {}

    def mk(idx):
        def hook(m, i, o):
            cap[idx] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    handles = [layers[L].register_forward_hook(mk("L")), layers[last].register_forward_hook(mk("last"))]

    HL, HLast, PRED = [], [], []
    with torch.no_grad():
        for b in range(0, n_chunks, args.batch):
            x = chunks[b:b + args.batch].to(dev0)
            out = model(input_ids=x, use_cache=False)
            hL = cap["L"][:, 1:, :].reshape(-1, cap["L"].shape[-1])       # drop position 0 (BOS / sink)
            hLast = cap["last"][:, 1:, :].reshape(-1, cap["last"].shape[-1])
            pred = out.logits[:, 1:, :].argmax(-1).reshape(-1)
            # float32 on the CPU: bf16 residual norms can exceed the float16 range (gpt-oss-20b)
            HL.append(hL.float().cpu()); HLast.append(hLast.float().cpu())
            PRED.append(pred.cpu())
            del out
    for h in handles:
        h.remove()
    HL = torch.cat(HL); HLast = torch.cat(HLast); PRED = torch.cat(PRED).numpy()
    N, D = HL.shape
    print(f"  collected {N} token states, D={D}  ({time.time()-t0:.0f}s)")

    # ---- ridge translator h_last ~= A h_L + b, lambda chosen on a held-out split
    n_ho = int(N * args.holdout_frac)
    Xtr, Ytr = HL[:-n_ho].float().to(dev0), HLast[:-n_ho].float().to(dev0)
    Xho, Yho = HL[-n_ho:].float().to(dev0), HLast[-n_ho:].float().to(dev0)
    mx, my = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
    Xc, Yc = Xtr - mx, Ytr - my
    Sxx = Xc.T @ Xc; Sxy = Xc.T @ Yc
    del Xtr, Ytr, Xc, Yc
    tr = torch.trace(Sxx) / D
    best = None
    for rel in (1e-4, 1e-3, 1e-2, 1e-1):
        A = torch.linalg.solve(Sxx + rel * tr * torch.eye(D, device=dev0), Sxy)  # (D, D): y = x A
        Yhat = (Xho - mx) @ A + my
        r2 = 1 - ((Yho - Yhat) ** 2).sum() / ((Yho - Yho.mean(0)) ** 2).sum()
        cos = torch.nn.functional.cosine_similarity(Yhat, Yho, dim=1).mean()
        print(f"  ridge rel-lambda={rel:.0e}: holdout R2={r2.item():.3f} cos={cos.item():.3f}")
        if best is None or r2 > best[0]:
            best = (r2.item(), cos.item(), rel, A)
    r2, cos, lam, A = best
    del Sxx, Sxy, Xho, Yho

    # ---- readout channels at layer L: right singular vectors of M = W_U Γ P A^T
    # (A maps row-vectors: h_last = h_L A + b, so as a column map it is A^T)
    gamma, is_ln = find_final_norm(model)
    if gamma is None:
        print("  WARNING: no final norm found, using gamma = 1")
        gamma = torch.ones(D)
    gamma = gamma.to(dev0)
    W = model.get_output_embeddings().weight.detach()            # (V, D)
    G = torch.zeros(D, D, device=dev0)
    for s in range(0, W.shape[0], 16384):
        w = W[s:s + 16384].float().to(dev0)
        G += w.T @ w
    Gg = gamma[:, None] * G * gamma[None, :]                      # Γ G Γ
    if is_ln:
        P = torch.eye(D, device=dev0) - torch.full((D, D), 1.0 / D, device=dev0)
        Gg = P @ Gg @ P
    At = A.T                                                      # column map
    MtM = At.T @ Gg @ At                                          # (D, D)
    readout_L = top_eigvecs(MtM, args.k_out)
    wu_top = top_eigvecs(G, args.k_out)
    del G, Gg, MtM, At, A, W

    # ---- depth-matched positive control: next-token-identity centroid PCA at layer L
    vals, counts = np.unique(PRED, return_counts=True)
    keep = vals[counts >= args.min_class]
    keep = keep[np.argsort(-counts[counts >= args.min_class])][:args.max_classes]
    mask = np.isin(PRED, keep)
    HLn, HLastn = HL.numpy(), HLast.numpy()
    nt_L = centroid_pca(HLn[mask], PRED[mask], args.k_pos)
    nt_last = centroid_pca(HLastn[mask], PRED[mask], args.k_pos)
    print(f"  next-token classes: {len(keep)} (>= {args.min_class} tokens each), {mask.sum()} tokens")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, layer=L, n_layers=n_layers, D=D, n_tokens=N,
             readout_L_top100=readout_L, wu_top100=wu_top,
             nexttok_pca_L=nt_L, nexttok_pca_last=nt_last,
             ridge_r2=r2, ridge_cos=cos, ridge_lambda=lam, n_classes=len(keep),
             layernorm=is_ln)
    print(f"[done] {args.out}  R2={r2:.3f} cos={cos:.3f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
