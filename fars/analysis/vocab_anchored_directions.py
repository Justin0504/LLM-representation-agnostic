#!/usr/bin/env python3
"""A published-style concept construction that should be a READOUT channel.

Every estimator the paper tests so far is built from *activations*, and every
one lands on the locus side. A test whose verdict never changes across the
constructions people actually use is hard to act on. This script builds the
other kind of "concept direction" the literature also uses -- one defined in
the model's *vocabulary*, not in its activations:

    for concept c, take the tokens that are diagnostic of c's stimuli
    (tf-idf over the concept's six surface forms against the other concepts),
    average the corresponding rows of W_U, and PCA the 18 resulting centroids
    into a rank-k subspace.

This is the construction behind token-anchored steering and logit-attribution
readings of "the direction of concept c". It needs no forward pass: it is a
function of W_U and the tokenizer alone. If the readout-orthogonality test is
discriminating rather than unanimous, this subspace should land on the readout
side while the activation-derived estimators stay on the locus side.

Output: analysis/vocab_anchored_<slug>.json with the same energy/d_G fields
used elsewhere, so it can be dropped straight into the nine-way table.

Usage:
    python analysis/vocab_anchored_directions.py --model openai-community/gpt2-xl \
        --slug gpt2-xl --out analysis/vocab_anchored_gpt2-xl.json
"""
from __future__ import annotations
import argparse, json, math, os, sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from benchmark.generate import generate_all_stimuli  # noqa: E402
from method_comparison_8way import grassmann, energy_in, _top_k_pca  # noqa: E402

K = 10


def concept_token_weights(tok, stim, n_concepts: int, top_n: int = 40):
    """tf-idf token weights per concept over its own stimuli."""
    docs = [Counter() for _ in range(n_concepts)]
    for s in stim:
        for t in tok(s.text, add_special_tokens=False)["input_ids"]:
            docs[s.concept_id][t] += 1
    df = Counter()
    for d in docs:
        for t in d:
            df[t] += 1
    out = []
    for c, d in enumerate(docs):
        total = sum(d.values()) or 1
        scores = {t: (n / total) * math.log(n_concepts / df[t]) for t, n in d.items() if df[t] < n_concepts}
        keep = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
        out.append(keep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--readout", default=None, help="readout_top10_<slug>.npz; default inferred")
    ap.add_argument("--fars-basis", default=None)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()
    k = args.k

    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from readout_rank_sweep import load_W_U

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stim = generate_all_stimuli()
    n_concepts = int(max(s.concept_id for s in stim)) + 1
    W = load_W_U(args.model).astype(np.float32)        # (V, D)
    V, D = W.shape
    print(f"[{args.slug}] W_U {W.shape}, {n_concepts} concepts")

    weights = concept_token_weights(tok, stim, n_concepts, args.top_n)
    C = np.zeros((n_concepts, D), dtype=np.float64)
    kept = []
    for c, kw in enumerate(weights):
        kw = [(t, w) for t, w in kw if 0 <= t < V]
        kept.append(len(kw))
        if not kw:
            continue
        idx = np.array([t for t, _ in kw]); w = np.array([w for _, w in kw], dtype=np.float64)
        C[c] = (W[idx].astype(np.float64) * w[:, None]).sum(0) / w.sum()
    print(f"  diagnostic tokens per concept: min {min(kept)}, median {int(np.median(kept))}")

    B_vocab = _top_k_pca(C, k)                          # (k, D) vocabulary-anchored subspace

    readout = args.readout or os.path.join(ROOT, f"analysis/readout_top10_{args.slug}.npz")
    rz = np.load(readout, allow_pickle=True)
    B_read = rz["basis" if "basis" in rz.files else "top_k_readout"].astype(np.float64)

    res = {"model": args.slug, "D": D, "V": V, "K": k,
           "ceiling": float(np.sqrt(k) * np.pi / 2),
           "haar_null_energy_pct": 100.0 * k / D,
           "tokens_per_concept_median": int(np.median(kept))}
    dG, ang = grassmann(B_vocab, B_read)
    res["VocabAnchored"] = {"d_G_vs_readout": dG, "energy_in_topk_readout_pct": 100 * energy_in(B_vocab, B_read),
                            "min_principal_angle_deg": float(ang.min())}

    fars_f = args.fars_basis
    if fars_f is None:
        import glob
        g = glob.glob(os.path.join(ROOT, f"results/{args.slug}/fars_basis_L*.npz"))
        fars_f = g[0] if g else None
    if fars_f:
        B_fars = np.load(fars_f, allow_pickle=True)["basis"].astype(np.float64)
        dGf, angf = grassmann(B_fars, B_read)
        res["FARS"] = {"d_G_vs_readout": dGf, "energy_in_topk_readout_pct": 100 * energy_in(B_fars, B_read),
                       "min_principal_angle_deg": float(angf.min())}
        # is the vocabulary subspace even about the same concepts? overlap with FARS
        res["vocab_vs_fars"] = {"d_G": grassmann(B_vocab, B_fars)[0],
                                "energy_pct": 100 * energy_in(B_vocab, B_fars)}

    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    res["Random"] = {"energy_in_topk_readout_pct": 100 * energy_in(Q.T, B_read)}

    # Circularity control: the vocabulary-anchored subspace is built from rows of
    # W_U, so some readout loading is guaranteed. The question is whether the
    # concept-diagnostic token choice matters. Rebuild the same construction from
    # random token sets of the same size.
    rngt = np.random.default_rng(7); rand_energies = []
    n_tok = int(np.median(kept))
    for _ in range(20):
        Cr = np.stack([W[rngt.choice(V, n_tok, replace=False)].astype(np.float64).mean(0)
                       for _ in range(n_concepts)])
        rand_energies.append(100 * energy_in(_top_k_pca(Cr, k), B_read))
    res["VocabRandomTokens"] = {"energy_in_topk_readout_pct": float(np.mean(rand_energies)),
                                "std": float(np.std(rand_energies)), "n_draws": 20,
                                "tokens_per_class": n_tok}

    e = lambda n: res[n]["energy_in_topk_readout_pct"]
    print(f"  vocabulary-anchored {e('VocabAnchored'):6.2f}%   FARS {e('FARS'):5.2f}%   "
          f"Haar {res['Random']['energy_in_topk_readout_pct']:.2f}%   (k/D = {res['haar_null_energy_pct']:.2f}%)")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
