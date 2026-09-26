#!/usr/bin/env python3
"""Published concept directions under the readout-orthogonality test.

The body's verdict covers subspaces we extract. This script builds two
directions the literature actually uses, on the same model and at the same
block as FARS, so they can be run through the identical test:

  refusal  mean(harmful prompts) - mean(harmless prompts)   (Arditi et al. 2024)
  truth    mean(true statements) - mean(false statements)   (Marks & Tegmark 2024)

Both are difference-of-means directions at the last input token. We also
record a held-out linear-separability check (AUC of the 1-d projection) so a
direction that fails to separate its classes is not over-interpreted.

Output npz per model:
    layer, D, refusal_dir (D,), truth_dir (D,), refusal_auc, truth_auc,
    n_harmful, n_harmless, n_true, n_false,
    refusal_dir_last (D,), truth_dir_last (D,)   -- same at the final block

The comparison against W_U and the pulled-back readout is done offline by
published_subspaces_analysis.py using depth_matched/<slug>.npz.

Usage:
    python analysis/published_subspaces.py --model meta-llama/Llama-3.1-8B-Instruct \
        --slug llama-3.1-8b-instruct --layer 8 --out analysis/published_dirs/llama-3.1-8b-instruct.npz
"""
from __future__ import annotations
import argparse, csv, os, random, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.extractor import RepresentationExtractor  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# (city, country) facts for true/false statements, in the style of the
# "cities" set of Marks & Tegmark (2024).
CITIES = [
    ("Paris", "France"), ("Berlin", "Germany"), ("Madrid", "Spain"), ("Rome", "Italy"),
    ("Lisbon", "Portugal"), ("Vienna", "Austria"), ("Warsaw", "Poland"), ("Prague", "the Czech Republic"),
    ("Athens", "Greece"), ("Dublin", "Ireland"), ("Oslo", "Norway"), ("Stockholm", "Sweden"),
    ("Helsinki", "Finland"), ("Copenhagen", "Denmark"), ("Amsterdam", "the Netherlands"),
    ("Brussels", "Belgium"), ("Bern", "Switzerland"), ("Budapest", "Hungary"), ("Bucharest", "Romania"),
    ("Sofia", "Bulgaria"), ("Ankara", "Turkey"), ("Cairo", "Egypt"), ("Nairobi", "Kenya"),
    ("Lagos", "Nigeria"), ("Johannesburg", "South Africa"), ("Tokyo", "Japan"), ("Seoul", "South Korea"),
    ("Beijing", "China"), ("Bangkok", "Thailand"), ("Hanoi", "Vietnam"), ("Jakarta", "Indonesia"),
    ("Manila", "the Philippines"), ("New Delhi", "India"), ("Islamabad", "Pakistan"), ("Tehran", "Iran"),
    ("Riyadh", "Saudi Arabia"), ("Ottawa", "Canada"), ("Mexico City", "Mexico"), ("Havana", "Cuba"),
    ("Lima", "Peru"), ("Santiago", "Chile"), ("Buenos Aires", "Argentina"), ("Brasilia", "Brazil"),
    ("Bogota", "Colombia"), ("Caracas", "Venezuela"), ("Canberra", "Australia"), ("Wellington", "New Zealand"),
    ("Moscow", "Russia"), ("Kyiv", "Ukraine"), ("Toronto", "Canada"),
]


def truth_statements(seed: int = 0):
    rng = random.Random(seed)
    countries = [c for _, c in CITIES]
    true_s, false_s = [], []
    for city, country in CITIES:
        true_s.append(f"The city of {city} is in {country}.")
        true_s.append(f"{city} is located in {country}.")
        wrong = rng.choice([c for c in countries if c != country])
        false_s.append(f"The city of {city} is in {wrong}.")
        wrong2 = rng.choice([c for c in countries if c not in (country, wrong)])
        false_s.append(f"{city} is located in {wrong2}.")
    return true_s, false_s


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Mann-Whitney AUC of a 1-d projection."""
    from itertools import product
    s = sum((p > n) + 0.5 * (p == n) for p, n in product(scores_pos, scores_neg))
    return float(s / (len(scores_pos) * len(scores_neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--layer", type=int, required=True, help="block index of the FARS layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--harmful", default=os.path.join(ROOT, "analysis/data/advbench_harmful_behaviors.csv"))
    ap.add_argument("--harmless", default=os.path.join(ROOT, "analysis/data/harmless_alpaca.txt"))
    args = ap.parse_args()
    t0 = time.time()

    with open(args.harmful) as f:
        harmful = [r["goal"].strip() for r in csv.DictReader(f)][: args.n_per_class]
    harmless = [l.strip() for l in open(args.harmless) if l.strip()][: args.n_per_class]
    true_s, false_s = truth_statements()
    print(f"[data] harmful {len(harmful)}  harmless {len(harmless)}  true {len(true_s)}  false {len(false_s)}")

    ex = RepresentationExtractor(args.model, device=args.device, dtype=DTYPES[args.dtype])
    ex.load()
    model, tok = ex._model, ex._tokenizer
    layers = ex._get_layer_modules(); L, last = args.layer, len(layers) - 1
    dev0 = model.device if args.device == "auto" else torch.device(args.device)
    chat = getattr(tok, "chat_template", None) is not None

    cap = {}
    def mk(k):
        def hook(m, i, o):
            cap[k] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    hs = [layers[L].register_forward_hook(mk("L")), layers[last].register_forward_hook(mk("last"))]

    def embed(texts, as_instruction):
        outL, outLast = [], []
        with torch.no_grad():
            for t in texts:
                if as_instruction and chat:
                    ids = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                                  return_tensors="pt")
                    # BatchEncoding subclasses UserDict, not dict, so isinstance(_, dict) misses it.
                    if hasattr(ids, "input_ids"):
                        ids = ids.input_ids
                    elif not torch.is_tensor(ids):
                        ids = ids["input_ids"]
                else:
                    ids = tok(t, return_tensors="pt")["input_ids"]
                model(input_ids=ids.to(dev0), use_cache=False)
                outL.append(cap["L"][0, -1].float().cpu().numpy())
                outLast.append(cap["last"][0, -1].float().cpu().numpy())
        return np.stack(outL), np.stack(outLast)

    H_L, H_last = embed(harmful, True); B_L, B_last = embed(harmless, True)
    T_L, T_last = embed(true_s, False); F_L, F_last = embed(false_s, False)
    for h in hs:
        h.remove()
    print(f"  embedded {len(harmful)+len(harmless)+len(true_s)+len(false_s)} texts ({time.time()-t0:.0f}s)")

    def diffmean(A, B, ntr):
        d = A[:ntr].mean(0) - B[:ntr].mean(0)
        d = d / (np.linalg.norm(d) + 1e-12)
        return d, auc(A[ntr:] @ d, B[ntr:] @ d)

    n_h = min(len(H_L), len(B_L)) // 2; n_t = min(len(T_L), len(F_L)) // 2
    r_dir, r_auc = diffmean(H_L, B_L, n_h)
    t_dir, t_auc = diffmean(T_L, F_L, n_t)
    # Record the final-block AUCs too: the depth contrast is only meaningful if the
    # final-block direction still separates its classes, rather than being a
    # degenerate vector that happens to lie near the readout span.
    r_last, r_auc_last = diffmean(H_last, B_last, n_h)
    t_last, t_auc_last = diffmean(T_last, F_last, n_t)
    print(f"  refusal AUC (held-out) {r_auc:.3f} | truth AUC {t_auc:.3f} "
          f"| final block: refusal {r_auc_last:.3f} truth {t_auc_last:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, layer=L, n_layers=len(layers), D=H_L.shape[1],
             refusal_dir=r_dir.astype(np.float32), truth_dir=t_dir.astype(np.float32),
             refusal_dir_last=r_last.astype(np.float32), truth_dir_last=t_last.astype(np.float32),
             refusal_auc=r_auc, truth_auc=t_auc,
             refusal_auc_last=r_auc_last, truth_auc_last=t_auc_last,
             n_harmful=len(harmful), n_harmless=len(harmless), n_true=len(true_s), n_false=len(false_s))
    print(f"[done] {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
