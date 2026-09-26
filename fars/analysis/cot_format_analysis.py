#!/usr/bin/env python3
"""Format-collapse control for the CoT-tail subspace.

Reads the traces + CoT-tail activations written by cot_format_control.py and
answers the confound directly: is the CoT-tail's cross-format retrieval carried
by genuinely different surface forms, or do the generated traces all collapse to
one form (typically English prose) so that "cross-format" is really within-format?

Two measurements:

  1. Collapse rate. Classify every generated trace's surface form from its own
     text and cross-tabulate against the prompt's form. If a Chinese prompt
     reliably yields an English trace, the prompt-form label is not the label
     that matters at this position.

  2. The control that settles it. The cross-format score X in
     cot_fars_retrieval.py groups stimuli by the *prompt's* form. We recompute it
     grouping by the *generated trace's* form. Scoring a stimulus against
     centroids built from traces in other surface forms is the honest version of
     the claim; if the traces have collapsed, the recomputed X is the number the
     paper should report.

Usage:
    python analysis/cot_format_analysis.py --slug r1-distill-llama-8b
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from cot_fars_retrieval import fars_at, best_layer  # same extraction as the paper
K = 10

CJK = re.compile(r"[一-鿿]")
FR_ACCENT = re.compile(r"[àâçéèêëîïôûùüÿœ]", re.I)
FR_WORDS = re.compile(r"\b(le|la|les|des|une|est|dans|pour|avec|nous|donc|alors|si)\b", re.I)
EN_WORDS = re.compile(r"\b(the|and|is|are|of|to|that|this|we|if|then|so|therefore|because)\b", re.I)
CODE = re.compile(r"\b(def|return|import|elif|lambda|print|for\s+\w+\s+in|range)\b|[=<>!]=|\bself\b")
MATH = re.compile(r"[∀∃⊢⇒→≤≥≠∈∧∨¬±×÷∑∏√≡]|\\\\(?:frac|forall|exists|implies|neg|land|lor)|\bmod\b")
LIST = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")


def classify(text: str) -> str:
    """Dominant surface form of a generated trace."""
    t = text.strip()
    if not t:
        return "empty"
    n = max(len(t), 1)
    if len(CJK.findall(t)) / n > 0.05:
        return "chinese"
    scores = {
        "code": 2.0 * len(CODE.findall(t)),
        "math": 2.0 * len(MATH.findall(t)),
        "list": 3.0 * len(LIST.findall(t)),
        "french": 1.5 * (len(FR_ACCENT.findall(t)) + len(FR_WORDS.findall(t))),
        "english": 1.0 * len(EN_WORDS.findall(t)),
    }
    top = max(scores, key=scores.get)
    return top if scores[top] > 0 else "english"


def retrieval_X(proj: np.ndarray, concept_ids: np.ndarray, group_ids: np.ndarray) -> dict:
    """Agnostic top-1 and the cross-group score X, grouping by `group_ids`."""
    n_c = int(concept_ids.max()) + 1
    groups = sorted(set(group_ids.tolist()))
    C_all = np.stack([proj[concept_ids == c].mean(0) for c in range(n_c)])
    agn = float((np.linalg.norm(proj[:, None] - C_all[None], axis=-1).argmin(1) == concept_ids).mean())

    C_by = {}
    for g in groups:
        C_by[g] = np.stack([proj[(group_ids == g) & (concept_ids == c)].mean(0)
                            if ((group_ids == g) & (concept_ids == c)).any() else np.full(proj.shape[1], np.inf)
                            for c in range(n_c)])
    X_hit, X_n, W_hit, W_n = 0, 0, 0, 0
    for i in range(len(proj)):
        others = [g for g in groups if g != group_ids[i]]
        if others:
            C_other = np.nanmean(np.stack([np.where(np.isinf(C_by[g]), np.nan, C_by[g]) for g in others]), axis=0)
            ok = ~np.isnan(C_other).any(1)
            if ok.sum() >= 2:
                d = np.linalg.norm(proj[i] - C_other[ok], axis=1)
                X_hit += int(np.arange(n_c)[ok][d.argmin()] == concept_ids[i]); X_n += 1
        Cw = C_by[group_ids[i]]
        ok = ~np.isinf(Cw).any(1)
        if ok.sum() >= 2:
            d = np.linalg.norm(proj[i] - Cw[ok], axis=1)
            W_hit += int(np.arange(n_c)[ok][d.argmin()] == concept_ids[i]); W_n += 1
    return {"agn": agn, "X": X_hit / X_n if X_n else float("nan"), "n_X": X_n,
            "W": W_hit / W_n if W_n else float("nan"), "n_W": W_n,
            "n_groups": len(groups)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "results_v2_cot"))
    ap.add_argument("--layer", type=int, default=-1, help="block index; default = best FARS layer file")
    args = ap.parse_args()

    d = os.path.join(args.root, args.slug)
    traces = json.load(open(os.path.join(d, "cot_traces.json")))["traces"]
    print(f"[{args.slug}] {len(traces)} traces, "
          f"{sum(t['truncated'] for t in traces)} truncated "
          f"({100*sum(t['truncated'] for t in traces)/len(traces):.0f}%)")

    prompt_form = [t["form"] for t in traces]
    trace_form = [classify(t["generated"]) for t in traces]

    print("\n  prompt form -> generated-trace form")
    forms = sorted(set(prompt_form))
    collapse_num = collapse_den = 0
    for pf in forms:
        idx = [i for i, f in enumerate(prompt_form) if f == pf]
        c = Counter(trace_form[i] for i in idx)
        top, n_top = c.most_common(1)[0]
        print(f"    {pf:<14s} n={len(idx):<4d} -> {dict(c)}")
        if pf != "en_prose":   # benchmark form names: en_prose, zh_prose, fr_prose, py_code, math_notation, structured_list
            collapse_den += len(idx)
            collapse_num += sum(1 for i in idx if trace_form[i] == "english")
    rate = 100 * collapse_num / collapse_den if collapse_den else float("nan")
    print(f"\n  collapse rate (non-English prompt -> English trace): {rate:.0f}%"
          f"  [{collapse_num}/{collapse_den}]")
    print(f"  distinct trace forms: {len(set(trace_form))}  {Counter(trace_form)}")

    res = {"slug": args.slug, "n": len(traces),
           "truncated_pct": 100 * sum(t["truncated"] for t in traces) / len(traces),
           "collapse_pct": rate, "n_trace_forms": len(set(trace_form)),
           "trace_form_counts": dict(Counter(trace_form)),
           "prompt_form_counts": dict(Counter(prompt_form))}

    # The control: regroup the cross-format metric by the generated trace's form.
    npz = os.path.join(d, "cot_tail_acts.npz")
    if os.path.exists(npz):
        A = np.load(npz)["activations"].astype(np.float64)   # (N, L, D)
        cids = np.array([t["concept_id"] for t in traces])
        # The paper's CoT-tail retrieval uses a FARS extracted *from the CoT-tail
        # activations themselves* at their own best mid-band layer, not the
        # last-input-token basis. Rebuild it the same way (cot_fars_retrieval.py)
        # so this control tests the claim the paper actually makes.
        L, _, ve = best_layer(A, cids, k=K)
        B = fars_at(A[:, L, :], cids, k=K)
        proj = A[:, L, :] @ B.T
        proj = proj - proj.mean(0, keepdims=True)
        by_prompt = retrieval_X(proj, cids, np.array(prompt_form))
        by_trace = retrieval_X(proj, cids, np.array(trace_form))
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((A.shape[2], K)))
        projr = A[:, L, :] @ Q
        projr = projr - projr.mean(0, keepdims=True)
        rand = retrieval_X(projr, cids, np.array(trace_form))
        print(f"\n  CoT-tail FARS at layer {L} (var explained {ve:.2f}); chance {100/ (cids.max()+1):.1f}%")
        print(f"  grouped by PROMPT form : Agn {by_prompt['agn']*100:.1f}%  "
              f"X {by_prompt['X']*100:.1f}% (n={by_prompt['n_X']}, {by_prompt['n_groups']} groups)")
        print(f"  grouped by TRACE  form : Agn {by_trace['agn']*100:.1f}%  "
              f"X {by_trace['X']*100:.1f}% (n={by_trace['n_X']}, {by_trace['n_groups']} groups)")
        print(f"  Haar rank-{K} control   : Agn {rand['agn']*100:.1f}%  X {rand['X']*100:.1f}%")
        res.update({"cot_layer": L, "cot_var_explained": ve, "chance": 1.0 / (int(cids.max()) + 1),
                    "by_prompt_form": by_prompt, "by_trace_form": by_trace, "haar_by_trace_form": rand})
    else:
        print("  [skip control] no cot_tail_acts.npz")

    out = os.path.join(ROOT, f"analysis/cot_format_control_{args.slug}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
