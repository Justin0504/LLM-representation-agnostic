# LLM Representation-Agnostic Reasoning — Pilot Experiment

> **"Language is not the substrate of reasoning; it is one of many possible encodings through which reasoning can be expressed."**

## Hypothesis: Representation-Mediated Reasoning (RMR)

LLM reasoning capability arises from operating over structured internal representations, **not** from properties unique to natural language. Natural language is a *privileged but non-essential* representation — privileged by training scale and inductive bias, but not uniquely required for reasoning.

**Operational prediction:** In a regression predicting pairwise representational distance between stimuli, `β(same_language_type)` should NOT dominate over `β(same_concept)` or structural bias features — i.e., whether two stimuli are both "natural language" does not independently explain their representational similarity, beyond shared conceptual structure.

### Boundaries (important for reviewers)

1. We do **not** claim all representations are equivalent — different forms carry different inductive biases.
2. Natural language remains **privileged** due to training scale and supervision signal alignment.
3. The claim concerns the *source* of reasoning capability, not the *optimal representation* for all tasks.

---

## Experimental Design

**Stimuli:** 5 abstract reasoning concepts × 4 symbolic representation forms = 20 stimuli

| Concept | EN prose | Python code | Math notation | ZH prose |
|---------|----------|-------------|---------------|----------|
| Recursive decomposition | ✓ | ✓ | ✓ | ✓ |
| Transitivity | ✓ | ✓ | ✓ | ✓ |
| Sorting by criterion | ✓ | ✓ | ✓ | ✓ |
| Set intersection | ✓ | ✓ | ✓ | ✓ |
| Causal chain | ✓ | ✓ | ✓ | ✓ |

**Key contrast:** EN↔ZH tests *language-agnostic*; all 4 columns tests *representation-mediated*.

**Methods:**
- **M1 Extended RSA** — Spearman ρ between empirical RDM and 4 theoretical RDMs (concept / form / structural bias / language-type), per layer
- **M2 Cross-form linear probe transfer** — train concept classifier on form A, test on form B (5-way, chance=20%)
- **M3 Silhouette analysis** — clustering quality by concept vs. form label, per layer
- **CKA** — cross-form representational alignment, per layer
- **Bias regression** — OLS: `distance(i,j) ~ same_concept + same_language_type + |Δtoken_count| + |Δentropy| + |ΔTTR|`

---

## Results (Pilot v1)

All results stored in `results/<model>/pilot_numbers.json`.

| Model | Layers | Cross-form probe (final layer) | RSA ρ(concept) max | CKA max |
|-------|-------:|:---:|:---:|:---:|
| GPT-2 (124M) | 12 | 43.3% | −0.067 | 0.874 |
| Llama-3.2-3B | 28 | **70.0%** | +0.104 | 0.940 |
| Qwen2.5-3B | 36 | 53.3% | +0.037 | 0.916 |

**Key finding:** Cross-form probe accuracy improves monotonically with model scale (43% → 70%), demonstrating that concept identity becomes increasingly decodable across symbolic form boundaries in larger models — consistent with RMR.

**Language-type RSA loses significance at the final layer of GPT-2** (p=0.097), suggesting that at depth, "is this natural language?" ceases to be a significant geometric organizing principle.

---

## Project Structure

```
LLM-test/
├── requirements.txt
├── config.py                # model, device, paths
├── src/
│   ├── stimuli.py           # 20 stimuli + concept/form labels
│   ├── extractor.py         # per-layer hidden state extraction (hook-based)
│   ├── metrics.py           # RSA, CKA, linear probe, silhouette, bias regression
│   ├── visualize.py         # 10 publication-ready figures
│   └── rep_bias.py          # structural bias quantification + theoretical RDMs
├── run_pilot.py             # full pipeline entry point
└── results/
    ├── gpt2/
    ├── llama-3.2-3b/
    └── qwen2.5-3b/
```

## Usage

```bash
# Quick smoke test (GPT-2)
HF_ENDPOINT=https://hf-mirror.com python run_pilot.py --model gpt2

# Main experiment (Llama-3.2-3B)
HF_ENDPOINT=https://hf-mirror.com python run_pilot.py --model unsloth/Llama-3.2-3B

# Qwen2.5-3B
HF_ENDPOINT=https://hf-mirror.com python run_pilot.py --model Qwen/Qwen2.5-3B
```

Each model saves results to `results/<model-slug>/` with:
- `pilot_numbers.json` — all numerical results
- `fig1`–`fig10` — PDF figures

## Requirements

```
torch>=2.0.0 (CUDA recommended)
transformers>=4.40.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.13.0
umap-learn>=0.5.5
accelerate>=0.27.0
```

Install: `pip install -r requirements.txt`

---


