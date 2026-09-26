# Known limitations and expansion priorities

## X-FARS optimizer

The historical `solve_basis` uses the polar factor of C.T @ target. It maximizes a trace objective, but generally does NOT minimize ||C B.T - target||^2 under orthonormal rows: the quadratic term is not constant. The outer iteration also normalizes projections. Historical alignment scores must not be presented as evidence of a globally optimal solution to that least-squares criterion. We preserve the numerical implementation for audit, and correct its misleading documentation. Establish a consistent objective, compare to fixed-PCA square Procrustes, and rerun held-out evaluation before strengthening claims.

## Data and claims

- Original and regenerated CoT runs use different layers and generation statistics. Do not combine them as a single run.
- Layer selectors vary across scripts; the supplied basis builder uses top-k centroid variance, not concept RSA.
- Cross-concept patching JSONs now cover four models, each with 40 pairs: GPT-2 XL moved 5%; SmolLM3-3B, Qwen2.5-3B-Instruct and Mamba-2.8B moved 0%. These results do not establish general causal sufficiency.
- A low rank-10 readout overlap does not establish absence of all readout information or causal computation.
- Recorded aggregate outputs are not a fresh reproduction. The old supplementary claim of less than 50 A100-hours was not verified and is omitted.
- The inherited repository has no detected license. No new license is imposed by this release.

## Next experiments

1. Resolve X-FARS objective and verify historical versus corrected outputs on cached activations; use held-out concepts/models and random controls.
2. Expand the four-model patching test across layers, intervention strengths, seeds and more pairs. Include full-vector and same-norm random positive/negative controls and confidence intervals.
3. Compare CoT budgets 512/1024/2048 with matched token positions and lengths; retain raw generations and distinguish surface-form collapse from reasoning changes.
4. Check completion of existing large-model cluster jobs before submitting duplicates. Broaden model count only after these controls.

Delta job status was not accessible during release preparation because interactive Duo authentication was required. No new cluster job was submitted for this release.
