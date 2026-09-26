# FARS research snapshot — 2026-09-26

Code and recorded outputs for the FARS preprint. This is a curated snapshot of the local supplementary material, with the fourth cross-concept patching result added. Recorded results have not been independently rerun for this release.

## Quick start: geometry only

```bash
cd fars  # from repository root
python -m pip install numpy
python -m unittest test_readout_geometry -v
python readout_geometry.py --basis B.npy --readout-basis V10.npy
# Alternative: --unembedding W_U.npy
```

B and V10 contain orthonormal basis rows of equal rank and ambient dimension. W_U is vocabulary-by-hidden-dimension. The reference implementation reports principal angles, Grassmann distance, mean readout energy, and the Haar expectation. Full SVD of W_U can be expensive. These tests use analytic toy matrices, not pretrained models.

## Research pipeline

Run commands from this directory, with this directory on PYTHONPATH:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python analysis/extract_activations.py --model openai-community/gpt2-xl --slug gpt2-xl --out-root results
python analysis/build_fars_basis.py --acts results/gpt2-xl/activations_cache.npz --out-dir results/gpt2-xl
# Substitute the actual layer printed by the preceding command:
python analysis/readout_rank_sweep.py --model openai-community/gpt2-xl --slug gpt2-xl --acts results/gpt2-xl/activations_cache.npz --fars-basis results/gpt2-xl/fars_basis_L<LAYER>.npz --out analysis/new_readout_gpt2-xl.json
```

Replace `<LAYER>` before running the last command. Extraction needs model downloads and sufficient memory. Some checkpoints require accepting their provider's access terms. This is an example pipeline, not a command claiming to reproduce every reported layer: the supplied basis builder selects centroid variance in a depth band, whereas some paper analyses select concept RSA. Record the selector for every run.

## Contents

- `benchmark/`: TriForm (324 stimuli) and disjoint Novelty (180 stimuli) generators.
- `src/`: extraction, subspace and statistical utilities.
- `analysis/`: experimental scripts and recorded JSON outputs (readout, depth controls, published directions, CoT, patching).
- `results/`: selected historical X-FARS and behavioral outputs.
- `run_xfars.py`: historical heuristic, with an explicit optimizer caveat.
- `readout_geometry.py`, `test_readout_geometry.py`: independent geometry reference and tests.
- `MANIFEST.json`: SHA-256 hashes for this curated snapshot.

## Reproduction boundaries

Activation caches and model weights are excluded. External AdvBench/benign-instruction inputs and the translator's generic-text corpus are not included. Historical scripts have differing default output roots; use explicit paths and inspect their CLI before scheduling. Requirements are lower bounds, not a locked environment. This snapshot does not promise all tables can be regenerated without further provenance work. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).
