# FARS: Concept Subspaces Beyond the Logit Lens

Research code for **Concept Subspaces Compute Beyond the Logit Lens: A Weights-Only Test for Locating Representations Upstream of Readout**.

The current FARS research snapshot is in [`fars/`](fars/README.md). It includes TriForm and Novelty generators, activation extraction, readout geometry and depth controls, recorded result JSONs, and historical X-FARS code. The original pilot and its upstream attribution remain in this repository; see [pilot documentation](PILOT_README.md).

**Status:** public research snapshot, not a certified end-to-end reproduction. See [known limitations](fars/KNOWN_LIMITATIONS.md) before using the results. The raw geometric readout test has a small independent NumPy reference implementation and analytic tests.

```bash
cd fars
python -m pip install numpy
python -m unittest test_readout_geometry -v
```

No model weights, credentials, private cluster configuration, or activation caches are included in the new snapshot. No new license is asserted over inherited code; public visibility alone does not grant an open-source license.
