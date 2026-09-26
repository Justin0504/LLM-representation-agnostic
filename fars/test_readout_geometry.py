"""Analytic checks of the reference diagnostic; no reported experiments rerun."""
import unittest
import numpy as np
from readout_geometry import compare, raw_readout_basis

class GeometryChecks(unittest.TestCase):
    def test_identical_and_disjoint(self):
        b = np.eye(20)[:4]
        same = compare(b, b)
        self.assertAlmostEqual(same['energy_fraction'], 1)
        self.assertAlmostEqual(same['grassmann_distance'], 0)
        apart = compare(b, np.eye(20)[4:8])
        self.assertAlmostEqual(apart['energy_fraction'], 0)
        self.assertAlmostEqual(apart['grassmann_distance'], np.pi)

    def test_known_angles_and_rotations(self):
        a = np.array([.2, .7, 1.0, 1.4])
        b = np.eye(8)[:4]
        v = np.diag(np.cos(a)) @ b + np.diag(np.sin(a)) @ np.eye(8)[4:]
        got = compare(b, v)
        np.testing.assert_allclose(got['principal_angles_radians'], a)
        self.assertAlmostEqual(got['energy_fraction'], np.mean(np.cos(a)**2))
        q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(4,4)))
        self.assertAlmostEqual(compare(q @ b, v)['grassmann_distance'], got['grassmann_distance'])

    def test_unembedding_orientation(self):
        w = np.diag([1., 4., 2., 3.])
        v = raw_readout_basis(w, 2)
        np.testing.assert_allclose(v.T @ v, np.diag([0.,1.,0.,1.]))

    def test_invalid_basis_rejected(self):
        with self.assertRaises(ValueError):compare(np.ones((2,4)), np.eye(4)[:2])
        with self.assertRaises(ValueError):compare(np.eye(4)[:2], np.eye(4)[:1])
        with self.assertRaises(ValueError):raw_readout_basis(np.zeros((4,4)), 2)

    def test_empirical_haar_expectation(self):
        rng = np.random.default_rng(7);fixed = np.eye(20)[:4]
        energy=[]
        for _ in range(250):
            q,_=np.linalg.qr(rng.normal(size=(20,4)))
            energy.append(compare(q.T,fixed)['energy_fraction'])
        self.assertLess(abs(np.mean(energy)-4/20),.015)

if __name__ == '__main__':unittest.main()
