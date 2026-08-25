import unittest

import numpy as np

from geoneuronx.population import (
    independent_oja_population,
    mean_pairwise_abs_correlation,
    sanger_population,
    temporal_coincidence_samples,
    whiten_population_input,
)


class PopulationLearningTests(unittest.TestCase):
    def test_temporal_coincidence_has_amuse_eigenvectors(self):
        rng = np.random.default_rng(1)
        n = 8000
        # Two already-separated colored processes mixed by a nontrivial matrix.
        s = np.zeros((n, 2), dtype=float)
        for j, a in enumerate((0.92, -0.55)):
            noise = rng.normal(size=n)
            for t in range(1, n):
                s[t, j] = a * s[t - 1, j] + noise[t]
            s[:, j] = (s[:, j] - s[:, j].mean()) / s[:, j].std()
        x = s @ np.array([[1.0, 0.4], [0.3, 1.0]])
        q, _ = whiten_population_input(x, 2)
        lag = 3
        u = temporal_coincidence_samples(q, lag)

        c_tau = (q[:-lag].T @ q[lag:]) / (len(q) - lag)
        c_tau = 0.5 * (c_tau + c_tau.T)
        cov_u = (u.T @ u) / len(u)

        # For whitened q, Cov[u] = I + C_tau_sym up to finite-sample edge error.
        self.assertTrue(np.allclose(cov_u, np.eye(2) + c_tau, atol=2e-2))

    def test_competition_prevents_oja_population_collapse(self):
        rng = np.random.default_rng(2)
        # Dominant first axis plus a weaker independent second axis.
        x = np.column_stack(
            [3.0 * rng.normal(size=6000), 0.7 * rng.normal(size=6000)]
        )
        oja = independent_oja_population(x, 2, lr=1e-4, epochs=3, seed=2)
        sanger = sanger_population(x, 2, lr=1e-4, epochs=3, seed=2)

        oja_out = x @ oja.T
        sanger_out = x @ sanger.T
        self.assertGreater(mean_pairwise_abs_correlation(oja_out), 0.90)
        self.assertLess(mean_pairwise_abs_correlation(sanger_out), 0.15)


if __name__ == "__main__":
    unittest.main()
