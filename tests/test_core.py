import unittest

import numpy as np

from geoneuronx.core import (
    AdaptiveAIS,
    amuse,
    effective_rank,
    make_temporal_sources,
    observe_through_morphology,
    path_transfer_matrix,
    recovery_score,
)


class CoreTests(unittest.TestCase):
    def test_equal_lengths_are_redundant(self):
        A = path_transfer_matrix(np.ones(8) * 0.8)
        self.assertEqual(effective_rank(A), 1)

    def test_length_diversity_generates_full_rank_basis(self):
        A = path_transfer_matrix(np.linspace(0.2, 1.4, 8))
        self.assertEqual(effective_rank(A), 3)

    def test_amuse_recovers_temporal_sources_from_wide_morphology(self):
        rng = np.random.default_rng(123)
        s = make_temporal_sources(12_000, rng)
        x, _ = observe_through_morphology(s, np.linspace(0.4, 1.2, 8), rng)
        y = amuse(x, 3, lag=1)
        score, _ = recovery_score(y, s)
        self.assertGreater(score, 0.98)

    def test_ais_moves_threshold_up_when_overactive(self):
        ais = AdaptiveAIS(target_rate=0.1, threshold=1.0, learning_rate=0.02)
        initial = ais.threshold
        for _ in range(1000):
            ais.step(10.0)
        self.assertGreater(ais.threshold, initial)

    def test_ais_moves_threshold_down_when_silent(self):
        ais = AdaptiveAIS(target_rate=0.1, threshold=1.0, learning_rate=0.02)
        initial = ais.threshold
        for _ in range(1000):
            ais.step(-10.0)
        self.assertLess(ais.threshold, initial)


if __name__ == "__main__":
    unittest.main()
