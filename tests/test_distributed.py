import unittest

import numpy as np

from geoneuronx.distributed import (
    fixed_leaf_path_allocation,
    four_channel_probe_groups,
    lagged_feature_matrix,
    simulate_distributed_currents,
)
from geoneuronx.passive_cable import (
    build_passive_tree,
    simulate_passive_tree,
)


class DistributedCableTests(unittest.TestCase):
    def test_fixed_leaf_path_allocation_preserves_material_and_paths(self):
        for bif in (40.0, 80.0, 120.0, 160.0, 210.0):
            lengths = fixed_leaf_path_allocation(bif)
            tree = build_passive_tree(lengths)
            self.assertAlmostEqual(float(np.sum(lengths)), 840.0, places=10)
            self.assertTrue(np.allclose(tree.leaf_path_lengths_um, 360.0))

    def test_distributed_solver_matches_single_drive_solver(self):
        tree = build_passive_tree(fixed_leaf_path_allocation(120.0))
        n = 400
        t = np.arange(n) * 0.5e-3
        drive = 0.03 * np.sin(2.0 * np.pi * 20.0 * t)
        record = tree.branch_probe_nodes

        single = simulate_passive_tree(
            tree,
            drive_nA=drive,
            drive_node=tree.soma_node,
            record_nodes=record,
            dt_ms=0.5,
        )
        currents = np.zeros((n, tree.n_nodes), dtype=float)
        currents[:, tree.soma_node] = drive
        multi = simulate_distributed_currents(
            tree,
            currents,
            record_nodes=record,
            dt_ms=0.5,
        )
        self.assertTrue(np.allclose(single, multi, rtol=1e-11, atol=1e-12))

    def test_probe_groups_are_valid(self):
        tree = build_passive_tree(fixed_leaf_path_allocation(120.0))
        groups = four_channel_probe_groups(tree)
        self.assertEqual(len(groups["distal"]), 4)
        self.assertEqual(len(groups["bifurcation"]), 4)
        self.assertEqual(len(groups["trunk"]), 4)
        self.assertEqual(len(groups["all_midpoints"]), 7)
        self.assertEqual(len(groups["soma"]), 1)
        for nodes in groups.values():
            self.assertTrue(all(0 <= node < tree.n_nodes for node in nodes))

    def test_lagged_feature_matrix(self):
        x = np.arange(200, dtype=float).reshape(100, 2)
        design, max_lag = lagged_feature_matrix(x, (0, 2, 5))
        self.assertEqual(max_lag, 5)
        self.assertEqual(design.shape, (95, 6))
        # First output row contains x[t], x[t-2], x[t-5] at t=5.
        self.assertTrue(np.array_equal(design[0, 0:2], x[5]))
        self.assertTrue(np.array_equal(design[0, 2:4], x[3]))
        self.assertTrue(np.array_equal(design[0, 4:6], x[0]))


if __name__ == "__main__":
    unittest.main()
