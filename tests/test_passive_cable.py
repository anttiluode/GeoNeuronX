import unittest

import numpy as np

from geoneuronx.passive_cable import (
    PassiveCableParams,
    build_passive_tree,
    normalized_transfer_condition,
    reciprocity_error,
    redistribute_section_lengths,
    simulate_passive_tree,
    transfer_impedance,
)


class PassiveCableTests(unittest.TestCase):
    def test_fixed_material_budget(self):
        totals = []
        areas = []
        for h in (0.0, 0.2, 0.4, 0.6, 0.8):
            for seed in range(3):
                lengths = redistribute_section_lengths(840.0, h, seed)
                tree = build_passive_tree(lengths)
                totals.append(tree.total_dendritic_length_um)
                areas.append(tree.dendritic_area_cm2)
        self.assertLess(np.ptp(totals), 1e-10)
        self.assertLess(np.ptp(areas), 1e-15)

    def test_heterogeneity_changes_path_lengths(self):
        equal = build_passive_tree(redistribute_section_lengths(840.0, 0.0, 0))
        varied = build_passive_tree(redistribute_section_lengths(840.0, 0.8, 0))
        self.assertAlmostEqual(float(np.std(equal.leaf_path_lengths_um)), 0.0)
        self.assertGreater(float(np.std(varied.leaf_path_lengths_um)), 1.0)

    def test_passive_transfer_is_reciprocal(self):
        tree = build_passive_tree(redistribute_section_lengths(840.0, 0.8, 1))
        err = reciprocity_error(
            tree,
            tree.soma_node,
            tree.branch_probe_nodes[-1],
            (2.0, 20.0, 100.0),
        )
        self.assertLess(err, 1e-10)

    def test_simulation_is_finite(self):
        tree = build_passive_tree(
            redistribute_section_lengths(840.0, 0.6, 2),
            PassiveCableParams(),
        )
        t = np.arange(2000) * 0.5e-3
        drive = 0.05 * np.sin(2 * np.pi * 20.0 * t)
        v = simulate_passive_tree(
            tree,
            drive,
            drive_node=tree.soma_node,
            record_nodes=tree.branch_probe_nodes,
            dt_ms=0.5,
        )
        self.assertEqual(v.shape, (len(t), 7))
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertGreater(float(np.std(v)), 0.0)

    def test_varied_geometry_improves_transfer_condition_in_reference_case(self):
        equal = build_passive_tree(redistribute_section_lengths(840.0, 0.0, 0))
        varied = build_passive_tree(redistribute_section_lengths(840.0, 0.8, 0))
        freqs = (2.0, 20.0, 100.0)
        h0 = transfer_impedance(
            equal, freqs, equal.soma_node, equal.branch_probe_nodes
        )
        h8 = transfer_impedance(
            varied, freqs, varied.soma_node, varied.branch_probe_nodes
        )
        self.assertLess(
            normalized_transfer_condition(h8),
            normalized_transfer_condition(h0),
        )


if __name__ == "__main__":
    unittest.main()
