"""Helpers for distributed-drive experiments in GeoNeuronX.

The functions in this module do not add new biophysics.  They reuse the passive
compartment matrices from :mod:`geoneuronx.passive_cable` and expose two
controls needed by Gate 3:

1. arbitrary current injection at many compartments at once;
2. redistribution of a fixed cable budget while keeping every soma-to-leaf
   path length fixed.

The second control is intentionally strong.  It lets us ask whether *where*
length sits relative to branch points changes the temporal coordinates, even
when total material and end-to-end path length are unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import factorized

from .passive_cable import PassiveTree

Array = np.ndarray


def simulate_distributed_currents(
    tree: PassiveTree,
    currents_nA: Array,
    record_nodes: Iterable[int] | None = None,
    dt_ms: float = 0.5,
) -> Array:
    """Simulate arbitrary compartment currents and return voltages in mV.

    ``currents_nA`` has shape ``(time, tree.n_nodes)``.  The update is the same
    backward-Euler passive cable step used by ``simulate_passive_tree``::

        (C/dt + G) V[t+1] = C/dt V[t] + I[t+1]

    This function only generalizes the input from one current-injection node to
    many simultaneous current-injection nodes.
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    current = np.asarray(currents_nA, dtype=float)
    if current.ndim != 2 or current.shape[1] != tree.n_nodes:
        raise ValueError("currents_nA must have shape (time, tree.n_nodes)")

    records = (
        list(range(tree.n_nodes))
        if record_nodes is None
        else [int(i) for i in record_nodes]
    )
    if any(i < 0 or i >= tree.n_nodes for i in records):
        raise ValueError("invalid record node")

    dt_s = dt_ms * 1e-3
    c_over_dt = tree.capacitance_F / dt_s
    step_matrix = tree.conductance_S + diags(c_over_dt)
    solve = factorized(csc_matrix(step_matrix))

    voltage_V = np.zeros(tree.n_nodes, dtype=float)
    out_mV = np.empty((len(current), len(records)), dtype=float)
    for t in range(len(current)):
        rhs = c_over_dt * voltage_V + current[t] * 1e-9
        voltage_V = solve(rhs)
        out_mV[t] = voltage_V[records] * 1e3
    return out_mV


def fixed_leaf_path_allocation(
    bifurcation_length_um: float,
    total_length_um: float = 840.0,
    leaf_path_length_um: float = 360.0,
) -> Array:
    """Return a symmetric seven-section morphology with two invariants fixed.

    The binary topology is::

        soma -- s0 --+-- s1 --+-- s3
                     |        `-- s4
                     `-- s2 --+-- s5
                              `-- s6

    ``s1`` and ``s2`` are the internal bifurcation sections.  All four terminal
    sections share one length and the trunk ``s0`` has one length.  We solve
    for trunk and terminal length such that both conditions hold exactly:

    * total cable length is ``total_length_um``;
    * every soma-to-leaf cable path is ``leaf_path_length_um``.

    Thus changing ``bifurcation_length_um`` moves cable between topological
    zones without changing the material budget or end-to-end leaf path length.
    """
    b = float(bifurcation_length_um)
    total = float(total_length_um)
    path = float(leaf_path_length_um)

    # Let r=trunk, b=each of 2 bifurcation sections, t=each of 4 terminals.
    # r + b + t = path
    # r + 2b + 4t = total
    # => t = (total - path - b) / 3
    terminal = (total - path - b) / 3.0
    trunk = path - b - terminal
    lengths = np.array(
        [trunk, b, b, terminal, terminal, terminal, terminal], dtype=float
    )
    if np.any(lengths <= 0):
        raise ValueError("requested allocation produces a non-positive section")
    return lengths


def four_channel_probe_groups(tree: PassiveTree) -> dict[str, list[int]]:
    """Four-channel local readouts at distal, bifurcation and trunk zones.

    The groups are deliberately same-width where possible so differences are
    not merely caused by giving one zone many more observed channels.
    ``all_midpoints`` is the seven-section readout used as a high-information
    reference, and ``soma`` is a one-channel bottleneck.
    """
    nodes = tree.section_nodes
    return {
        "distal": [nodes[i][len(nodes[i]) // 2] for i in (3, 4, 5, 6)],
        "bifurcation": [
            nodes[1][len(nodes[1]) // 2],
            nodes[1][-1],
            nodes[2][len(nodes[2]) // 2],
            nodes[2][-1],
        ],
        "trunk": [
            tree.soma_node,
            nodes[0][1],
            nodes[0][len(nodes[0]) // 2],
            nodes[0][-1],
        ],
        "all_midpoints": [section[len(section) // 2] for section in nodes],
        "soma": [tree.soma_node],
    }


def lagged_feature_matrix(
    x: Array,
    lags: tuple[int, ...] = (0, 2, 4, 8, 16, 32, 64),
) -> tuple[Array, int]:
    """Expose explicit digital history for the FIR/delay-line attacker."""
    data = np.asarray(x, dtype=float)
    if data.ndim != 2:
        raise ValueError("x must have shape (time, channel)")
    if not lags or min(lags) < 0:
        raise ValueError("lags must be non-empty and non-negative")
    if len(set(lags)) != len(lags):
        raise ValueError("lags must be unique")
    max_lag = max(lags)
    if len(data) <= max_lag:
        raise ValueError("time series is shorter than the largest lag")

    blocks = []
    for lag in lags:
        end = None if lag == 0 else -lag
        blocks.append(data[max_lag - lag : end])
    return np.concatenate(blocks, axis=1), max_lag
