"""Biophysically-parameterized passive compartmental cable for GeoNeuronX.

This is a deliberately small cable-equation model, not a reconstructed neuron.

The dendrite is a fixed binary topology with seven cable sections:

    soma -- s0 --+-- s1 --+-- s3
                 |        `-- s4
                 `-- s2 --+-- s5
                          `-- s6

Each section is split into the same number of compartments.  Geometry changes
only the physical section lengths.  Diameter, passive membrane properties,
branch count, compartment count, and *total dendritic cable length* are held
fixed.  With constant diameter, fixed total length also fixes total membrane
area.

The state obeys the standard passive compartment equation

    C dV/dt = -G V + I

where G contains membrane leak and axial conductances.  Backward Euler is used
for stable time stepping.  No hand-written H(f, L) transfer law appears here:
frequency responses are obtained by solving the compartmental admittance
matrix itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix
from scipy.sparse.linalg import factorized, spsolve

Array = np.ndarray

# Fixed topology: section parent index.  None means soma.
SECTION_PARENTS: tuple[int | None, ...] = (None, 0, 0, 1, 1, 2, 2)
LEAF_SECTIONS: tuple[int, ...] = (3, 4, 5, 6)


@dataclass(frozen=True)
class PassiveCableParams:
    """Passive membrane and cable parameters in conventional biophysical units."""

    diameter_um: float = 1.5
    soma_diameter_um: float = 20.0
    cm_uF_per_cm2: float = 1.0
    rm_ohm_cm2: float = 20_000.0
    ra_ohm_cm: float = 150.0
    compartments_per_section: int = 8


@dataclass
class PassiveTree:
    """Matrices and bookkeeping for one passive dendritic tree."""

    section_lengths_um: Array
    params: PassiveCableParams
    capacitance_F: Array
    conductance_S: csc_matrix
    section_nodes: list[list[int]]
    section_start_nodes: list[int]
    section_end_nodes: list[int]
    node_dx_cm: Array
    node_area_cm2: Array
    dendritic_area_cm2: float

    @property
    def n_nodes(self) -> int:
        return int(len(self.capacitance_F))

    @property
    def soma_node(self) -> int:
        return 0

    @property
    def total_dendritic_length_um(self) -> float:
        return float(np.sum(self.section_lengths_um))

    @property
    def leaf_nodes(self) -> list[int]:
        return [self.section_end_nodes[i] for i in LEAF_SECTIONS]

    @property
    def branch_probe_nodes(self) -> list[int]:
        """One local voltage probe at the midpoint of every cable section."""
        return [nodes[len(nodes) // 2] for nodes in self.section_nodes]

    @property
    def leaf_path_lengths_um(self) -> Array:
        L = self.section_lengths_um
        return np.array(
            [
                L[0] + L[1] + L[3],
                L[0] + L[1] + L[4],
                L[0] + L[2] + L[5],
                L[0] + L[2] + L[6],
            ],
            dtype=float,
        )


def redistribute_section_lengths(
    total_length_um: float = 840.0,
    heterogeneity: float = 0.0,
    seed: int = 0,
) -> Array:
    """Redistribute a fixed cable budget across the seven sections.

    ``heterogeneity=0`` gives seven equal sections.  Increasing heterogeneity
    moves length from some sections to others along a deterministic random
    zero-sum direction.  The sum is exactly restored to ``total_length_um``.

    Values above about 0.9 create extremely short sections and are intentionally
    outside the default gate sweep.
    """
    if not 0.0 <= heterogeneity < 1.0:
        raise ValueError("heterogeneity must lie in [0, 1)")
    n = len(SECTION_PARENTS)
    mean = float(total_length_um) / n
    if heterogeneity == 0.0:
        return np.full(n, mean, dtype=float)

    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    z -= z.mean()
    max_abs = float(np.max(np.abs(z)))
    if max_abs == 0.0:
        z[0], z[1] = 1.0, -1.0
        max_abs = 1.0
    z /= max_abs

    lengths = mean * (1.0 + heterogeneity * z)
    # z is zero mean, but restore exactly to avoid accumulating float drift.
    lengths *= float(total_length_um) / float(np.sum(lengths))
    if np.any(lengths <= 0):
        raise ValueError("redistribution produced a non-positive cable section")
    return lengths


def build_passive_tree(
    section_lengths_um: Iterable[float],
    params: PassiveCableParams = PassiveCableParams(),
) -> PassiveTree:
    """Discretize the binary dendrite and assemble passive cable matrices."""
    lengths = np.asarray(tuple(section_lengths_um), dtype=float)
    if lengths.shape != (len(SECTION_PARENTS),):
        raise ValueError(f"expected {len(SECTION_PARENTS)} section lengths")
    if np.any(lengths <= 0):
        raise ValueError("all section lengths must be positive")
    if params.compartments_per_section < 2:
        raise ValueError("compartments_per_section must be >= 2")

    nseg = params.compartments_per_section
    n_nodes = 1 + len(lengths) * nseg
    node_dx_cm = np.zeros(n_nodes, dtype=float)
    node_area_cm2 = np.zeros(n_nodes, dtype=float)

    section_nodes: list[list[int]] = []
    section_start_nodes: list[int] = []
    section_end_nodes: list[int] = []
    endpoint: dict[int, int] = {}

    d_cm = params.diameter_um * 1e-4
    next_node = 1
    for k, length_um in enumerate(lengths):
        parent = SECTION_PARENTS[k]
        start = 0 if parent is None else endpoint[parent]
        nodes = list(range(next_node, next_node + nseg))
        next_node += nseg

        dx_um = float(length_um) / nseg
        dx_cm = dx_um * 1e-4
        area = np.pi * d_cm * dx_cm
        node_dx_cm[nodes] = dx_cm
        node_area_cm2[nodes] = area

        section_nodes.append(nodes)
        section_start_nodes.append(start)
        section_end_nodes.append(nodes[-1])
        endpoint[k] = nodes[-1]

    soma_radius_cm = params.soma_diameter_um * 1e-4 / 2.0
    node_area_cm2[0] = 4.0 * np.pi * soma_radius_cm**2
    # Axial center-distance proxy for the soma-to-first-compartment connection.
    node_dx_cm[0] = params.soma_diameter_um * 1e-4

    capacitance_F = (
        params.cm_uF_per_cm2 * 1e-6 * node_area_cm2
    )
    leak_S = (1.0 / params.rm_ohm_cm2) * node_area_cm2

    G = lil_matrix((n_nodes, n_nodes), dtype=float)
    for i, leak in enumerate(leak_S):
        G[i, i] = leak

    cross_section_cm2 = np.pi * (d_cm / 2.0) ** 2
    for k, nodes in enumerate(section_nodes):
        previous = section_start_nodes[k]
        for node in nodes:
            center_distance_cm = 0.5 * (
                node_dx_cm[previous] + node_dx_cm[node]
            )
            g_axial = cross_section_cm2 / (
                params.ra_ohm_cm * center_distance_cm
            )
            G[previous, previous] += g_axial
            G[node, node] += g_axial
            G[previous, node] -= g_axial
            G[node, previous] -= g_axial
            previous = node

    dendritic_area_cm2 = float(
        np.pi * d_cm * np.sum(lengths * 1e-4)
    )

    return PassiveTree(
        section_lengths_um=lengths,
        params=params,
        capacitance_F=capacitance_F,
        conductance_S=csc_matrix(G),
        section_nodes=section_nodes,
        section_start_nodes=section_start_nodes,
        section_end_nodes=section_end_nodes,
        node_dx_cm=node_dx_cm,
        node_area_cm2=node_area_cm2,
        dendritic_area_cm2=dendritic_area_cm2,
    )


def simulate_passive_tree(
    tree: PassiveTree,
    drive_nA: Array,
    drive_node: int = 0,
    record_nodes: Iterable[int] | None = None,
    dt_ms: float = 0.5,
) -> Array:
    """Simulate one current drive and return recorded voltages in mV.

    Backward Euler solves

        (C/dt + G) V[t+1] = C/dt V[t] + I[t+1].
    """
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    drive = np.asarray(drive_nA, dtype=float)
    if drive.ndim != 1:
        raise ValueError("drive_nA must be a 1-D time series")
    if not 0 <= drive_node < tree.n_nodes:
        raise ValueError("invalid drive_node")

    records = (
        list(range(tree.n_nodes))
        if record_nodes is None
        else [int(i) for i in record_nodes]
    )
    if any(i < 0 or i >= tree.n_nodes for i in records):
        raise ValueError("invalid record node")

    dt_s = dt_ms * 1e-3
    C_over_dt = tree.capacitance_F / dt_s
    step_matrix = tree.conductance_S + diags(C_over_dt)
    solve = factorized(csc_matrix(step_matrix))

    voltage_V = np.zeros(tree.n_nodes, dtype=float)
    out_mV = np.empty((len(drive), len(records)), dtype=float)

    for t, current_nA in enumerate(drive):
        rhs = C_over_dt * voltage_V
        rhs[drive_node] += current_nA * 1e-9
        voltage_V = solve(rhs)
        out_mV[t] = voltage_V[records] * 1e3

    return out_mV


def transfer_impedance(
    tree: PassiveTree,
    frequencies_hz: Iterable[float],
    drive_node: int,
    record_nodes: Iterable[int],
) -> Array:
    """Complex transfer impedance in mV/nA from the compartment matrices.

    For a sinusoidal current at frequency f, solve

        (G + j 2 pi f C) V = I.

    This is measured from the cable model itself; no analytic H(f, L) shortcut
    is used.
    """
    freqs = np.asarray(tuple(frequencies_hz), dtype=float)
    records = [int(i) for i in record_nodes]
    response = np.zeros((len(records), len(freqs)), dtype=complex)
    current_A = np.zeros(tree.n_nodes, dtype=complex)
    current_A[drive_node] = 1e-9  # 1 nA

    for j, freq in enumerate(freqs):
        admittance = tree.conductance_S + diags(
            1j * 2.0 * np.pi * freq * tree.capacitance_F
        )
        voltage_V = spsolve(admittance, current_A)
        response[:, j] = voltage_V[records] * 1e3  # mV per 1 nA

    return response


def normalized_transfer_condition(response: Array) -> float:
    """Condition number after normalizing each temporal-mode response column."""
    H = np.asarray(response, dtype=complex)
    Hn = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-30)
    singular = np.linalg.svd(Hn, compute_uv=False)
    if len(singular) == 0 or singular[-1] <= 1e-15:
        return float("inf")
    return float(singular[0] / singular[-1])


def reciprocity_error(
    tree: PassiveTree,
    node_a: int,
    node_b: int,
    frequencies_hz: Iterable[float] = (2.0, 20.0, 100.0),
) -> float:
    """Maximum relative error of passive transfer impedance Z_ab == Z_ba."""
    ab = transfer_impedance(tree, frequencies_hz, node_a, [node_b])[0]
    ba = transfer_impedance(tree, frequencies_hz, node_b, [node_a])[0]
    denom = np.maximum(np.maximum(np.abs(ab), np.abs(ba)), 1e-30)
    return float(np.max(np.abs(ab - ba) / denom))
