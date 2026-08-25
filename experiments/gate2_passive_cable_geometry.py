"""Gate 2: passive cable geometry under a fixed material budget.

This gate removes Gate 0's hand-written H(f, L).  The dendrite is now a
biophysically-parameterized passive RC cable tree.  Across morphologies we hold
fixed:

- topology and branch count;
- number of compartments;
- dendrite diameter;
- total cable length;
- therefore total dendritic membrane area;
- passive Cm, Rm, and axial resistivity.

Only the allocation of the fixed cable budget across seven tree sections
changes.

The clean assay drives the soma with one scalar mixture and observes one local
voltage from every section.  This deliberately isolates the *transfer basis*
created by the cable.  It is not meant as the biological direction of synaptic
drive.  A passive cable has reciprocal transfer impedance, and the gate reports
the numerical reciprocity error.  A later gate should distribute synapses over
the dendrites and score inward propagation directly.

Two source regimes are used:

1. Narrow-band dynamical sources (2, 20, 100 Hz): a case where different
   passive filters ought to be useful.
2. Broad AR(1) colored sources: a stress test where a passive low-pass tree is
   not guaranteed to help.

The primary question is whether branch-local signals become more source
separable as fixed cable is redistributed into more diverse path lengths.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoneuronx.core import amuse, make_temporal_sources, pca_scores, recovery_score
from geoneuronx.passive_cable import (
    PassiveCableParams,
    build_passive_tree,
    normalized_transfer_condition,
    reciprocity_error,
    redistribute_section_lengths,
    simulate_passive_tree,
    transfer_impedance,
)


DT_MS = 0.5
FS_HZ = 1000.0 / DT_MS
N_SAMPLES = 16_000
BURN = 2_000
SEEDS = tuple(range(5))
HETEROGENEITY = (0.0, 0.2, 0.4, 0.6, 0.8)
TOTAL_LENGTH_UM = 840.0
OSC_FREQS_HZ = (2.0, 20.0, 100.0)
AMUSE_LAG = 16  # one lagged covariance at tau = 8 ms
MEASUREMENT_NOISE_FRACTION = 0.003
DRIVE_NA = 0.05


def ar2_oscillator(
    frequency_hz: float,
    n: int,
    rng: np.random.Generator,
    radius: float = 0.997,
) -> np.ndarray:
    """Stationary narrow-band AR(2) process with unit marginal variance."""
    theta = 2.0 * np.pi * frequency_hz / FS_HZ
    a1 = 2.0 * radius * np.cos(theta)
    a2 = -(radius**2)
    innovation = rng.normal(size=n)
    x = np.zeros(n, dtype=float)
    for t in range(2, n):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + innovation[t]
    x -= x.mean()
    x /= x.std() + 1e-12
    return x


def make_oscillatory_sources(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack(
        [ar2_oscillator(freq, n, rng) for freq in OSC_FREQS_HZ]
    )


def add_measurement_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    fraction: float = MEASUREMENT_NOISE_FRACTION,
) -> np.ndarray:
    scale = float(np.std(x))
    if scale == 0.0 or fraction == 0.0:
        return x.copy()
    return x + fraction * scale * rng.normal(size=x.shape)


def oracle_linear_metrics(
    x: np.ndarray,
    sources: np.ndarray,
    train_fraction: float = 0.60,
    ridge: float = 1e-4,
) -> tuple[float, float]:
    """Cross-validated supervised linear attacker.

    Returns mean same-source absolute correlation and normalized test MSE.
    This attacker asks whether the information is in the branch basis at all;
    it does not get to claim blindness.
    """
    split = int(len(x) * train_fraction)
    x_train = x[:split]
    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True) + 1e-12
    xs = (x - mean) / scale

    gram = xs[:split].T @ xs[:split]
    weights = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]),
        xs[:split].T @ sources[:split],
    )
    pred = xs[split:] @ weights
    truth = sources[split:]

    corr = [
        abs(np.corrcoef(pred[:, i], truth[:, i])[0, 1])
        for i in range(truth.shape[1])
    ]
    nmse = float(
        np.mean((pred - truth) ** 2) / (np.mean(truth**2) + 1e-12)
    )
    return float(np.mean(corr)), nmse


def digital_rc_bank(
    mixture: np.ndarray,
    taus_ms: tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100),
) -> np.ndarray:
    """Boring explicit temporal-filter attacker with seven scalar states."""
    out = np.zeros((len(mixture), len(taus_ms)), dtype=float)
    for j, tau_ms in enumerate(taus_ms):
        a = np.exp(-DT_MS / tau_ms)
        for t in range(1, len(mixture)):
            out[t, j] = a * out[t - 1, j] + (1.0 - a) * mixture[t]
    return out


def score_branch_basis(
    x: np.ndarray,
    truth: np.ndarray,
    amuse_lag: int = AMUSE_LAG,
) -> dict[str, float]:
    pca = pca_scores(x, truth.shape[1])
    pca_recovery, _ = recovery_score(pca, truth)

    am = amuse(x, truth.shape[1], lag=amuse_lag)
    amuse_recovery, _ = recovery_score(am, truth)

    oracle_corr, oracle_nmse = oracle_linear_metrics(x, truth)
    return {
        "pca_recovery": pca_recovery,
        "amuse_recovery": amuse_recovery,
        "oracle_corr": oracle_corr,
        "oracle_nmse": oracle_nmse,
    }


def run_one(
    heterogeneity: float,
    seed: int,
    source_kind: str,
) -> dict[str, float]:
    rng = np.random.default_rng(10_000 + seed)
    if source_kind == "oscillatory":
        source = make_oscillatory_sources(N_SAMPLES, rng)
    elif source_kind == "ar1":
        source = make_temporal_sources(N_SAMPLES, rng)
    else:
        raise ValueError(source_kind)

    mixture = source.sum(axis=1) / np.sqrt(source.shape[1])
    lengths = redistribute_section_lengths(
        total_length_um=TOTAL_LENGTH_UM,
        heterogeneity=heterogeneity,
        seed=seed,
    )
    tree = build_passive_tree(lengths, PassiveCableParams())
    probes = tree.branch_probe_nodes

    branch_voltage = simulate_passive_tree(
        tree,
        drive_nA=DRIVE_NA * mixture,
        drive_node=tree.soma_node,
        record_nodes=probes,
        dt_ms=DT_MS,
    )
    branch_voltage = add_measurement_noise(branch_voltage, rng)

    x = branch_voltage[BURN:]
    truth = source[BURN:]
    metrics = score_branch_basis(x, truth)

    response = transfer_impedance(
        tree,
        OSC_FREQS_HZ,
        drive_node=tree.soma_node,
        record_nodes=probes,
    )
    metrics.update(
        {
            "heterogeneity": heterogeneity,
            "seed": seed,
            "path_std_um": float(np.std(tree.leaf_path_lengths_um)),
            "min_section_um": float(np.min(lengths)),
            "max_section_um": float(np.max(lengths)),
            "total_length_um": tree.total_dendritic_length_um,
            "dendritic_area_cm2": tree.dendritic_area_cm2,
            "transfer_condition": normalized_transfer_condition(response),
        }
    )
    return metrics


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = (
        "path_std_um",
        "transfer_condition",
        "pca_recovery",
        "amuse_recovery",
        "oracle_corr",
        "oracle_nmse",
        "total_length_um",
        "dendritic_area_cm2",
    )
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def run() -> dict:
    by_kind: dict[str, dict[str, dict[str, float]]] = {}
    raw_rows: dict[str, list[dict[str, float]]] = {}

    for kind in ("oscillatory", "ar1"):
        rows: list[dict[str, float]] = []
        summary: dict[str, dict[str, float]] = {}
        for h in HETEROGENEITY:
            local = [run_one(h, seed, kind) for seed in SEEDS]
            rows.extend(local)
            summary[f"{h:.1f}"] = summarize(local)
        by_kind[kind] = summary
        raw_rows[kind] = rows

    # Explicit seven-state digital filter bank attacker on the exact same source
    # realizations.  This exists to prevent "geometry is a better filter bank"
    # from sneaking into the interpretation.
    digital: dict[str, dict[str, float]] = {}
    for kind in ("oscillatory", "ar1"):
        scores = []
        for seed in SEEDS:
            rng = np.random.default_rng(10_000 + seed)
            source = (
                make_oscillatory_sources(N_SAMPLES, rng)
                if kind == "oscillatory"
                else make_temporal_sources(N_SAMPLES, rng)
            )
            mixture = source.sum(axis=1) / np.sqrt(source.shape[1])
            bank = digital_rc_bank(mixture)
            scores.append(score_branch_basis(bank[BURN:], source[BURN:]))
        digital[kind] = {
            key: float(np.mean([s[key] for s in scores]))
            for key in scores[0]
        }

    # Passive reciprocity check.  We use one heterogeneous tree and compare
    # soma->branch transfer with branch->soma transfer across the source bands.
    check_lengths = redistribute_section_lengths(
        TOTAL_LENGTH_UM, heterogeneity=0.8, seed=0
    )
    check_tree = build_passive_tree(check_lengths)
    reciprocity = reciprocity_error(
        check_tree,
        check_tree.soma_node,
        check_tree.branch_probe_nodes[-1],
        OSC_FREQS_HZ,
    )

    osc0 = by_kind["oscillatory"]["0.0"]
    osc8 = by_kind["oscillatory"]["0.8"]
    ar0 = by_kind["ar1"]["0.0"]
    ar8 = by_kind["ar1"]["0.8"]

    material_lengths = [
        row["total_length_um"]
        for kind_rows in raw_rows.values()
        for row in kind_rows
    ]
    material_areas = [
        row["dendritic_area_cm2"]
        for kind_rows in raw_rows.values()
        for row in kind_rows
    ]
    length_rel_range = (
        max(material_lengths) - min(material_lengths)
    ) / np.mean(material_lengths)
    area_rel_range = (
        max(material_areas) - min(material_areas)
    ) / np.mean(material_areas)

    gates = {
        "fixed_material": bool(
            length_rel_range < 1e-12 and area_rel_range < 1e-12
        ),
        "passive_reciprocity": bool(reciprocity < 1e-10),
        "transfer_basis_improves": bool(
            osc8["transfer_condition"] < 0.5 * osc0["transfer_condition"]
        ),
        "oscillatory_blind_separation_improves": bool(
            osc8["amuse_recovery"] > osc0["amuse_recovery"] + 0.08
        ),
        "oscillatory_information_improves": bool(
            osc8["oracle_corr"] > osc0["oracle_corr"] + 0.05
        ),
    }

    return {
        "config": {
            "dt_ms": DT_MS,
            "n_samples": N_SAMPLES,
            "burn": BURN,
            "seeds": list(SEEDS),
            "heterogeneity": list(HETEROGENEITY),
            "total_length_um": TOTAL_LENGTH_UM,
            "oscillatory_frequencies_hz": list(OSC_FREQS_HZ),
            "amuse_lag_samples": AMUSE_LAG,
            "amuse_lag_ms": AMUSE_LAG * DT_MS,
            "measurement_noise_fraction": MEASUREMENT_NOISE_FRACTION,
        },
        "summary": by_kind,
        "digital_rc_attacker": digital,
        "reciprocity_relative_error": reciprocity,
        "fixed_material_relative_length_range": length_rel_range,
        "fixed_material_relative_area_range": area_rel_range,
        "gates": gates,
        "boundary": {
            "ar1_amuse_change_h0_to_h08": (
                ar8["amuse_recovery"] - ar0["amuse_recovery"]
            ),
            "ar1_oracle_change_h0_to_h08": (
                ar8["oracle_corr"] - ar0["oracle_corr"]
            ),
        },
    }


def print_table(result: dict) -> None:
    for kind in ("oscillatory", "ar1"):
        title = (
            "NARROW-BAND OSCILLATORY SOURCES"
            if kind == "oscillatory"
            else "BROAD AR(1) STRESS TEST"
        )
        print("\n" + title)
        print(
            "h    pathSD   cond(H)   PCA      AMUSE    oracle   oracle NMSE"
        )
        print("-" * 72)
        for h in HETEROGENEITY:
            s = result["summary"][kind][f"{h:.1f}"]
            print(
                f"{h:0.1f}  "
                f"{s['path_std_um']:7.2f}  "
                f"{s['transfer_condition']:8.1f}  "
                f"{s['pca_recovery']:7.4f}  "
                f"{s['amuse_recovery']:7.4f}  "
                f"{s['oracle_corr']:7.4f}  "
                f"{s['oracle_nmse']:10.4f}"
            )

    print("\nDIGITAL 7-STATE RC FILTER BANK ATTACKER")
    for kind, score in result["digital_rc_attacker"].items():
        print(
            f"{kind:12s} "
            f"PCA={score['pca_recovery']:.4f} "
            f"AMUSE={score['amuse_recovery']:.4f} "
            f"oracle={score['oracle_corr']:.4f} "
            f"NMSE={score['oracle_nmse']:.4f}"
        )

    print(
        "\nreciprocity relative error:",
        f"{result['reciprocity_relative_error']:.3e}",
    )
    print("gates:", result["gates"])
    print("boundary:", result["boundary"])


def main() -> None:
    result = run()
    print_table(result)

    out = Path("results/gate2_passive_cable_geometry.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
