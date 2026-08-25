"""Gate 4: the first explicit GeoNeuronX dynamical-neuron computation.

Gates 2-3 established a linear fact: passive dendritic geometry can convert
recent input history into simultaneous spatial branch state, but an explicit
digital delay line can recover the same information.  Gate 4 asks the next
question:

    does applying nonlinear computation LOCALLY to those history-bearing branch
    coordinates before soma collapse create a qualitatively richer primitive?

This is intentionally an abstraction, not a conductance-based NMDA model.
Each recorded branch midpoint is standardized and passed through several local
rectified thresholds.  A linear soma/readout then learns from those nonlinear
branch features.

The task is delayed XOR on ONE scalar input stream.  The target at time t is
+1 when the sign of the current input differs from its sign 16 ms earlier, and
-1 otherwise.  This task is chosen because:

* a linear memoryless readout cannot solve it;
* a linear FIR can possess the relevant history and still cannot solve XOR;
* a nonlinear feature map over history can solve it.

The attacks are therefore:

1. soma voltage -> linear readout;
2. nonlinearities only AFTER soma collapse;
3. all branch voltages -> linear readout;
4. branch-local nonlinearities -> linear soma/readout;
5. explicit digital FIR -> linear readout;
6. the same FIR plus all quadratic products -> linear readout.

The last arm is load-bearing.  If it wins cheaply, GeoNeuronX has not invented a
superior nonlinear computer; it has shown one physical way to instantiate a
stateful nonlinear temporal feature map.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoneuronx.distributed import (  # noqa: E402
    fixed_leaf_path_allocation,
    simulate_distributed_currents,
)
from geoneuronx.dynamical import (  # noqa: E402
    delayed_xor_target,
    quadratic_feature_map,
    rectified_threshold_basis,
    ridge_readout_metrics,
    telegraph_signal,
)
from geoneuronx.passive_cable import (  # noqa: E402
    LEAF_SECTIONS,
    PassiveCableParams,
    build_passive_tree,
)


DT_MS = 0.5
N_SAMPLES = 12_000
BURN = 2_000
SEEDS = tuple(range(5))
BIFURCATION_LENGTHS_UM = (40.0, 120.0, 210.0)
TOTAL_LENGTH_UM = 840.0
LEAF_PATH_LENGTH_UM = 360.0
TARGET_LAG = 32  # 16 ms
TELEGRAPH_HOLD_SAMPLES = 4
LOCAL_THRESHOLDS = (-1.5, -0.75, 0.0, 0.75, 1.5)
DIGITAL_LAGS = (0, 2, 4, 8, 16, 32, 64)
LEAF_GAINS = np.array((1.0, 0.70, -0.55, 0.35), dtype=float)
DRIVE_NA = 0.05


def digital_delay_coordinates(signal: np.ndarray, start: int) -> np.ndarray:
    """Return explicit delayed copies aligned to output times ``start:``."""
    n = len(signal)
    columns = []
    for lag in DIGITAL_LAGS:
        end = n if lag == 0 else n - lag
        columns.append(signal[start - lag : end])
    return np.column_stack(columns)


def local_nonlinear_score(x: np.ndarray, target: np.ndarray) -> dict[str, float]:
    train_rows = int(len(target) * 0.60)
    features = rectified_threshold_basis(
        x,
        train_rows=train_rows,
        thresholds=LOCAL_THRESHOLDS,
    )
    return ridge_readout_metrics(features, target)


def run_one(bifurcation_um: float, seed: int) -> dict:
    rng = np.random.default_rng(30_000 + seed)
    signal = telegraph_signal(
        N_SAMPLES,
        rng,
        hold_samples=TELEGRAPH_HOLD_SAMPLES,
        noise_std=0.05,
    )

    lengths = fixed_leaf_path_allocation(
        bifurcation_um,
        total_length_um=TOTAL_LENGTH_UM,
        leaf_path_length_um=LEAF_PATH_LENGTH_UM,
    )
    tree = build_passive_tree(lengths, PassiveCableParams())

    # One scalar temporal stream is copied to four distal leaves through fixed
    # gains.  No clean delayed coordinates are handed to the morphology.
    current = np.zeros((N_SAMPLES, tree.n_nodes), dtype=float)
    for section, gain in zip(LEAF_SECTIONS, LEAF_GAINS):
        current[:, tree.section_end_nodes[section]] += DRIVE_NA * gain * signal

    branch_midpoints = [nodes[len(nodes) // 2] for nodes in tree.section_nodes]
    record_nodes = [tree.soma_node] + branch_midpoints
    voltage = simulate_distributed_currents(
        tree,
        current,
        record_nodes=record_nodes,
        dt_ms=DT_MS,
    )

    start = max(BURN, TARGET_LAG, max(DIGITAL_LAGS))
    target = delayed_xor_target(signal, start=start, lag=TARGET_LAG)
    soma = voltage[start:, 0:1]
    branches = voltage[start:, 1:]
    digital = digital_delay_coordinates(signal, start)

    scores = {
        "soma_linear": ridge_readout_metrics(soma, target),
        "soma_nonlinear": local_nonlinear_score(soma, target),
        "branch_linear": ridge_readout_metrics(branches, target),
        "branch_local_nonlinear": local_nonlinear_score(branches, target),
        "digital_fir_linear": ridge_readout_metrics(digital, target),
        "digital_fir_quadratic": ridge_readout_metrics(
            quadratic_feature_map(digital), target
        ),
    }

    return {
        "seed": seed,
        "bifurcation_length_um": bifurcation_um,
        "section_lengths_um": lengths.tolist(),
        "total_length_um": tree.total_dendritic_length_um,
        "dendritic_area_cm2": tree.dendritic_area_cm2,
        "positive_fraction": float(np.mean(target > 0)),
        "scores": scores,
    }


def summarize(rows: list[dict]) -> dict:
    arms = rows[0]["scores"].keys()
    out = {}
    for arm in arms:
        out[arm] = {
            key: float(np.mean([row["scores"][arm][key] for row in rows]))
            for key in rows[0]["scores"][arm]
        }
    out["positive_fraction"] = float(
        np.mean([row["positive_fraction"] for row in rows])
    )
    return out


def run() -> dict:
    raw: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for bif in BIFURCATION_LENGTHS_UM:
        rows = [run_one(bif, seed) for seed in SEEDS]
        key = f"{bif:g}"
        raw[key] = rows
        summary[key] = summarize(rows)

    short = summary["40"]
    long = summary["210"]
    all_lengths = [
        row["total_length_um"]
        for rows in raw.values()
        for row in rows
    ]
    all_areas = [
        row["dendritic_area_cm2"]
        for rows in raw.values()
        for row in rows
    ]
    length_rel_range = (
        max(all_lengths) - min(all_lengths)
    ) / np.mean(all_lengths)
    area_rel_range = (
        max(all_areas) - min(all_areas)
    ) / np.mean(all_areas)

    gates = {
        "fixed_material": bool(
            length_rel_range < 1e-12 and area_rel_range < 1e-12
        ),
        "linear_memory_is_not_enough": bool(
            long["branch_linear"]["accuracy"] < 0.53
            and long["digital_fir_linear"]["accuracy"] < 0.53
        ),
        "local_nonlinearity_beats_branch_linear": bool(
            long["branch_local_nonlinear"]["accuracy"]
            > long["branch_linear"]["accuracy"] + 0.07
        ),
        "local_before_collapse_beats_soma_nonlinearity": bool(
            long["branch_local_nonlinear"]["accuracy"]
            > long["soma_nonlinear"]["accuracy"] + 0.06
        ),
        "geometry_modulates_nonlinear_compute": bool(
            long["branch_local_nonlinear"]["accuracy"]
            > short["branch_local_nonlinear"]["accuracy"] + 0.01
        ),
        "digital_quadratic_attacker_wins": bool(
            long["digital_fir_quadratic"]["accuracy"] > 0.99
        ),
    }

    return {
        "config": {
            "dt_ms": DT_MS,
            "n_samples": N_SAMPLES,
            "burn": BURN,
            "seeds": list(SEEDS),
            "bifurcation_lengths_um": list(BIFURCATION_LENGTHS_UM),
            "total_length_um": TOTAL_LENGTH_UM,
            "leaf_path_length_um": LEAF_PATH_LENGTH_UM,
            "target_lag_samples": TARGET_LAG,
            "target_lag_ms": TARGET_LAG * DT_MS,
            "telegraph_hold_samples": TELEGRAPH_HOLD_SAMPLES,
            "local_thresholds": list(LOCAL_THRESHOLDS),
            "digital_lags_samples": list(DIGITAL_LAGS),
            "leaf_gains": LEAF_GAINS.tolist(),
        },
        "summary": summary,
        "fixed_material_relative_length_range": float(length_rel_range),
        "fixed_material_relative_area_range": float(area_rel_range),
        "gates": gates,
        "raw": raw,
    }


def print_result(result: dict) -> None:
    print("Gate 4 — DYNAMICAL NEURON / LOCAL NONLINEAR HISTORY COMPUTE\n")
    print("bif  arm                         corr     accuracy    NMSE")
    print("-" * 66)
    for bif in BIFURCATION_LENGTHS_UM:
        s = result["summary"][f"{bif:g}"]
        for arm in (
            "soma_linear",
            "soma_nonlinear",
            "branch_linear",
            "branch_local_nonlinear",
            "digital_fir_linear",
            "digital_fir_quadratic",
        ):
            m = s[arm]
            print(
                f"{bif:3.0f}  {arm:27s} "
                f"{m['corr']:7.4f}   {m['accuracy']:7.4f}   {m['nmse']:7.4f}"
            )
        print()
    print("gates:", result["gates"])


def main() -> None:
    result = run()
    print_result(result)
    out = Path("results/gate4_dynamical_neuron.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
