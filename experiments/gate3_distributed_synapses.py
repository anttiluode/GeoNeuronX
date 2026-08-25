"""Gate 3: distributed synaptic drive and where temporal coordinates live.

Gate 2 established that a passive cable can instantiate a useful temporal
filter bank under a fixed material budget.  Gate 3 makes the drive direction
more neuron-like and strengthens the morphology control.

Three hidden narrow-band dynamical sources are summed into ONE scalar mixture.
Copies of that same mixture are injected at four distal leaf synapses with
fixed gains.  The input is therefore rank-one at any instant: spatial synapse
identity alone cannot solve the three-source problem.

We then move the same 840 um cable budget between three topological zones:

    trunk <-> bifurcation sections <-> terminal sections

while keeping EVERY soma-to-leaf cable path exactly 360 um.  So this is not a
"longer overall path wins" test.  It asks whether where cable sits relative to
branch points changes the spatial accessibility of temporal history.

At distal, bifurcation, trunk, all-section, and soma readouts we compare:

* AMUSE: blind one-lag second-order separation;
* static oracle: supervised memoryless linear readout;
* FIR oracle: the same supervised attacker with an explicit digital delay line.

The FIR attacker is load-bearing.  If a digital delay line removes the geometry
difference, the correct interpretation is not "geometry created information".
It is that morphology physically *materialized temporal state into space* so a
memoryless local reader could access it.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoneuronx.core import amuse, recovery_score  # noqa: E402
from geoneuronx.distributed import (  # noqa: E402
    fixed_leaf_path_allocation,
    four_channel_probe_groups,
    lagged_feature_matrix,
    simulate_distributed_currents,
)
from geoneuronx.passive_cable import (  # noqa: E402
    LEAF_SECTIONS,
    PassiveCableParams,
    build_passive_tree,
)


DT_MS = 0.5
FS_HZ = 1000.0 / DT_MS
N_SAMPLES = 12_000
BURN = 2_000
SEEDS = tuple(range(5))
BIFURCATION_LENGTHS_UM = (40.0, 80.0, 120.0, 160.0, 210.0)
TOTAL_LENGTH_UM = 840.0
LEAF_PATH_LENGTH_UM = 360.0
OSC_FREQS_HZ = (2.0, 20.0, 100.0)
AMUSE_LAG = 16  # 8 ms
MEASUREMENT_NOISE_FRACTION = 0.003
DRIVE_NA = 0.05
LEAF_GAINS = np.array((1.0, 0.70, -0.55, 0.35), dtype=float)
FIR_LAGS = (0, 2, 4, 8, 16, 32, 64)  # up to 32 ms of explicit digital history


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


def make_sources(n: int, rng: np.random.Generator) -> np.ndarray:
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
    """Cross-validated supervised memoryless linear attacker."""
    split = int(len(x) * train_fraction)
    mean = x[:split].mean(axis=0, keepdims=True)
    scale = x[:split].std(axis=0, keepdims=True) + 1e-12
    z = (x - mean) / scale

    gram = z[:split].T @ z[:split]
    weights = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]),
        z[:split].T @ sources[:split],
    )
    pred = z[split:] @ weights
    truth = sources[split:]
    corr = [
        abs(np.corrcoef(pred[:, i], truth[:, i])[0, 1])
        for i in range(truth.shape[1])
    ]
    nmse = float(
        np.mean((pred - truth) ** 2) / (np.mean(truth**2) + 1e-12)
    )
    return float(np.mean(corr)), nmse


def oracle_fir_metrics(
    x: np.ndarray,
    sources: np.ndarray,
    lags: tuple[int, ...] = FIR_LAGS,
) -> tuple[float, float]:
    """Same oracle, but hand it explicit delayed copies of each observed channel."""
    design, max_lag = lagged_feature_matrix(x, lags)
    return oracle_linear_metrics(
        design,
        sources[max_lag:],
        train_fraction=0.60,
        ridge=1e-2,
    )


def score_readout(
    x: np.ndarray,
    truth: np.ndarray,
    allow_amuse: bool = True,
) -> dict[str, float | None]:
    if allow_amuse:
        separated = amuse(x, truth.shape[1], lag=AMUSE_LAG)
        amuse_recovery, _ = recovery_score(separated, truth)
    else:
        amuse_recovery = None

    static_corr, static_nmse = oracle_linear_metrics(x, truth)
    fir_corr, fir_nmse = oracle_fir_metrics(x, truth)
    return {
        "amuse_recovery": amuse_recovery,
        "static_oracle_corr": static_corr,
        "static_oracle_nmse": static_nmse,
        "fir_oracle_corr": fir_corr,
        "fir_oracle_nmse": fir_nmse,
    }


def run_one(bifurcation_um: float, seed: int) -> dict:
    rng = np.random.default_rng(20_000 + seed)
    sources = make_sources(N_SAMPLES, rng)
    mixture = sources.sum(axis=1) / np.sqrt(sources.shape[1])

    lengths = fixed_leaf_path_allocation(
        bifurcation_um,
        total_length_um=TOTAL_LENGTH_UM,
        leaf_path_length_um=LEAF_PATH_LENGTH_UM,
    )
    tree = build_passive_tree(lengths, PassiveCableParams())
    groups = four_channel_probe_groups(tree)

    current = np.zeros((N_SAMPLES, tree.n_nodes), dtype=float)
    for section, gain in zip(LEAF_SECTIONS, LEAF_GAINS):
        current[:, tree.section_end_nodes[section]] += DRIVE_NA * gain * mixture

    record_nodes = sorted({node for nodes in groups.values() for node in nodes})
    voltage = simulate_distributed_currents(
        tree,
        current,
        record_nodes=record_nodes,
        dt_ms=DT_MS,
    )
    voltage = add_measurement_noise(voltage, rng)
    index = {node: i for i, node in enumerate(record_nodes)}

    truth = sources[BURN:]
    scores = {}
    for name, nodes in groups.items():
        x = voltage[BURN:, [index[node] for node in nodes]]
        scores[name] = score_readout(
            x,
            truth,
            allow_amuse=(name != "soma"),
        )

    return {
        "seed": seed,
        "bifurcation_length_um": bifurcation_um,
        "section_lengths_um": lengths.tolist(),
        "total_length_um": tree.total_dendritic_length_um,
        "dendritic_area_cm2": tree.dendritic_area_cm2,
        "leaf_paths_um": tree.leaf_path_lengths_um.tolist(),
        "scores": scores,
    }


def summarize(rows: list[dict]) -> dict:
    groups = rows[0]["scores"].keys()
    summary: dict[str, dict[str, float | None]] = {}
    for group in groups:
        keys = rows[0]["scores"][group].keys()
        local = {}
        for key in keys:
            vals = [row["scores"][group][key] for row in rows]
            vals = [v for v in vals if v is not None]
            local[key] = None if not vals else float(np.mean(vals))
        summary[group] = local
    return summary


def run() -> dict:
    raw: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for bif in BIFURCATION_LENGTHS_UM:
        rows = [run_one(bif, seed) for seed in SEEDS]
        key = f"{bif:g}"
        raw[key] = rows
        summary[key] = summarize(rows)

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
    all_paths = np.array(
        [row["leaf_paths_um"] for rows in raw.values() for row in rows]
    )
    length_rel_range = (max(all_lengths) - min(all_lengths)) / np.mean(all_lengths)
    area_rel_range = (max(all_areas) - min(all_areas)) / np.mean(all_areas)
    path_rel_range = float(
        (np.max(all_paths) - np.min(all_paths)) / np.mean(all_paths)
    )

    short = summary["40"]
    long = summary["210"]
    fir_values = [
        summary[f"{b:g}"][group]["fir_oracle_corr"]
        for b in BIFURCATION_LENGTHS_UM
        for group in ("distal", "bifurcation", "all_midpoints", "soma")
    ]

    gates = {
        "fixed_material": bool(length_rel_range < 1e-12 and area_rel_range < 1e-12),
        "fixed_leaf_path_length": bool(path_rel_range < 1e-12),
        "distal_blind_access_improves": bool(
            long["distal"]["amuse_recovery"]
            > short["distal"]["amuse_recovery"] + 0.05
        ),
        "bifurcation_blind_access_improves": bool(
            long["bifurcation"]["amuse_recovery"]
            > short["bifurcation"]["amuse_recovery"] + 0.20
        ),
        "history_is_redistributed": bool(
            long["bifurcation"]["static_oracle_corr"]
            > short["bifurcation"]["static_oracle_corr"] + 0.10
            and long["trunk"]["static_oracle_corr"]
            < short["trunk"]["static_oracle_corr"] - 0.04
        ),
        "digital_delay_erases_geometry_advantage": bool(
            max(fir_values) - min(fir_values) < 0.02
        ),
        "soma_needs_history": bool(
            long["soma"]["static_oracle_corr"] < 0.40
            and long["soma"]["fir_oracle_corr"] > 0.90
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
            "source_frequencies_hz": list(OSC_FREQS_HZ),
            "amuse_lag_samples": AMUSE_LAG,
            "amuse_lag_ms": AMUSE_LAG * DT_MS,
            "fir_lags_samples": list(FIR_LAGS),
            "leaf_gains": LEAF_GAINS.tolist(),
            "measurement_noise_fraction": MEASUREMENT_NOISE_FRACTION,
        },
        "summary": summary,
        "fixed_material_relative_length_range": length_rel_range,
        "fixed_material_relative_area_range": area_rel_range,
        "fixed_leaf_path_relative_range": path_rel_range,
        "gates": gates,
        "raw": raw,
    }


def print_result(result: dict) -> None:
    print("Gate 3 — DISTRIBUTED SYNAPSES / WHERE DOES HISTORY LIVE?\n")
    print(
        "bif  zone          AMUSE    static-oracle   FIR-oracle"
    )
    print("-" * 60)
    for bif in BIFURCATION_LENGTHS_UM:
        key = f"{bif:g}"
        for group in ("distal", "bifurcation", "trunk", "all_midpoints", "soma"):
            s = result["summary"][key][group]
            amuse_value = s["amuse_recovery"]
            amuse_text = "   n/a " if amuse_value is None else f"{amuse_value:7.4f}"
            print(
                f"{bif:3.0f}  {group:13s} {amuse_text}   "
                f"{s['static_oracle_corr']:7.4f}        "
                f"{s['fir_oracle_corr']:7.4f}"
            )
        print()
    print("gates:", result["gates"])


def main() -> None:
    result = run()
    print_result(result)
    out = Path("results/gate3_distributed_synapses.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
