"""Gate 5 — can dynamical neurons divide a temporal world among themselves?

Gate 4 established the single-unit abstraction: internal dynamics generate
history-bearing coordinates and local nonlinearities can compute on them before
soma collapse.  Gate 5 returns to Tuesday's source-separation question and asks
whether several outputs can specialize without source labels.

The experiment deliberately does NOT claim a biological implementation of
AMUSE.  Instead it builds a mathematical bridge:

1. the passive dendrite supplies seven branch-state coordinates;
2. those coordinates are whitened (still a batch convenience in this gate);
3. temporal coincidence samples u(t)=(q(t)+q(t-tau))/sqrt(2) are formed;
4. Sanger's generalized Hebbian rule learns several competing output axes.

For whitened q, Cov[u] = I + C_tau_sym, so the temporal-coincidence covariance
has the same eigenvectors as the AMUSE lag operator.  Sanger therefore provides
an online Oja-like route to the same temporal specialization problem.

Attackers:
* independent Oja copies: no lateral competition, expected to collapse;
* time-shuffled temporal coincidence: same marginals, destroyed lag relation;
* batch AMUSE: mature matrix baseline / upper reference;
* zero-lag Sanger: competition without the temporal contrast.

All source labels are used only for scoring after learning.
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
    simulate_distributed_currents,
)
from geoneuronx.passive_cable import (  # noqa: E402
    LEAF_SECTIONS,
    PassiveCableParams,
    build_passive_tree,
)
from geoneuronx.population import (  # noqa: E402
    independent_oja_population,
    mean_pairwise_abs_correlation,
    sanger_population,
    shuffled_temporal_coincidence_samples,
    temporal_coincidence_samples,
    whiten_population_input,
)

DT_MS = 0.5
FS_HZ = 1000.0 / DT_MS
N_SAMPLES = 16_000
BURN = 2_000
SEEDS = tuple(range(5))
BIFURCATION_LENGTHS_UM = (40.0, 120.0, 210.0)
SOURCE_FREQUENCIES_HZ = (2.0, 20.0, 100.0)
TOTAL_LENGTH_UM = 840.0
LEAF_PATH_LENGTH_UM = 360.0
LEAF_GAINS = np.array((1.0, 0.70, -0.55, 0.35), dtype=float)
DRIVE_NA = 0.05
MEASUREMENT_NOISE_FRACTION = 0.003
N_COMPONENTS = 3
TEMPORAL_LAG = 16  # 8 ms
LEARNING_RATE = 2e-4
EPOCHS = 4


def ar2_oscillator(
    frequency_hz: float,
    n: int,
    rng: np.random.Generator,
    radius: float = 0.997,
) -> np.ndarray:
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
        [ar2_oscillator(freq, n, rng) for freq in SOURCE_FREQUENCIES_HZ]
    )


def branch_state_dataset(
    bifurcation_um: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(40_000 + seed)
    sources = make_sources(N_SAMPLES, rng)
    mixture = sources.sum(axis=1) / np.sqrt(sources.shape[1])

    lengths = fixed_leaf_path_allocation(
        bifurcation_um,
        total_length_um=TOTAL_LENGTH_UM,
        leaf_path_length_um=LEAF_PATH_LENGTH_UM,
    )
    tree = build_passive_tree(lengths, PassiveCableParams())

    current = np.zeros((N_SAMPLES, tree.n_nodes), dtype=float)
    for section, gain in zip(LEAF_SECTIONS, LEAF_GAINS):
        current[:, tree.section_end_nodes[section]] += DRIVE_NA * gain * mixture

    probes = tree.branch_probe_nodes
    voltage = simulate_distributed_currents(
        tree,
        current,
        record_nodes=probes,
        dt_ms=DT_MS,
    )
    noise_scale = MEASUREMENT_NOISE_FRACTION * float(np.std(voltage))
    if noise_scale:
        voltage += noise_scale * rng.normal(size=voltage.shape)

    return voltage[BURN:], sources[BURN:]


def score_outputs(outputs: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    recovery, _ = recovery_score(outputs, truth)
    duplication = mean_pairwise_abs_correlation(outputs)
    return recovery, duplication


def run_one(bifurcation_um: float, seed: int) -> dict[str, float]:
    branch_state, truth = branch_state_dataset(bifurcation_um, seed)
    q, _ = whiten_population_input(branch_state, N_COMPONENTS)

    temporal_samples = temporal_coincidence_samples(q, TEMPORAL_LAG)

    # Independent Oja units: same input statistics, no competition.
    w_oja = independent_oja_population(
        temporal_samples,
        N_COMPONENTS,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        seed=seed,
    )
    oja_outputs = q @ w_oja.T
    oja_recovery, oja_duplication = score_outputs(oja_outputs, truth)

    # Sanger/GHA: Oja-like update plus lateral competition.
    w_temporal = sanger_population(
        temporal_samples,
        N_COMPONENTS,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        seed=seed,
    )
    temporal_outputs = q @ w_temporal.T
    temporal_recovery, temporal_duplication = score_outputs(temporal_outputs, truth)

    # Destroy time relation while preserving the branch-state marginals.
    shuffled = shuffled_temporal_coincidence_samples(
        q,
        TEMPORAL_LAG,
        np.random.default_rng(90_000 + seed),
    )
    w_shuffled = sanger_population(
        shuffled,
        N_COMPONENTS,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        seed=seed,
    )
    shuffled_outputs = q @ w_shuffled.T
    shuffled_recovery, shuffled_duplication = score_outputs(shuffled_outputs, truth)

    # Competition without the temporal contrast.
    w_zero = sanger_population(
        q,
        N_COMPONENTS,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        seed=seed,
    )
    zero_outputs = q @ w_zero.T
    zero_recovery, zero_duplication = score_outputs(zero_outputs, truth)

    # Mature batch matrix reference.
    batch_outputs = amuse(branch_state, N_COMPONENTS, lag=TEMPORAL_LAG)
    batch_recovery, batch_duplication = score_outputs(batch_outputs, truth)

    return {
        "seed": seed,
        "bifurcation_length_um": bifurcation_um,
        "independent_oja_recovery": oja_recovery,
        "independent_oja_duplication": oja_duplication,
        "sanger_temporal_recovery": temporal_recovery,
        "sanger_temporal_duplication": temporal_duplication,
        "shuffled_temporal_recovery": shuffled_recovery,
        "shuffled_temporal_duplication": shuffled_duplication,
        "zero_lag_sanger_recovery": zero_recovery,
        "zero_lag_sanger_duplication": zero_duplication,
        "batch_amuse_recovery": batch_recovery,
        "batch_amuse_duplication": batch_duplication,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key not in {"seed", "bifurcation_length_um"}
    }


def run() -> dict:
    raw: dict[str, list[dict[str, float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for bif in BIFURCATION_LENGTHS_UM:
        rows = [run_one(bif, seed) for seed in SEEDS]
        key = f"{bif:g}"
        raw[key] = rows
        summary[key] = summarize(rows)

    short = summary["40"]
    long = summary["210"]
    gates = {
        "independent_oja_collapses": bool(
            long["independent_oja_duplication"] > 0.95
        ),
        "lateral_competition_diversifies": bool(
            long["sanger_temporal_duplication"] < 0.05
        ),
        "temporal_relation_matters": bool(
            long["sanger_temporal_recovery"]
            > long["shuffled_temporal_recovery"] + 0.05
        ),
        "online_rule_near_batch_amuse": bool(
            long["sanger_temporal_recovery"]
            > long["batch_amuse_recovery"] - 0.03
        ),
        "geometry_modulates_learned_specialization": bool(
            long["sanger_temporal_recovery"]
            > short["sanger_temporal_recovery"] + 0.05
        ),
    }

    return {
        "config": {
            "dt_ms": DT_MS,
            "n_samples": N_SAMPLES,
            "burn": BURN,
            "seeds": list(SEEDS),
            "bifurcation_lengths_um": list(BIFURCATION_LENGTHS_UM),
            "source_frequencies_hz": list(SOURCE_FREQUENCIES_HZ),
            "temporal_lag_samples": TEMPORAL_LAG,
            "temporal_lag_ms": TEMPORAL_LAG * DT_MS,
            "n_components": N_COMPONENTS,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_whitening_convenience": True,
        },
        "summary": summary,
        "gates": gates,
        "raw": raw,
    }


def print_result(result: dict) -> None:
    print("Gate 5 — POPULATION TEMPORAL SPECIALIZATION\n")
    print(
        "bif   Oja rec  Oja dup   Sanger rec  Sanger dup   shuffled   AMUSE"
    )
    print("-" * 77)
    for bif in BIFURCATION_LENGTHS_UM:
        s = result["summary"][f"{bif:g}"]
        print(
            f"{bif:3.0f}   "
            f"{s['independent_oja_recovery']:.4f}   "
            f"{s['independent_oja_duplication']:.4f}     "
            f"{s['sanger_temporal_recovery']:.4f}      "
            f"{s['sanger_temporal_duplication']:.4f}       "
            f"{s['shuffled_temporal_recovery']:.4f}     "
            f"{s['batch_amuse_recovery']:.4f}"
        )
    print("\ngates:", result["gates"])


def main() -> None:
    result = run()
    print_result(result)
    out = Path("results/gate5_population_specialization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
