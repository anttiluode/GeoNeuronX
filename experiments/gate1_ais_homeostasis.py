"""Gate 1 — THE AIS IS AN ADAPTIVE NECK, NOT A FIXED THRESHOLD.

A morphology/load sweep drives the same latent processes through the same branch
bank.  We compare a fixed spike threshold with a minimal homeostatic threshold
controller.  The controller is only a functional abstraction of AIS plasticity:
high sustained firing makes spike initiation harder; low sustained firing makes
it easier.

The point is to separate two jobs:

    dendrite/soma: rich analog temporal computation
    AIS: adaptive conversion of that state into a narrow event stream
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoneuronx.core import (  # noqa: E402
    AdaptiveAIS,
    make_temporal_sources,
    observe_through_morphology,
    soma_from_branches,
)


def one_seed(seed: int, n: int = 20_000) -> list[dict]:
    rng = np.random.default_rng(seed)
    sources = make_temporal_sources(n, rng)
    lengths = np.linspace(0.2, 1.4, 8)
    branch, _ = observe_through_morphology(sources, lengths, rng, noise_std=0.03)
    base_soma = soma_from_branches(branch)

    target = 0.10
    fixed_threshold = 1.18
    burn = 5_000
    rows = []

    for load in (0.5, 1.0, 1.5, 2.0):
        signal = load * base_soma
        fixed = float(np.mean(signal[burn:] > fixed_threshold))

        ais = AdaptiveAIS(
            target_rate=target,
            threshold=fixed_threshold,
            learning_rate=0.01,
            rate_alpha=0.01,
        )
        spikes = np.array([ais.step(float(v)) for v in signal], dtype=float)
        adaptive = float(np.mean(spikes[burn:]))
        rows.append(
            {
                "seed": seed,
                "load": load,
                "fixed_rate": fixed,
                "adaptive_rate": adaptive,
                "final_effective_threshold": float(ais.threshold),
                "target_rate": target,
            }
        )
    return rows


def main() -> None:
    seeds = [21, 22, 23, 24, 25]
    rows = [row for seed in seeds for row in one_seed(seed)]

    print("\nGeoNeuronX Gate 1 — AIS HOMEOSTATIC NECK\n")
    print(f"{'load':>6} {'fixed rate':>12} {'adaptive':>12} {'threshold':>12}")
    print("-" * 48)

    summary = []
    for load in (0.5, 1.0, 1.5, 2.0):
        part = [r for r in rows if r["load"] == load]
        item = {
            "load": load,
            "mean_fixed_rate": float(np.mean([r["fixed_rate"] for r in part])),
            "mean_adaptive_rate": float(np.mean([r["adaptive_rate"] for r in part])),
            "mean_final_threshold": float(
                np.mean([r["final_effective_threshold"] for r in part])
            ),
        }
        summary.append(item)
        print(
            f"{load:6.2f} {item['mean_fixed_rate']:12.4f} "
            f"{item['mean_adaptive_rate']:12.4f} {item['mean_final_threshold']:12.4f}"
        )

    target = 0.10
    fixed_dev = float(np.mean([abs(s["mean_fixed_rate"] - target) for s in summary]))
    adaptive_dev = float(
        np.mean([abs(s["mean_adaptive_rate"] - target) for s in summary])
    )
    passed = adaptive_dev < 0.04 and adaptive_dev < 0.35 * fixed_dev

    receipt = {
        "gate": "G1_AIS_HOMEOSTATIC_NECK",
        "target_rate": target,
        "fixed_mean_abs_deviation": fixed_dev,
        "adaptive_mean_abs_deviation": adaptive_dev,
        "pass": bool(passed),
        "summary": summary,
        "rows": rows,
        "interpretation": (
            "A narrow output event channel can be normalized across large changes in dendritic/somatic "
            "drive by adapting an effective spike-initiation threshold. This is only a functional AIS "
            "abstraction; it does not model AIS position, channel density, or cable biophysics."
        ),
    }

    out = ROOT / "results" / "gate1_reference.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nfixed mean |rate-target|    = {fixed_dev:.5f}")
    print(f"adaptive mean |rate-target| = {adaptive_dev:.5f}")
    print(f"PASS = {passed}")
    print(f"receipt: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
