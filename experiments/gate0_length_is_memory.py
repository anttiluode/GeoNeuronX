"""Gate 0 — LENGTH IS A TEMPORAL COORDINATE.

Question
--------
If branch path length changes the transfer coefficients seen by different
temporal frequencies, does *diversity of path length* create a better basis for
recovering latent dynamical sources?

This is intentionally not a biological neuron simulation.  It is the smallest
mathematical test of the proposed reason for length.

Controls
--------
* equal-length bank: same number of branches, but redundant transfer rows;
* PCA: zero-lag variance basis;
* Oja: online self-limiting PCA-like learner;
* AMUSE(lag=1): uses one lag covariance;
* digital random matrix: proves geometry is not uniquely privileged.

Pass criterion for a development run
------------------------------------
For wide length diversity (span >= 0.8), AMUSE mean source recovery must exceed
0.95 and beat PCA by at least 0.10 across seeds.  Equal-length morphology must
remain rank-deficient.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoneuronx.core import (  # noqa: E402
    amuse,
    best_single_source_correlation,
    condition_number,
    effective_rank,
    make_temporal_sources,
    observe_through_morphology,
    oja_first_component,
    pca_scores,
    recovery_score,
)


def one_case(seed: int, span: float, n: int = 20_000) -> dict:
    rng = np.random.default_rng(seed)
    sources = make_temporal_sources(n, rng)

    center = 0.8
    lengths = np.linspace(center - span / 2.0, center + span / 2.0, 8)
    x, A = observe_through_morphology(sources, lengths, rng, noise_std=0.03)

    pca = pca_scores(x, 3)
    am = amuse(x, 3, lag=1)
    pca_score, _ = recovery_score(pca, sources)
    amuse_score, _ = recovery_score(am, sources)

    oja_y, _ = oja_first_component(x, lr=2e-4, seed=seed)
    oja_best = best_single_source_correlation(oja_y[n // 2 :], sources[n // 2 :])

    # Boring attacker: an arbitrary full-rank digital observation matrix.
    B = rng.normal(size=(8, 3))
    x_digital = sources @ B.T + 0.03 * rng.normal(size=(n, 8))
    digital_score, _ = recovery_score(amuse(x_digital, 3, lag=1), sources)

    return {
        "seed": seed,
        "span": float(span),
        "lengths": lengths.tolist(),
        "effective_rank": effective_rank(A),
        "condition_number": condition_number(A),
        "pca_recovery": pca_score,
        "oja_best_single_source_corr": oja_best,
        "amuse_recovery": amuse_score,
        "digital_random_amuse_recovery": digital_score,
    }


def main() -> None:
    spans = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20]
    seeds = [11, 12, 13, 14, 15]

    rows = [one_case(seed, span) for span in spans for seed in seeds]

    summary = []
    print("\nGeoNeuronX Gate 0 — LENGTH IS A TEMPORAL COORDINATE\n")
    print(
        f"{'span':>6} {'rank':>6} {'cond':>11} {'PCA':>9} {'Oja1':>9} "
        f"{'AMUSE':>9} {'digital':>9}"
    )
    print("-" * 72)

    for span in spans:
        part = [r for r in rows if r["span"] == span]
        item = {
            "span": span,
            "mean_rank": float(np.mean([r["effective_rank"] for r in part])),
            "mean_condition": float(np.mean([r["condition_number"] for r in part])),
            "mean_pca": float(np.mean([r["pca_recovery"] for r in part])),
            "mean_oja_best": float(np.mean([r["oja_best_single_source_corr"] for r in part])),
            "mean_amuse": float(np.mean([r["amuse_recovery"] for r in part])),
            "mean_digital": float(
                np.mean([r["digital_random_amuse_recovery"] for r in part])
            ),
        }
        summary.append(item)
        print(
            f"{span:6.2f} {item['mean_rank']:6.2f} {item['mean_condition']:11.2f} "
            f"{item['mean_pca']:9.4f} {item['mean_oja_best']:9.4f} "
            f"{item['mean_amuse']:9.4f} {item['mean_digital']:9.4f}"
        )

    equal = next(s for s in summary if s["span"] == 0.0)
    wide = next(s for s in summary if s["span"] == 0.80)
    passed = (
        equal["mean_rank"] < 3.0
        and wide["mean_amuse"] > 0.95
        and wide["mean_amuse"] - wide["mean_pca"] > 0.10
    )

    receipt = {
        "gate": "G0_LENGTH_IS_TEMPORAL_COORDINATE",
        "development_thresholds": {
            "equal_length_rank_lt": 3.0,
            "wide_span_amuse_gt": 0.95,
            "wide_span_amuse_minus_pca_gt": 0.10,
        },
        "pass": bool(passed),
        "summary": summary,
        "rows": rows,
        "interpretation": (
            "Path-length diversity can generate a better-conditioned temporal transfer basis. "
            "AMUSE then exploits source memory to recover the hidden processes. The digital "
            "random-matrix attacker remains excellent, so this is a representability/use result, "
            "not evidence that morphology beats ordinary matrices."
        ),
    }

    out = ROOT / "results" / "gate0_reference.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nPASS = {passed}")
    print(f"receipt: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
