"""Small helpers for the GeoNeuronX dynamical-neuron abstraction.

The point of this module is not to pretend that these equations are molecular
neuroscience.  It exposes three deliberately boring operations needed to test
the abstraction:

1. a stateful physical front end can turn recent history into simultaneous
   branch coordinates;
2. branch-local nonlinearities can act on those coordinates before collapse at
   the soma;
3. a simple learned soma/readout can then use the resulting feature map.

Matched digital delay and polynomial feature maps remain explicit attackers.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

Array = np.ndarray


def telegraph_signal(
    n: int,
    rng: np.random.Generator,
    hold_samples: int = 4,
    noise_std: float = 0.05,
) -> Array:
    """Random +/-1 piecewise-constant drive with optional analog noise."""
    if n < 1:
        raise ValueError("n must be positive")
    if hold_samples < 1:
        raise ValueError("hold_samples must be positive")
    values = rng.choice((-1.0, 1.0), size=(n + hold_samples - 1) // hold_samples)
    x = np.repeat(values, hold_samples)[:n].astype(float)
    if noise_std:
        x += float(noise_std) * rng.normal(size=n)
    return x


def delayed_xor_target(signal: Array, start: int, lag: int) -> Array:
    """Return +/-1 delayed XOR labels for ``signal[t]`` and ``signal[t-lag]``.

    This is intentionally a nonlinear temporal task.  A linear FIR has the
    relevant history but cannot represent XOR by a linear readout.  Adding a
    nonlinear feature map should therefore be necessary.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1-D")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if start < lag or start >= len(x):
        raise ValueError("start must satisfy lag <= start < len(signal)")
    previous = x[start - lag : len(x) - lag]
    current = x[start:]
    xor = (current > 0) ^ (previous > 0)
    return np.where(xor, 1.0, -1.0)


def rectified_threshold_basis(
    x: Array,
    train_rows: int,
    thresholds: Iterable[float] = (-1.5, -0.75, 0.0, 0.75, 1.5),
) -> Array:
    """Apply local thresholded nonlinearities to each observed coordinate.

    Each channel is standardized using only the training prefix, then copied
    through several ReLU-like thresholds.  In the GeoNeuronX abstraction this
    stands in for multiple local nonlinear subunits acting on differently
    filtered branch voltages *before* soma aggregation.

    It is deliberately not called NMDA.  A later gate may replace this with a
    conductance-based branch mechanism.
    """
    data = np.asarray(x, dtype=float)
    if data.ndim != 2:
        raise ValueError("x must have shape (time, channel)")
    if not 1 <= train_rows <= len(data):
        raise ValueError("train_rows must index a non-empty training prefix")
    cuts = tuple(float(v) for v in thresholds)
    if not cuts:
        raise ValueError("thresholds must be non-empty")

    mean = data[:train_rows].mean(axis=0, keepdims=True)
    scale = data[:train_rows].std(axis=0, keepdims=True) + 1e-9
    z = (data - mean) / scale
    return np.concatenate([np.maximum(z - cut, 0.0) for cut in cuts], axis=1)


def quadratic_feature_map(x: Array) -> Array:
    """Return linear coordinates plus all quadratic products ``x_i*x_j``.

    This is the boring nonlinear digital attacker used in Gate 4.  For ``p``
    input coordinates it yields ``p + p(p+1)/2`` features.
    """
    data = np.asarray(x, dtype=float)
    if data.ndim != 2:
        raise ValueError("x must have shape (time, channel)")
    blocks: list[Array] = [data]
    for i in range(data.shape[1]):
        for j in range(i, data.shape[1]):
            blocks.append((data[:, i] * data[:, j])[:, None])
    return np.concatenate(blocks, axis=1)


def ridge_readout_metrics(
    x: Array,
    target: Array,
    train_fraction: float = 0.60,
    ridge: float = 1e-2,
) -> dict[str, float]:
    """Fit a train-only standardized linear readout and score held-out data.

    ``target`` is expected to be a continuous signal or +/-1 labels.  For the
    latter, ``accuracy`` is the fraction with the correct sign.  ``corr`` is
    absolute Pearson correlation and ``nmse`` is normalized mean-squared error.
    """
    data = np.asarray(x, dtype=float)
    y = np.asarray(target, dtype=float)
    if data.ndim != 2 or y.ndim != 1 or len(data) != len(y):
        raise ValueError("x must be (time, feature) and target must be matching 1-D")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    split = int(len(y) * train_fraction)
    if split < 2 or len(y) - split < 2:
        raise ValueError("not enough samples for train/test split")

    mean = data[:split].mean(axis=0, keepdims=True)
    scale = data[:split].std(axis=0, keepdims=True) + 1e-9
    z = (data - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])

    penalty = float(ridge) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design[:split].T @ design[:split] + penalty,
        design[:split].T @ y[:split],
    )
    pred = design[split:] @ weights
    truth = y[split:]

    if np.std(pred) < 1e-12 or np.std(truth) < 1e-12:
        corr = 0.0
    else:
        corr = float(abs(np.corrcoef(pred, truth)[0, 1]))
    accuracy = float(np.mean((pred >= 0.0) == (truth >= 0.0)))
    nmse = float(np.mean((pred - truth) ** 2) / (np.mean(truth**2) + 1e-12))
    return {"corr": corr, "accuracy": accuracy, "nmse": nmse}
