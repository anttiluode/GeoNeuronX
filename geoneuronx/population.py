"""Population-learning helpers for the GeoNeuronX dynamical-neuron abstraction.

Gate 5 asks whether several units receiving the same dynamical branch state can
specialize without source labels.

This module deliberately bridges two mature abstractions rather than claiming a
new biological rule:

* Oja/Sanger-style generalized Hebbian learning supplies an online/local-ish
  weight update with lateral competition.
* AMUSE says that, after whitening, the useful axes are eigenvectors of a
  symmetrized lagged covariance.

For whitened branch state q(t), define a temporal coincidence sample

    u(t) = (q(t) + q(t-tau)) / sqrt(2)

Then

    Cov[u] = I + C_tau_sym

so ``u`` has the same eigenvectors as the AMUSE lag operator.  Feeding these
samples into Sanger's generalized Hebbian algorithm is therefore a small bridge
from Oja-like local learning to AMUSE-like temporal specialization.

Important caveat: the current gate still uses batch whitening as a convenience.
It is not yet an end-to-end local neural implementation of whitening + BSS.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def whiten_population_input(
    x: Array,
    n_components: int,
    eps: float = 1e-10,
) -> tuple[Array, Array]:
    """Center and PCA-whiten a multichannel branch-state trajectory."""
    data = np.asarray(x, dtype=float)
    if data.ndim != 2:
        raise ValueError("x must have shape (time, channel)")
    if n_components < 1:
        raise ValueError("n_components must be positive")

    x0 = data - data.mean(axis=0, keepdims=True)
    cov = (x0.T @ x0) / len(x0)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if len(values) == 0 or values[0] <= 0:
        raise ValueError("input covariance is degenerate")

    keep = values > eps * values[0]
    values = values[keep]
    vectors = vectors[:, keep]
    k = min(int(n_components), len(values))
    values = values[:k]
    vectors = vectors[:, :k]
    whitening = vectors @ np.diag(1.0 / np.sqrt(values))
    return x0 @ whitening, whitening


def temporal_coincidence_samples(q: Array, lag: int) -> Array:
    """Return ``(q[t] + q[t-lag]) / sqrt(2)`` temporal coincidence samples."""
    data = np.asarray(q, dtype=float)
    if data.ndim != 2:
        raise ValueError("q must have shape (time, feature)")
    if lag < 1 or len(data) <= lag:
        raise ValueError("lag must satisfy 1 <= lag < len(q)")
    return (data[lag:] + data[:-lag]) / np.sqrt(2.0)


def shuffled_temporal_coincidence_samples(
    q: Array,
    lag: int,
    rng: np.random.Generator,
) -> Array:
    """Time-shuffle control preserving marginal branch-state statistics."""
    data = np.asarray(q, dtype=float)
    if data.ndim != 2:
        raise ValueError("q must have shape (time, feature)")
    if lag < 1 or len(data) <= lag:
        raise ValueError("lag must satisfy 1 <= lag < len(q)")
    past = data[:-lag].copy()
    rng.shuffle(past, axis=0)
    return (data[lag:] + past) / np.sqrt(2.0)


def sanger_population(
    samples: Array,
    n_outputs: int,
    lr: float = 2e-4,
    epochs: int = 4,
    seed: int = 0,
) -> Array:
    """Learn ordered population axes with Sanger's generalized Hebbian rule.

    For each sample ``u`` and output ``y = W u``::

        dw_i = eta * y_i * (u - sum_{j<=i} y_j w_j)

    The subtraction is the lateral/competition term.  Unlike several
    independent Oja units, Sanger's rule discourages every output from learning
    the same strongest direction.

    Rows are orthonormalized only once at initialization; no batch QR cleanup is
    performed during learning, so output decorrelation in the gate is produced
    by the update itself rather than by a post-hoc matrix operation.
    """
    data = np.asarray(samples, dtype=float)
    if data.ndim != 2:
        raise ValueError("samples must have shape (time, feature)")
    if not 1 <= n_outputs <= data.shape[1]:
        raise ValueError("n_outputs must be within the input dimension")
    if lr <= 0 or epochs < 1:
        raise ValueError("lr and epochs must be positive")

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(data.shape[1], n_outputs))
    q, _ = np.linalg.qr(raw)
    weights = q[:, :n_outputs].T.copy()

    for _ in range(int(epochs)):
        for u in data:
            y = weights @ u
            for i in range(n_outputs):
                reconstruction = np.sum(
                    y[: i + 1, None] * weights[: i + 1],
                    axis=0,
                )
                weights[i] += lr * y[i] * (u - reconstruction)
    return weights


def independent_oja_population(
    samples: Array,
    n_outputs: int,
    lr: float = 2e-4,
    epochs: int = 4,
    seed: int = 0,
) -> Array:
    """Several independent Oja learners with no lateral competition.

    This is the collapse attacker.  If all copies see the same statistics, each
    should tend toward the same strongest temporal-coincidence direction.
    """
    data = np.asarray(samples, dtype=float)
    if data.ndim != 2:
        raise ValueError("samples must have shape (time, feature)")
    if not 1 <= n_outputs <= data.shape[1]:
        raise ValueError("n_outputs must be within the input dimension")

    rng = np.random.default_rng(seed)
    weights = np.empty((n_outputs, data.shape[1]), dtype=float)
    for i in range(n_outputs):
        w = rng.normal(size=data.shape[1])
        w /= np.linalg.norm(w) + 1e-12
        for _ in range(int(epochs)):
            for u in data:
                y = float(w @ u)
                w += lr * y * (u - y * w)
            w /= np.linalg.norm(w) + 1e-12
        weights[i] = w
    return weights


def mean_pairwise_abs_correlation(outputs: Array) -> float:
    """Mean absolute zero-lag correlation between distinct population outputs."""
    y = np.asarray(outputs, dtype=float)
    if y.ndim != 2:
        raise ValueError("outputs must have shape (time, unit)")
    if y.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(y.T)
    tri = np.triu_indices(y.shape[1], 1)
    return float(np.mean(np.abs(corr[tri])))
