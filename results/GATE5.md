# Gate 5 — population temporal specialization

Development receipt, not preregistered confirmatory evidence.

Gate 5 asks whether several outputs receiving the same dynamical branch state can divide a temporal world among themselves without source labels.

The passive seven-branch tree is the same fixed-material / fixed-leaf-path setup used in Gates 3-4. Three independent AR(2) sources centered on 2, 20 and 100 Hz are summed into one scalar mixture and injected at the four distal leaves with fixed gains. Seven branch-midpoint voltages form the internal state seen by the learning population.

The bridge from Tuesday's AMUSE result to local-learning abstractions is:

```text
branch state x(t)
      ↓
batch whitening q(t)          <-- still a convenience / caveat
      ↓
u(t) = [q(t) + q(t-tau)] / sqrt(2)
      ↓
Oja / Sanger generalized Hebbian learning
      ↓
population output axes
```

For whitened `q`,

```text
Cov[u] = I + C_tau_sym
```

so the temporal-coincidence samples have the same eigenvectors as the symmetrized AMUSE lag operator. Sanger's generalized Hebbian algorithm therefore gives a simple online Oja-like route to the same temporal specialization problem.

No source labels enter learning. Labels are used only after training to score recovery.

## Arms

```text
Independent Oja copies
    same temporal-coincidence input
    NO lateral competition

Temporal Sanger population
    same input
    generalized Hebbian lateral subtraction

Time-shuffled Sanger
    same branch-state marginals
    lag relation destroyed

Zero-lag Sanger
    competition without temporal contrast

Batch AMUSE
    mature matrix reference
```

Five seeds, 16,000 samples, 2,000-sample burn, `tau = 8 ms`, three outputs.

## Development result

| bifurcation allocation | independent Oja recovery | Oja output duplication | **temporal Sanger recovery** | Sanger duplication | shuffled-time recovery | zero-lag Sanger | batch AMUSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 µm  | 0.3330 | 0.9996 | **0.7705** | 0.0139 | 0.6869 | 0.6654 | 0.7911 |
| 120 µm | 0.3293 | 0.9999 | **0.8004** | 0.0111 | 0.7272 | 0.7461 | 0.8055 |
| 210 µm | 0.3311 | 0.9996 | **0.8440** | 0.0129 | 0.7622 | 0.7811 | 0.8479 |

Recovery is permutation/sign-invariant mean absolute correlation to the three hidden sources. Duplication is mean absolute zero-lag correlation between distinct population outputs.

## What survived

### 1. Oja alone collapses the population

Three independent Oja units all see the same temporal statistics. They converge to essentially the same strongest direction:

```text
output duplication ≈ 0.9996
source recovery     ≈ 0.33
```

This is the expected failure and it matters. Self-normalization is not enough to make a population divide the world.

### 2. Lateral competition creates distinct temporal listeners

With Sanger's generalized Hebbian subtraction, output duplication falls to about `0.01` while source recovery rises to `0.77-0.84`.

At the longest internal bifurcation allocation:

```text
independent Oja      recovery 0.3311   duplication 0.9996
Sanger + competition recovery 0.8440   duplication 0.0129
```

The population has actually specialized into different temporal axes rather than three copies of one listener.

### 3. The temporal relation is doing work

Shuffle the past samples before constructing the temporal-coincidence input and recovery falls:

```text
210 µm morphology:
real temporal relation   0.8440
shuffled temporal past   0.7622
```

Zero-lag competition reaches `0.7811`, also below the temporal rule.

So the gain is not only "orthogonalize three outputs." The lag relation contributes useful source identity.

### 4. The online rule nearly reaches batch AMUSE

At `210 µm`:

```text
online temporal Sanger   0.8440
batch AMUSE              0.8479
```

This is the nicest result in the gate. A very ordinary generalized Hebbian population, when fed a transformed temporal-coincidence sample, approaches the mature batch eigensolver.

The interpretation is intentionally modest:

> **AMUSE-like temporal specialization can be expressed as an Oja/Sanger-style population learning problem once the unit's dynamical state has made history available as coordinates.**

That is a mathematical bridge, not a claim that real synapses implement this exact algebra.

### 5. Geometry still modulates what the population can learn

Temporal Sanger recovery rises:

```text
0.7705 -> 0.8004 -> 0.8440
```

as the same `840 µm` cable budget is moved into longer internal bifurcation sections while total soma-to-leaf path length remains `360 µm`.

So geometry is not merely an inert prelude to the learning rule. It changes the branch-state basis on which the population learns.

## The important caveat: whitening is still global

This gate is not yet a fully local end-to-end neuron model.

Before Oja/Sanger learning, the seven branch coordinates are batch-whitened down to three dimensions. That is a major convenience. The Sanger stage itself is online and uses only feedforward activity plus lower-order population outputs, but the whitening stage has not earned a local implementation yet.

Therefore the current claim is not:

> a biological population performs AMUSE locally.

It is:

> **given normalized branch-state coordinates, Oja-like learning plus lateral competition can learn distinct temporal processes using the same lag structure that batch AMUSE exploits.**

## Why this matters for the dynamical-neuron abstraction

The emerging computational scheme is now:

```text
WORLD MIXTURE
     ↓
DYNAMICAL DENDRITIC STATE
history materialized into local coordinates
     ↓
LOCAL / POPULATION LEARNING
self-normalization + lateral competition
     ↓
DIFFERENT UNITS SPECIALIZE
     ↓
SOMA / AIS EVENT STREAMS
     ↓
next population
```

The matrix `W` remains useful notation, but it is no longer the conceptual center. The center is a stateful tempo-spatial substrate that supplies the coordinates on which local adaptive rules operate.

## Next attacks

1. **Online whitening.** Replace batch PCA whitening with a streaming decorrelation/homeostasis stage. If this fails, the local-learning story remains incomplete.
2. **SOBI-like multi-lag learning.** Present several temporal-coincidence timescales and ask whether specialization is more robust when one lag is ambiguous.
3. **Conductance feedback.** Gate 5A remains open: put the branch nonlinearity inside the cable dynamics rather than after recording.
4. **Task utility.** Freeze the specialized population and ask whether its outputs improve prediction, anomaly detection, control, or extraction after sensor/morphology changes.
5. **Matched adaptive-filter attacker.** Compare against an ordinary online filter bank with the same state and parameter budget.

The most immediate kill condition is simple: if online whitening + a standard adaptive filter bank reproduces everything more cleanly, call the dynamical neuron a physical implementation bias rather than a new algorithm.
