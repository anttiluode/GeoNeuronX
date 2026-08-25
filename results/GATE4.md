# Gate 4 — the dynamical neuron

Development receipt, not preregistered confirmatory evidence.

Gate 4 is the first experiment in GeoNeuronX that stops asking only whether morphology exposes sources and asks whether the proposed unit can perform a genuinely **nonlinear temporal computation**.

The abstraction is:

```text
recent input history
        ↓
passive dendritic dynamics
        ↓
simultaneous branch-local state
        ↓
local nonlinear subunits
        ↓
linear learned soma/readout
        ↓
AIS / event channel later
```

The local nonlinearities are deliberately simple rectified thresholds applied to branch voltages. They are **not** called NMDA and they do not feed current back into the cable. This gate tests the computational placement of nonlinearity before soma collapse. A conductance-based branch gate is a later experiment.

## Task: delayed XOR

One noisy +/-1 telegraph signal is copied through fixed gains to four distal leaf synapses. The target at time `t` is:

```text
+1  if sign(x[t]) != sign(x[t - 16 ms])
-1  otherwise
```

The positive class fraction is approximately `0.499`.

This task is useful because a linear filter may contain the relevant history and still cannot solve XOR with a linear output. A nonlinear feature map over that history is required.

As in Gate 3, every morphology has the same:

- seven-section binary topology;
- diameter and passive membrane constants;
- compartment count;
- total dendritic cable length: `840 µm`;
- total dendritic membrane area;
- soma-to-leaf path length: `360 µm`.

Only cable allocation relative to the branch points changes. The two internal bifurcation sections are tested at `40`, `120`, and `210 µm`.

## Arms

```text
soma linear
    one soma voltage -> linear readout

soma nonlinear
    soma voltage -> threshold bank -> linear readout

branch linear
    seven branch midpoint voltages -> linear readout

branch local nonlinear
    EACH branch voltage -> threshold bank -> linear soma/readout

digital FIR linear
    explicit [0,1,2,4,8,16,32] ms history -> linear readout

digital FIR quadratic
    same history + all pairwise products -> linear readout
```

With seven FIR coordinates, the quadratic attacker has `7 + 7*8/2 = 35` features. The branch-local arm also has `7 branches * 5 thresholds = 35` nonlinear features, so the successful digital attacker is not hiding behind an enormous feature count.

## Five-seed development result

Accuracy is held-out sign accuracy. Chance is about `0.5`.

| bifurcation µm | soma linear | soma nonlinear | branch linear | **branch local nonlinear** | FIR linear | **FIR quadratic** |
|---:|---:|---:|---:|---:|---:|---:|
| 40  | 0.4986 | 0.5115 | 0.5008 | **0.5792** | 0.4990 | **1.0000** |
| 120 | 0.4984 | 0.5104 | 0.5012 | **0.5849** | 0.4990 | **1.0000** |
| 210 | 0.4980 | 0.5094 | 0.5015 | **0.5935** | 0.4990 | **1.0000** |

Corresponding held-out target correlations for the branch-local arm are:

```text
40 µm   0.1878
120 µm  0.2122
210 µm  0.2345
```

while the quadratic digital delay attacker reaches `0.9974`.

## What survived

### 1. State/memory alone is insufficient for this task

Both the branch-linear arm and an explicit digital FIR with the exact `16 ms` target lag remain at chance. They have temporal information, but the requested relation is nonlinear.

So the primitive cannot stop at:

```text
history -> coordinates -> weighted sum
```

For this task it needs:

```text
history -> coordinates -> LOCAL NONLINEAR MAP -> weighted sum
```

### 2. The placement of nonlinearity matters

Putting the same threshold idea only after the branch state has collapsed to the one-dimensional soma trace gives about `0.509` accuracy.

Applying nonlinearities locally to the seven different history-bearing branch coordinates before collapse reaches `0.593` in the longest-bifurcation morphology.

That is the cleanest computational distinction in this gate:

> **different filtered histories can be acted on nonlinearly before they are summed away.**

### 3. Geometry modulates the nonlinear feature map

Under fixed total cable and fixed leaf path length, increasing internal bifurcation allocation moves branch-local nonlinear accuracy from `0.579 -> 0.593` and correlation from `0.188 -> 0.235`.

This is not a huge effect, but it is in the direction predicted by Gates 2-3: geometry changes the temporal projections presented to the local nonlinearities, so changing geometry changes the nonlinear computation.

### 4. Ordinary digital computation destroys any superiority claim

The 35-feature quadratic FIR attacker solves the task perfectly (`1.000` accuracy) with essentially perfect target correlation.

So Gate 4 does **not** show that dendritic geometry is a better way to compute delayed XOR.

It shows something narrower and more useful for the abstraction:

> **A stateful branched substrate plus local nonlinear subunits is already a nonlinear temporal computing primitive.**

A conventional digital delay/polynomial feature map can implement the same class of computation far more directly.

## Why call it a dynamical neuron?

McCulloch-Pitts preserves weighted integration plus thresholding:

```text
y = H(w^T x - theta)
```

The GeoNeuronX abstraction preserves a different set of features:

```text
input history
    ↓
stateful geometry / temporal filtering
    ↓
branch-local nonlinear subunits
    ↓
learned soma integration
    ↓
adaptive AIS eventization
```

The crucial difference is that the coordinates presented to the learned weights are **generated by the unit's own dynamics**. The unit therefore has state even when the final readout is memoryless.

A compact mathematical form is:

```text
r_j(t) = (h_j * x)(t)
u_j(t) = phi_j(r_j(t))
z(t)   = sum_j w_j u_j(t)
y(t)   = AIS(z(t), slow_homeostatic_state)
```

where morphology influences the family of filters `h_j`.

That is the current candidate **dynamical-neuron abstraction**. It is not a claim that biology implements these exact equations.

## Next gates

Two different questions should now be kept separate.

### Gate 5A — conductance-based local nonlinearity

Replace the post-hoc threshold subunits with a transparent voltage-dependent branch current and then, only if that survives, with an NMDA-like conductance model.

Ask whether nonlinear feedback inside the cable changes the temporal state in ways that a readout-only nonlinearity does not.

### Gate 5B — learning and population specialization

The current soma/readout is supervised ridge regression. Replace that convenience with local/online learning abstractions:

```text
Oja / normalized Hebb
prediction-error plasticity
lateral covariance suppression / competition
```

Several dynamical neurons receive the same mixtures. The test is whether they specialize into different useful temporal processes without global source labels.

That is where the source-separation mathematics returns to the architecture.

Raw development receipt: `results/gate4_dynamical_neuron.json`.
