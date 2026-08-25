# Gate 2 — passive cable geometry under fixed material

Development receipt, not preregistered confirmatory evidence.

Gate 2 removes the analytic `H(f,L)` used in Gate 0. The model is now a passive compartmental cable with conventional membrane parameters (`Cm=1 µF/cm²`, `Rm=20 kΩ·cm²`, `Ra=150 Ω·cm`) and a fixed binary topology. Every morphology has the same branch count, compartment count, diameter, total dendritic length (`840 µm`) and total dendritic membrane area. Only the allocation of cable length among sections changes.

The passive state obeys

```text
C dV/dt = -G V + I
```

and branch transfer responses are measured by solving the compartmental admittance matrix, not by evaluating a hand-written frequency response.

## Narrow-band dynamical sources

Five seeds, 16,000 samples, dt `0.5 ms`. Three independent narrow-band AR(2) processes centered on 2, 20 and 100 Hz are summed into one scalar drive. Seven local branch voltages are recorded. AMUSE uses one lagged covariance at `tau = 8 ms`.

| heterogeneity | leaf path SD µm | cond(H) | PCA | AMUSE | oracle corr | oracle NMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.00 | 601.3 | 0.7217 | 0.7356 | 0.7848 | 0.3788 |
| 0.2 | 16.95 | 335.9 | 0.7766 | 0.8200 | 0.8452 | 0.2818 |
| 0.4 | 33.90 | 223.4 | 0.8005 | 0.8493 | 0.8677 | 0.2450 |
| 0.6 | 50.85 | 171.5 | 0.8146 | 0.8615 | 0.8772 | 0.2293 |
| 0.8 | 67.80 | 144.2 | 0.8242 | **0.8666** | **0.8810** | **0.2231** |

The same membrane budget becomes a better-conditioned temporal observation basis as path lengths diversify. In this narrow-band source class, both blind AMUSE recovery and supervised linear recoverability improve substantially.

## Boundary: broad AR(1) sources

The effect is not generic.

```text
AMUSE: 0.4191 -> 0.4123
oracle: 0.6297 -> 0.6183
```

for heterogeneity `0.0 -> 0.8`.

A passive low-pass tree is therefore not automatically useful for arbitrary colored processes. Source dynamics must interact usefully with the cable transfer diversity.

## Boring attacker

A seven-state explicit digital RC filter bank reaches:

```text
oscillatory oracle corr = 0.9548
oscillatory oracle NMSE = 0.0897
```

versus the best cable values `0.8810` and `0.2231`.

So Gate 2 does **not** show a computational advantage over normal DSP. It shows that geometry can physically instantiate a useful temporal basis under a conserved material budget.

## Sanity checks

```text
fixed total length relative range = 5.4e-16
fixed total area relative range   = 5.1e-16
passive reciprocity error         = 3.3e-16
all five development gates        = PASS
10/10 repository tests            = PASS
```

Raw receipt: `results/gate2_passive_cable_geometry.json`.
