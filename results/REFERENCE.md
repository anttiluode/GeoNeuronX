# First development receipts

These are deterministic development runs produced while building the initial gates. The thresholds and parameters were chosen during implementation, so treat them as calibration receipts, **not preregistered confirmatory evidence**.

## Gate 0 — length is a temporal coordinate

Five seeds per path-length span, 20,000 samples per seed.

| length span | mean rank | condition | PCA recovery | Oja best one-source corr | AMUSE recovery | digital random + AMUSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 3.14e12 | 0.2596 | 0.7695 | 0.2709 | 0.9998 |
| 0.05 | 3.00 | 1303.63 | 0.5510 | 0.7690 | 0.6441 | 0.9998 |
| 0.10 | 3.00 | 323.57 | 0.6271 | 0.7677 | 0.7445 | 0.9998 |
| 0.20 | 3.00 | 78.62 | 0.7617 | 0.7622 | 0.9258 | 0.9998 |
| 0.40 | 3.00 | 17.67 | 0.8009 | 0.7408 | 0.9940 | 0.9998 |
| 0.80 | 3.00 | 3.31 | 0.7641 | 0.6871 | **0.9996** | 0.9998 |
| 1.20 | 3.00 | 1.57 | 0.7609 | 0.8169 | **0.9998** | 0.9998 |

Development gate: **PASS**.

Interpretation: widening the path-length distribution changes the morphology-generated transfer matrix from redundant/ill-conditioned to well-conditioned. Once the basis is good enough, AMUSE uses lag-1 memory to recover the three hidden processes almost perfectly. The digital random-matrix attacker is equally excellent, so the result supports **geometry as a physical generator of useful temporal coordinates**, not geometry as a superior algorithm.

## Gate 1 — AIS homeostatic neck

Five seeds. Target output event rate = 0.10.

| dendritic/somatic load | fixed threshold rate | adaptive AIS rate | final effective threshold |
|---:|---:|---:|---:|
| 0.50 | 0.0010 | 0.0991 | 0.5569 |
| 1.00 | 0.0949 | 0.0989 | 1.1261 |
| 1.50 | 0.1981 | 0.0989 | 1.7222 |
| 2.00 | 0.2648 | 0.0989 | 2.3132 |

Mean absolute deviation from target:

```text
fixed threshold   0.09173
adaptive AIS      0.00105
```

Development gate: **PASS**.

Interpretation: a narrow event output can be held near a stable activity budget while upstream analog drive changes by 4×. This is only a functional homeostasis abstraction; it is not a model of real AIS position, length, channel density or electrogenesis.
