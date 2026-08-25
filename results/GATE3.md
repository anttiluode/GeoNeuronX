# Gate 3 — distributed synapses: geometry materializes history

Development receipt, not preregistered confirmatory evidence.

Gate 3 changes the direction of drive from Gate 2. Three independent narrow-band AR(2) sources (2, 20 and 100 Hz) are summed into **one scalar mixture**. Copies of that same mixture are injected at four distal leaf synapses with fixed gains. At any instant the synaptic drive is therefore rank one; the tree is not handed three spatial source channels.

The morphology control is stronger than Gate 2. Every morphology has:

- the same seven-section binary topology;
- the same diameter and passive membrane parameters;
- the same compartment count;
- the same total dendritic length: `840 µm`;
- the same total dendritic membrane area;
- **the same soma-to-leaf cable path length: `360 µm` for all four leaves**.

Only the location of cable relative to the branch points changes. The two internal bifurcation sections are swept from `40 -> 210 µm`; trunk and terminal sections pay for that change exactly.

Readouts are taken at distal branches, bifurcation branches, the trunk, all seven section midpoints, and the soma. Three attacks are compared:

1. **AMUSE** — blind, one lag (`8 ms`);
2. **static oracle** — supervised but memoryless linear readout;
3. **FIR oracle** — the same supervised readout after explicit digital delay taps `[0,1,2,4,8,16,32] ms`.

Five seeds, 12,000 samples per seed, 2,000-sample burn, 0.3% measurement noise.

## Main receipt

| bifurcation section µm | zone | AMUSE | static oracle | FIR oracle |
|---:|---|---:|---:|---:|
| 40 | distal | 0.7638 | 0.8118 | 0.9620 |
| 40 | bifurcation | 0.5730 | 0.7293 | 0.9614 |
| 40 | trunk | 0.6993 | 0.7440 | 0.9613 |
| 40 | all midpoints | 0.7993 | 0.8967 | 0.9625 |
| 40 | soma | — | 0.3363 | 0.9550 |
| 120 | distal | 0.8140 | 0.8380 | 0.9624 |
| 120 | bifurcation | 0.7767 | 0.8112 | 0.9620 |
| 120 | trunk | 0.6136 | 0.6984 | 0.9609 |
| 120 | all midpoints | 0.8120 | 0.8432 | 0.9623 |
| 120 | soma | — | 0.3385 | 0.9553 |
| 210 | distal | **0.8367** | **0.8516** | 0.9627 |
| 210 | bifurcation | **0.8592** | **0.8748** | 0.9625 |
| 210 | trunk | 0.5565 | 0.6863 | 0.9602 |
| 210 | all midpoints | **0.8532** | **0.9065** | 0.9630 |
| 210 | soma | — | 0.3407 | 0.9555 |

## What changed

Moving cable from trunk/terminal zones into the internal bifurcation sections changes **where temporal source information is instantaneously readable**.

From the shortest to the longest bifurcation allocation:

```text
distal AMUSE             0.7638 -> 0.8367
bifurcation AMUSE        0.5730 -> 0.8592
bifurcation static       0.7293 -> 0.8748
trunk static             0.7440 -> 0.6863
```

So the effect is not a generic increase everywhere. Temporal accessibility is **redistributed through the tree**. Length placed around the bifurcating interior makes those local branch states much easier to use as source coordinates, while the trunk becomes less informative to a memoryless reader.

## The important attacker: explicit history kills most of the geometry story

Give the readout its own digital delay line and nearly every zone reaches the same recovery:

```text
FIR oracle across distal/bifurcation/all: about 0.961-0.963
soma + FIR taps:                         about 0.955
```

The FIR result barely changes as cable is moved.

That is the strongest result in this gate.

The passive tree did **not create new information** about the hidden sources. A digital reader with explicit memory can recover almost the same information even from the one-dimensional soma trace. What geometry changed was how much of that temporal information had already been converted into **simultaneously available spatial state** before the readout.

A useful interpretation is therefore:

> **Dendritic geometry can materialize history into space.**

Long/redistributed cable is not a memory store in the strong sense. It is a physical temporal feature transform. A downstream memoryless local readout may get temporal coordinates "for free" because the cable has already performed the filtering.

This is the cleanest connection so far to the old Takens abstraction: the artificial delay vector explicitly copied history into coordinates; the passive cable produces history-dependent spatial voltages through its own dynamics.

## What this does NOT establish

- real dendrites evolved to perform blind source separation;
- longer bifurcation branches are always better;
- AMUSE is a biological learning rule;
- the passive tree adds information unavailable in the input history;
- morphology beats a digital delay line.

Quite the opposite on the last point: the explicit FIR/delay attacker is excellent.

## Next gate

The passive result now earns one nonlinear question:

> **Does a local NMDA-like branch nonlinearity make the morphology do something that a simple linear delay bank cannot remove so easily?**

The clean attack is morphology fixed, distributed mixed drive fixed, then compare:

```text
passive branches
vs
local saturating / regenerative branch nonlinearity
vs
matched digital nonlinear filter bank
```

Do not call the nonlinearity "NMDA" until a conductance-based implementation is used. Start with a transparent local voltage-dependent current, then upgrade only if the effect survives.
