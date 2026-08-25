# GeoNeuronX

**Length is a temporal coordinate.**

GeoNeuronX is a falsification-first restart of the Geometric Neuron idea after the Monday/Tuesday source-separation work.

It is **not** a claim that dendrites implement AMUSE, Takens, ICA, Oja, or any one named algorithm. Those are abstractions and attackers. The biological papers motivate a narrower question:

> **Does the physical allocation of dendritic path length create useful temporal transfer diversity before the soma, and can an adaptive axon initial segment normalize the resulting high-dimensional computation into a narrow output event stream?**

That question is testable.

---

## Why restart now?

Three independent lines finally meet without needing metaphor.

### 1. Aizenbud et al. 2026: morphology really changes single-neuron I/O complexity

Aizenbud et al. quantify the input/output complexity of detailed human and rat cortical pyramidal neuron models by asking a fixed temporal convolutional network to imitate each cell. Human neurons are harder to emulate. The strongest morphological predictor is **total dendritic area** (`R² = 0.74`), while **longest bifurcation branch** is also strongly associated with complexity (`R² = 0.44`). Total area + longest bifurcation branch reaches `R² = 0.81`. Their interpretation emphasizes large dendritic trees, electrical compartmentalization, semi-independent dendritic subunits, and strong NMDA-mediated nonlinearities.

This repo asks a more specific mechanistic question than their paper does:

> Could part of the computational value of extra path length be that different paths implement different temporal transfer functions?

That is an inference/hypothesis, not a result from Aizenbud et al.

### 2. AMUSE/SOBI: history itself can identify hidden processes

AMUSE whitens a multivariate time series and diagonalizes **one lagged covariance matrix**. SOBI generalizes this by jointly diagonalizing several lagged covariance matrices. In Tuesday's theorem-friendly calibration, a single lag was already enough to separate two dynamical sources essentially perfectly.

So there is now a precise mathematical reason why a physical system exposing differently delayed/filtered versions of activity could be useful:

> **different causes can be distinguishable because they relate differently to their own past.**

### 3. The AIS is an adaptive neck

The axon initial segment (AIS) is not just wire. It generates and shapes the action potential, separates the somatodendritic and axonal compartments, and exhibits activity-dependent plasticity. Reduced activity can lengthen the AIS; elevated activity can shift it distally. These changes are generally interpreted as homeostatic tuning of intrinsic excitability, and their effect depends on the dendritic arbor.

That suggests a clean division of labor:

```text
many synaptic mixtures
        ↓
DENDRITIC TREE
path-dependent filtering
local history
local nonlinear subunits
        ↓
SOMA
rich analog state
        ↓
AIS
adaptive eventization / excitability governor
        ↓
AXON
narrow spike stream, then large downstream fan-out
```

The dendrite expands and transforms; the AIS compresses into an event channel; the axon distributes that event onward.

---

## The core abstraction

For a branch/path of length `L`, start with the minimal frequency response

```text
H(f, L) = exp(-alpha L) * exp(-i 2π f L / v)
```

Longer path:

- attenuates differently;
- rotates phase differently;
- delays differently;
- therefore changes the effective computation seen at the soma.

A bank of paths gives a matrix:

```text
                 source / temporal mode
              s1        s2        s3
branch L1   H11       H12       H13
branch L2   H21       H22       H23
branch L3   H31       H32       H33
   ...       ...       ...       ...
```

or simply

```text
x(t) = A(L) s(t)
```

where **morphology generates the observation matrix** `A(L)`.

Equal path lengths can produce redundant rows. Diverse path lengths can produce a higher-rank, better-conditioned temporal basis.

That does **not** mean morphology beats a digital matrix. A random digital matrix is an explicit attacker in Gate 0.

---

## Where Takens now fits

The original Geometric Neuron used a literal delay vector:

```text
[x(t), x(t-τ), x(t-2τ), ...]
```

and projected it against a hand-designed cosine receptor mosaic.

GeoNeuronX demotes the strong claim and keeps the useful abstraction:

> **Takens/delay coordinates were a cheap stand-in for physical history distributed across paths.**

A real dendrite is not a uniform delay line. It is a branched, lossy, nonlinear cable with many local compartments. But both constructions expose **history as coordinates**.

---

## Where Oja now fits

Oja's rule

```text
y = wᵀx
Δw = η y (x - y w)
```

is included as an online local baseline.

The important correction is that Oja does **not** normalize the incoming signal. The `-y²w` term limits the growth of the learned weight vector, producing a PCA-like principal direction.

So the hierarchy is:

```text
fixed receptor mosaic  → listen for a hand-specified temporal pattern
Oja / PCA              → learn a strong variance direction
AMUSE                  → learn a basis from one delayed covariance
SOBI                   → learn a basis from several delayed covariances
ICA                     → use higher-order / non-Gaussian structure
IVA                     → align corresponding sources across views
IVE                     → extract only the source currently wanted
```

GeoNeuronX will not award points for using the fanciest method. The boring method wins if it works better.

---

# Gate 0 — LENGTH IS A TEMPORAL COORDINATE

`experiments/gate0_length_is_memory.py`

Three independent AR(1) sources have equal marginal variance but different temporal autocorrelation. They are observed through eight branch-local channels. The observation matrix is generated only from branch path lengths using a delayed/attenuated transfer law.

The sweep holds branch count fixed and changes only the **span of path lengths**.

Attacks:

```text
equal-length morphology
PCA
online Oja (one component)
AMUSE(τ=1)
random digital full-rank matrix
```

Question:

> As path-length diversity increases, does the morphology-generated transfer matrix gain rank/conditioning, and does AMUSE recover the underlying dynamical sources more reliably?

This is a **candidate mathematical reason for length**, not a biological validation.

The digital random-matrix arm is load-bearing: if it does just as well, the correct conclusion is that geometry can *embody* a useful basis, not that geometry is algorithmically superior.

---

# Gate 1 — THE AIS IS AN ADAPTIVE NECK

`experiments/gate1_ais_homeostasis.py`

The same dendritic branch activity is scaled across four large changes in total somatic drive. Compare:

```text
fixed spike threshold
vs.
minimal homeostatic AIS threshold
```

The adaptive AIS tracks firing-rate error:

```text
high sustained output → harder to fire
low sustained output  → easier to fire
```

This intentionally models only the **functional direction** of AIS homeostasis. It does not pretend that a scalar threshold equals AIS position, length, channel density, or real electrogenesis.

Question:

> Can the output neck preserve a stable event budget while the complexity/load upstream changes?

This is the first explicit `INNER → AIS → OUTER` experiment in the restart.

---

## What would count as a stronger result later?

The first two gates are intentionally small. The serious next experiments are already visible:

1. **Real cable morphology.** Replace the analytic `H(f,L)` bank with a compartmental cable model or reconstructed morphology.
2. **Morphology-preserving attacks.** Preserve total area while redistributing path lengths; preserve branch count while collapsing long bifurcation paths; shuffle synapse locations.
3. **NMDA ablation.** Add local dendritic nonlinearities, then remove them while holding morphology fixed.
4. **Where does separability appear?** Score source recovery at synaptic input, distal branch, bifurcation branch, proximal dendrite, soma.
5. **Rat-like vs human-like geometry.** Ask whether the more complex morphology provides a larger useful temporal basis under matched input statistics.
6. **Population specialization.** Several units receive the same mixtures; local/Oja/PEM-like learning plus lateral competition should make different units specialize to different temporal causes.
7. **IVE/attention.** Once a population has candidate causes, extract only the source relevant to the current task rather than globally disentangling everything.
8. **Axonal fan-out.** One narrow event stream can drive many downstream targets; test whether downstream recipients can reconstruct/use different aspects of the same source under different local filters.

The stop condition is simple: if matched FIR/SSM/filter-bank baselines do everything equally well for less cost, say so.

---

## Claims explicitly NOT made

- dendrites implement AMUSE, SOBI, ICA, Takens, or Oja;
- Aizenbud et al. demonstrated source separation;
- AIS position is literally a scalar threshold;
- dendritic length exists *because* it separates sources;
- morphology beats digital filters;
- this is a brain simulator;
- this is evidence for consciousness or a new theory of intelligence.

The current claim is only:

> **Path length changes temporal transfer. Temporal transfer diversity can create additional coordinates. Mature source-separation mathematics tells us how to test whether those coordinates make hidden dynamical causes easier to recover. The AIS provides a biologically real adaptive boundary where rich internal computation is converted into a narrow outgoing event stream.**

That is enough to build.

---

## Run

```bash
python -m pip install -r requirements.txt
python experiments/gate0_length_is_memory.py
python experiments/gate1_ais_homeostasis.py
python -m unittest discover -s tests -v
```

Receipts are written to `results/`.

---

## Literature anchors

- Ido Aizenbud et al. (2026), **Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons**, PNAS 123(28), e2533168123. DOI: 10.1073/pnas.2533168123.
- Christophe Leterrier (2018), **The Axon Initial Segment: An Updated Viewpoint**, Journal of Neuroscience 38(9):2135–2145. DOI: 10.1523/JNEUROSCI.1922-17.2018.
- Pan, Matilainen, Taskinen & Nordhausen, **A review of second-order blind identification methods** — AMUSE/SOBI and extensions.
- Bariscan Bozkurt et al. (2026), **Normative Networks for Source Separation via Local Plasticity and Dendritic Computation**, arXiv:2605.19965.
- Erkki Oja / Hyvärinen & Oja — local normalized Hebbian learning, PCA/ICA lineage.

---

## Lineage

```text
Geometric Neuron
    delay geometry + fixed receptor mosaic
          ↓
Saturday
    state/material changes future processing
          ↓
Sunday
    separate fast state from persistent written operator
          ↓
Monday
    hidden causes + ICA/IVA/IVE; morphology attacked by FIR
          ↓
Tuesday
    AMUSE/SOBI: history itself can identify a dynamical cause
          ↓
GeoNeuronX
    LENGTH → TEMPORAL TRANSFER → LOCAL CAUSE COORDINATES
                         ↓
                  adaptive AIS neck
                         ↓
                    axonal event
```

**Build the boring abstraction first. Let the biology earn every stronger interpretation.**
