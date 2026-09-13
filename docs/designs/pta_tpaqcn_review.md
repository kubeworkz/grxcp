# TPA-QCN photonic tensor accelerator: a review of the proposal

**Reviewing:** `docs/GRX_AI_Photonics_PTA_TPA_QCN.md` (TR-PTA-2026-001 v1.0,
13,101 lines, seven layers, ~9,000 lines of Python).

**Method.** The Tier-1 physics code was extracted from the document, installed
against torch 2.14 and torchdiffeq 0.2.5, and run. Everything numerical below
is a measurement from that code or an independent closed-form cross-check of
it, not a reading of the prose. The audit script is
`docs/designs/pta_tpaqcn_audit.py`; the corrected physics is
`docs/designs/pta_tpaqcn_waveguide_pol.py` with
`docs/designs/pta_tpaqcn_phase_match.py` driving it. Together they reproduce
every number here in about ten minutes on CPU. Material values are taken from
the published literature, cited in section 4.1, not assumed.

**Checked against the device.** The first draft of this review missed that the
waveguide the proposal needs has already been built and published: Thériault,
Petit, Anand V.S. and Kéna-Cohen, *Sci. Adv.* 12, eaeg3170 (May 2026, open
access). Section 2.3 checks the review against it, section 4.4 re-derives the
energy from its measured efficiency, and two claims below are withdrawn.
`docs/designs/pta_tpaqcn_measured.py` reproduces all of it in seconds with the
standard library alone.

**Verdict in one paragraph.** The central idea is good and the gap it targets
is real: TPA-QCN has demonstrated second-harmonic generation in waveguides but
has never been engineered into a neural-network activation function, and
conditioning a generative model on a *transfer curve* rather than a scalar
figure of merit is a genuine contribution. Layer 1 is real, runnable,
differentiable code. But the physics layer as written cannot express the
mechanism the whole project depends on, its coupling coefficient is
dimensionally inconsistent, and when both are repaired the headline energy
claim moves by five orders of magnitude and lands *worse* than the
optoelectronic path the architecture exists to eliminate — and with the
material's own published chi(2) rather than the assumed value, worse by a
further factor of forty. The published TPA-QCN waveguide now settles both
halves with measurements rather than models: the phase-matching mechanism is
real, in a 230 nm film, and run through the same activation definition the
device as built costs about **330 pJ per activation** at 2 mm, 65x the O-E-O
path — about 8 pJ in a 6 mm unit once its avoidable leakage is removed and the
best known derivative is used. A pulsed clock with group-velocity-matched units
could take that far below a digital O-E-O activation, but only into the range
of an analog optoelectronic neuron, not clearly below it (section 4.5). None of
that kills the idea. It relocates the project: the thing to build first is not Layer 7,
it is a corrected Layer 1 and a numerical experiment we can run on hardware we
already have.

---

## 1. What is real

**The coupled-mode solver runs.** `CMEFunc` integrates the standard type-I SHG
equations over a four-real-component state through `torchdiffeq`, batched and
differentiable end to end. Once phase matching is imposed by hand, its output
matches the closed-form depleted-pump result
`P_ff(z) = P_in sech^2(kappa sqrt(P_in) z)` to better than 10% across three
decades of length. The integration is correct. What feeds it is not.

**The framing in the report's background section is the best idea in the
document.** The design target for an activation unit is a *function*, not a
number, and that genuinely does require a generative model conditioned on
curves. That should be the paper's central claim; at present it is one
paragraph inside 13,000 lines.

**The self-assessment is honest where it is present.** The effective-index
model is flagged as a placeholder in three separate places. The limitations
table names the missing measured devices and the two-level chi(2) treatment.
That candour is worth preserving as the document is revised.

---

## 2. The physics layer cannot phase-match

Phase matching is not a detail here. The entire argument for TPA-QCN over
lithium niobate is *poling-free* phase matching: giant negative birefringence
lets the TM00 index curve at the second harmonic cross below the TE00 curve at
the fundamental, so no electrodes and no domain engineering are needed.

The model cannot represent that crossing.

Sweeping the full design space the code itself defines — 38,400 geometries,
width 0.15–6.0 um by height 0.05–3.0 um:

| quantity | value |
|---|---|
| min `n_eff(2w) - n_eff(FF)` | **+0.0427** |
| max `n_eff(2w) - n_eff(FF)` | +0.3110 |
| crosses zero anywhere | **no** |
| reference unit (1.0 x 0.4 um), delta_beta | 1.094e6 /m |
| delta_beta x L over the reference 0.5 mm unit | **547 radians** |

The cause is structural, not a tuning problem. The model is

```
n_eff = n_clad + (n_core - n_clad) * Gamma(V),    V ~ width / lambda
```

with **one core index per wavelength and no polarisation axis at all**. Because
V scales as 1/lambda, the second harmonic is always better confined, and
because `n_core_sh = 1.85 > n_core_ff = 1.75` it always starts higher. The SH
index is therefore higher for two independent reasons and can never cross. TE
and TM are never distinguished anywhere in the file, so birefringence — the
one material property the project is built on — has nowhere to live.

Two consequences follow, and the second is the serious one.

1. Every transfer function this solver has ever produced came from a deeply
   phase-mismatched interaction (547 radians of walk-off), not from the
   designed mechanism. They are cascading artefacts.
2. **Layer 3 inherits this.** A diffusion model trained on datasets generated
   by this solver is learning to inverse-design against physics that cannot
   express its own target. The training data is not noisy, it is structurally
   wrong, and no amount of adjoint guidance repairs that.

### 2.1 The fix, and what it shows

The replacement is `docs/designs/pta_tpaqcn_waveguide_pol.py`: a uniaxial film
(n_o in-plane seen by TE, n_e out-of-plane seen by TM), confinement from a real
asymmetric-slab dispersion relation solved by bisection rather than a sigmoid,
and the effective-index method for the second dimension. TE and TM differ by
the n² ratios in the TM boundary condition, which is where the polarisation
dependence physically lives. Type-I here is TE00(w) + TE00(w) -> TM00(2w), so
the tensor element is **chi_31, not chi_33** — two in-plane fundamentals
driving an out-of-plane polarisation.

The sign is right for the mechanism. Published films show S ≈ −0.30 to −0.35,
a preference for the molecular long axis lying *in* the plane, so in-plane
polarisability exceeds out-of-plane: n_o > n_e, negative uniaxial, optic axis
normal. That is exactly what lets the SH branch fall below the FF branch.

**With the axis in place, phase matching appears.** Sweeping width now produces
a genuine zero crossing — for example at a 0.20 birefringence against 0.10
dispersion, at h = 0.8 um the crossing sits at w = 1.371 um.

It also reveals something the document does not discuss: phase matching needs
the birefringence inside a **window**, not merely above a threshold.

| material dispersion n(2w)−n(w) | birefringence window | as a multiple |
|---|---|---|
| 0.05 | 0.08 – 0.24 | 1.6x – 4.8x |
| 0.10 | 0.14 – 0.28 | 1.4x – 2.8x |
| 0.15 | 0.18 – 0.34 | 1.2x – 2.3x |
| 0.20 | 0.24 – 0.38 | 1.2x – 1.9x |

Below the window the SH branch stays above the FF branch at every width and
never crosses. Above it the SH branch sits below the FF branch everywhere —
already past the crossing — and eventually falls under the oxide index
entirely and stops being guided. "More birefringence is better" is false.

This was the prediction to check: if TPA-QCN's birefringence at 1550/775 nm
sits inside the window for its own dispersion, the mechanism is sound and the
document's premise survives. **It has been checked, and it does** (section
2.3): the measured ordinary dispersion is 0.06 and the birefringence 0.19–0.23,
inside the window interpolated from the table above (about 0.10–0.25), and a
fabricated waveguide phase-matches.

### 2.2 Two things the corrected model surfaced, one since withdrawn

**The tensor element costs a factor.** chi_31 and chi_33 are close for TPA-QCN
itself (8.1 ± 0.3 and 7 ± 3 pm/V at 1550 nm) but not for its derivatives —
compound 5 has chi_33/chi_31 ≈ 2, which is 4x in energy. Any pipeline that optimises chi(2)
without tracking which component the polarisation combination actually uses
will select the wrong molecules.

**Withdrawn: "the film has to be far thicker than anyone has made one."** The
first draft put the phase-matched heights at 0.4–2.0 um against published films
of 49–108 nm, and concluded that either the deposition had to be pushed
twenty-fold or the device had to become an organic-loaded silicon-nitride
guide. The published device uses a 230 nm film and phase-matches. The error was
the geometry, not the material. The model is an air-clad channel on oxide, a
strongly asymmetric slab: with the measured indices its TE0 fundamental at
1550 nm is cut off below 194 nm, so a 230 nm film barely guides, and the model
still finds no crossing at 0.4 um or below. Put a low-index cover on the same
film — the device's resist strip, n = 1.418 — and the cutoff falls to 64 nm;
230 nm is then comfortably guided. The deposition worry does not survive
either: the 230 nm film, grown at 4 Å/s, delivers the efficiency in section
2.3. The organic-core guide the document assumes is viable, and an
organic-loaded nitride guide is an option rather than a necessity.

### 2.3 The device exists: what it measures

Thériault, Petit, Anand V.S. and Kéna-Cohen, *Sci. Adv.* 12, eaeg3170 (27 May
2026) — the group behind the materials paper in section 4.1 — built the
TE00(w) → TM00(2w) waveguide this review argues for: a 230 nm TPA-QCN film on
2 um of thermal oxide, a 15 nm TCTA cap, and a low-index resist strip for
lateral confinement. What it reports, with the indices digitised from its
Fig. 2A and the acceptance from its Fig. 3C (axes calibrated on the figures'
own tick labels, and Fig. 2A additionally on its 775/1550 nm marker lines,
which land within 1 nm; index readings good to about ±0.002):

| quantity | value | source |
|---|---|---|
| chi_31, chi_33 at 1550 nm | 8.1 ± 0.3, 7 ± 3 pm/V | text |
| n_o, n_e at 1550 nm | 1.764, 1.572 | Fig. 2A |
| n_o, n_e at 775 nm | 1.827, 1.595 | Fig. 2A |
| phase matched at | strip width 1.9 um, n_eff = 1.477 | text |
| modal overlap | 96% | text |
| propagation loss, 1550 and 780 nm | 20 ± 2 dB/cm both | text |
| efficiency, best device (1.7 mm) | P_2w / (P_w² L²) = **29 %/W/cm²** | text |
| same, leakage removed (5 dB/cm) | 68 %/W/cm², projected | text |
| phase-matching FWHM | about 12 nm of pump wavelength | Fig. 3C |
| geometric tuning | +50 nm of strip width → +22 nm of peak | text |

Three conclusions.

**The premise holds.** In bulk, n_e(2w) already sits 0.169 below n_o(w); the
guide has to close that gap, which is why the crossing lives at the weakly
confined end, n_eff 1.477 against the oxide's 1.444. More than 10 dB/cm of the
loss is leakage the authors identify as avoidable — into the substrate through
the thin buffer, and laterally from the TM second harmonic into TE slab modes —
leaving about 5 dB/cm of scattering; the film's own absorption at the second
harmonic is under 1 dB/cm.

**The corrected model was directionally right and quantitatively loose.**
Removing the propagation loss from the measurement gives a lossless efficiency
of 93 %/W/cm², kappa ≈ 96 W^-1/2 m^-1; the paper's own 68 %/W/cm² projection
reproduces from it to within 2%, which confirms the loss treatment. Section
4.1's corrected model gave kappa 121 at 8 pm/V: 1.6x optimistic in efficiency,
for a different geometry. A strip-loaded version of the model (effective-index
method, with the TM wave treated as extraordinary rather than as isotropic n_e)
lands the phase-matched n_eff within 0.01 of the paper but puts the width near
1.0 um rather than 1.9 um, and moving the indices 0.005 moves it 0.45 um. The
phase-matching width is decided by index differences of order 10^-3.

**That is Layer 3's problem in one number.** A generative model that designs
activation units must get phase matching right to 10^-3 in index, and
effective-index physics is not accurate enough to train it on. The dataset
regeneration in section 10 should be calibrated against this device — n_eff
1.477 at 1.9 um, 93 %/W/cm² lossless — before anything is retrained.

---

## 3. The coupling coefficient is dimensionally inconsistent

The model computes

```
kappa = (omega/2) * sqrt(eta0) * chi2 / (n_ff * n_sh) * overlap
```

Dimensionally: `[omega] = 1/s`, `[sqrt(eta0)] = V/sqrt(W)`, `[chi2] = m/V`,
so the product carries **m s^-1 W^-1/2**. The coupled-mode equation requires
**m^-1 W^-1/2**. The expression is short by a factor with units of s m^-2 —
which is exactly `1 / (c * sqrt(A_eff))`. There is no effective mode area
anywhere in the file, and SHG coupling cannot be area-independent: the same
power in a tighter mode is a higher intensity and converts faster.

Two independent repairs agree:

| A_eff | doc / (c sqrt(A_eff)) | textbook sqrt(eta_norm) | doc / textbook |
|---|---|---|---|
| 0.25 um² | 1195 | 2319 | 77x |
| 0.50 um² | 845 | 1640 | **109x** |
| 1.00 um² | 598 | 1160 | 154x |

where the textbook column is `kappa = sqrt(eta_norm)` with
`eta_norm = 8 pi^2 d_eff^2 / (eps0 c n_ff^2 n_sh lambda^2 A_eff)`, the standard
normalised SHG efficiency. Patching the document's own expression and deriving
from scratch land within a factor of two of each other — the residual is the
chi(2)-versus-d_eff convention and the overlap factor.

As written the coupling is about **109x too large**, and because threshold
power goes as `1/(kappa L)^2`, the code understates the power an activation
needs by roughly four orders of magnitude.

---

## 4. Energy per activation is picojoules, not attojoules

Define the knee as the input power at which half the fundamental has converted
— the midpoint of the sigmoid an activation unit is biased around — with
propagation loss divided out so the knee is nonlinear, not lossy. One
activation is one symbol at the document's own claimed 100 GHz, so 10 ps.

Phase matching imposed, textbook kappa, A_eff = 0.5 um², chi(2) = 50 pm/V:

| L | P_knee (solver) | P_knee (analytic) | E per activation | FF loss at 15 dB/cm |
|---|---|---|---|---|
| 0.1 mm | 29.3 W | 28.9 W | 293 pJ | 0.2 dB |
| **0.5 mm** | **1.245 W** | 1.155 W | **12.5 pJ** | 0.8 dB |
| 1 mm | 0.335 W | 0.289 W | 3.4 pJ | 1.5 dB |
| 2 mm | 0.097 W | — | 0.97 pJ | 3.0 dB |
| 10 mm | 0.0104 W | 0.0029 W | 0.10 pJ | 15 dB |

Solver and closed form agree where loss is negligible and diverge exactly where
it is not, which is the correct behaviour and a good check on the integration.

Against the claims in the document:

| | value | ratio |
|---|---|---|
| measured, 0.5 mm reference unit | 12.5 pJ | — |
| document's activation energy | 100 aJ | **124,546x** |
| the O-E-O activation being replaced | ~5 pJ | **2x worse** |

At the document's own reference geometry the all-optical activation costs
*more* than the optoelectronic path the architecture exists to remove. That is
the finding that matters, and it is not close.

### 4.1 With the material's measured chi(2)

The 50 pm/V above is the document's assumption. The measured values, from
ellipsometry and nonlinear polarimetry on these exact films
(Thériault et al., *Adv. Opt. Mater.* 2026, arXiv:2511.13682), at 1550 nm:

| compound | chi_31 | chi_33 | Tg |
|---|---|---|---|
| **TPA-QCN** (parent) | **8.1 ± 0.3 pm/V** | 7 ± 3 pm/V | 110 °C |
| compound 4 (best measured) | 16 pm/V | 18 pm/V | 76 °C |

(The parent's values are as reported with the waveguide, section 2.3; the
materials paper gives compound 4 as a twofold improvement on them. The 8 pm/V
row below predates the 8.1 value; at 8.1 it falls by 2.5%.)

The assumed value is **6x** the real one, and energy goes as 1/chi². Running
the phase-matched geometries from section 2.1 with the correct tensor element,
correct film overlap and a real mode area, at 2 mm (a 3 dB loss budget):

| chi_31 | kappa | E per activation at 2 mm |
|---|---|---|
| 50 pm/V (document's assumption) | 753 | 4.6 pJ |
| **8 pm/V (TPA-QCN, measured)** | 121 | **179 pJ** |
| 16 pm/V (compound 4, measured) | 241 | 45 pJ |

So the model's figure for the material as it exists is **179 pJ per
activation** — about 36x *worse* than the O-E-O path, and 1.8 million times the
document's claim. The best molecule anyone has made still lands at 45 pJ, 9x
worse than O-E-O. The measured device is worse again: **326 pJ** at the same
2 mm (section 4.4).

Note the Tg column. The best-performing derivative has the **lowest glass
transition**, 76 °C. That is the ageing trade-off from section 8 made concrete:
the molecular-engineering pathway the document relies on moves chi(2) up and
thermal stability down, in a device that needs a TEC anyway.

### 4.2 What would reach 100 aJ

Reaching 100 aJ at 100 GHz needs a 10 uW knee, which needs `eta_norm` about
**1.2e5x** larger. Each single-variable route is unphysical:

- **Length alone**: 170 mm, costing 255 dB of fundamental loss at 15 dB/cm.
- **chi(2) alone**: 17,000 pm/V. The document assumes 50; lithium niobate d33
  is about 27. The stated molecular-engineering pathway is a 2x improvement,
  which is 4x in energy — real, and four parts in 10^5.
- **Mode area alone**: 4.3e-6 um², far below a square nanometre.

**Resonant enhancement is the only remaining route, and it contradicts a
different row of the same table.** At 1550 nm the carrier is 193.4 THz, so an
activation bandwidth above 100 GHz caps the loaded Q at about **1,934**. High-Q
buildup and wide bandwidth are one knob turned in opposite directions. The
document quotes both ends of that trade simultaneously: >100 GHz *and*
attojoules.

### 4.3 Where the honest design point is

Energy falls as `1/L^2` while loss grows linearly in L, so there is no interior
energy optimum — the binding constraint is cascadability, not energy. At the
document's assumed 50 pm/V, budget 3 dB of loss per activation and the unit is
2 mm long and costs about 1 pJ. Budget 15 dB and you reach 0.1 pJ but now need
roughly 30x optical gain per activation stage to stay cascadable, and that gain
needs a pump whose power is nowhere in the energy model.

**Withdrawn: "about 1 pJ, the same order as the O-E-O path", as the number to
manage the programme against.** It used the assumed chi(2), which section 4.1
had already shown to be 6x the real one. Anchored on the measured device
(section 4.4), the same 3 dB budget buys **33 pJ** with TPA-QCN once the
avoidable leakage is gone, and **8 pJ** with compound 4 — both still above
O-E-O. Reaching 1 pJ at 3 dB needs roughly another 8x in efficiency on top of
both, which tighter confinement (the fully etched and slot geometries the
authors propose) might supply. Reaching 100 fJ is beyond even periodically
poled thin-film lithium niobate, the most efficient chi(2) waveguide platform
built (about 5000 %/W/cm²), which at 1 cm and no loss lands near 155 fJ.

The defensible thesis is therefore narrower than the first draft's: a
poling-free, back-end-compatible nonlinearity that can plausibly come within a
small factor of O-E-O, chosen for its integration rather than its energy.
"Attojoules at 100 GHz" is 1,500x below the best chi(2) platform at 1 cm.

### 4.4 Anchored on the measured device

The same definition as section 4 — the input power at which half the
fundamental has converted, its linear loss divided out, times one 10 ps symbol
— applied to the kappa ≈ 96 W^-1/2 m^-1 recovered from the measured
29 %/W/cm² (section 2.3), at the measured loss and at the loss the authors
project once leakage is removed. The integration reproduces this review's own
179 pJ from its own inputs (178 pJ), so the tables compare directly.

| case | L | loss | P_knee | E per activation | vs O-E-O |
|---|---|---|---|---|---|
| TPA-QCN as built, best device | 1.7 mm | 3.4 dB | 42.4 W | 424 pJ | 85x |
| TPA-QCN as built, this review's 2 mm unit | 2 mm | 4.0 dB | 32.6 W | **326 pJ** | 65x |
| leakage removed (5 dB/cm) | 2 mm | 1.0 dB | 23.5 W | 235 pJ | 47x |
| leakage removed, 3 dB budget | 6 mm | 3.0 dB | 3.26 W | **32.6 pJ** | 6.5x |
| + compound 4's chi(2) | 6 mm | 3.0 dB | 0.835 W | **8.35 pJ** | 1.7x |
| + compound 4, 15 dB budget | 30 mm | 15 dB | 0.105 W | 1.05 pJ | 0.21x |

Only the last row beats O-E-O, and it needs the ~30x of optical gain per stage
that section 4.3 prices at a pump nobody has modelled. The compound 4 rows
assume its films guide and lose like TPA-QCN's, which has not been measured.

### 4.5 What far better than O-E-O would take

Every energy above prices an activation as a full 10 ps symbol at constant
power. That is an assumption about the clock, not the material, and it is the
largest single factor in the comparison: the knee is a threshold in *peak*
power, and the energy is that power times however long it is held. With a
pulsed clock:

| unit | P_knee | 10 ps | 1 ps | 100 fs |
|---|---|---|---|---|
| TPA-QCN as built, 2 mm | 32.6 W | 326 pJ | 32.6 pJ | 3.26 pJ |
| compound 4, leakage removed, 6 mm | 0.835 W | 8.35 pJ | 835 fJ | 83.5 fJ |
| TFLN-class (~5000 %/W/cm²), 1 cm, no loss | 15.5 mW | 155 fJ | 15.5 fJ | 1.55 fJ |

The femtosecond column is not hypothetical: dispersion-engineered lithium
niobate nanowaveguides have switched with energies down to 80 fJ and times down
to about 46 fs, with no cavity (Guo et al., *Nat. Photonics* 16, 625, 2022).

**Walk-off sets how short the pulse can be.** The fundamental and second
harmonic travel at different group velocities, and a pulse shorter than the
delay they build up over the unit converts as if it were longer. The measured
acceptance is that delay seen from the other side: to first order, 12 nm FWHM
is about 0.3 ps of walk-off over the paper's devices. A 6 mm unit would walk
off 0.6–1.8 ps, depending on the length of the devices behind Fig. 3C, which
the paper does not give. So pulses of one to two picoseconds fit the dispersion
already demonstrated, and they bring the compound 4 unit to 0.8–1.7 pJ, three
to six times better than the document's O-E-O figure. 100 fs needs a
group-index mismatch at or below 0.005, some ten times smaller than those
devices imply: group-velocity matching as a design target alongside phase
matching, a requirement the published device, pumped continuous-wave, never
faced.

**5 pJ is the wrong baseline.** It prices a digital path per activation:
receiver, ADC, DSP, DAC, driver. What an all-optical activation actually
competes with is an analog optoelectronic neuron, a photodiode charging a
modulator directly: enough light to deliver the charge C·V, plus C·V² from the
supply. That is 172 fJ for a 50 fF device swinging 1.5 V, 18 fJ at 10 fF and
1 V, and 650 aJ at 1 fF and 0.5 V. The architecture has been demonstrated end
to end — optical linear layers with optoelectronic activations classify images
on-chip in under 570 ps (Ashtiani, Geers and Aflatouni, *Nature* 606, 501,
2022) — and it comes with what a χ(2) activation has to buy: gain, fan-out,
isolation between input and output, and restoration of signal levels (Miller,
*Nat. Photonics* 4, 3, 2010). A parametric activation gets gain only from a
pump. Both sides are counted at the device, before the laser's wall-plug
efficiency, which penalises the all-optical side more because nearly all of its
energy is light.

Per neuron at 100 GHz, energy times rate: digital O-E-O 500 mW, the compound 4
unit at 100 fs 8.4 mW, the 18 fJ analog neuron 1.8 mW. The optimistic
all-optical case lands in the analog neuron's range, not below it. Only
TFLN-class efficiency with femtosecond pulses gets well under today's analog
devices, and it arrives where nanoscale optoelectronics already points, around a
femtojoule — where an activation carries about 12,000 photons and shot noise, at
about 1%, starts to decide how far a signal can travel between resets (section
5.2).

**So the honest answer has two halves.** Far better than a *digital* O-E-O
activation is physically reachable, with a pulsed clock and
group-velocity-matched units. Far better than an *analog* optoelectronic neuron
is not supported by any number in this review. An all-optical nonlinearity wins
clearly only where there is no electronic signal to begin with and latency is
the product — optical signal processing, sensing before detection, picosecond
classification — because there it removes a conversion instead of cheapening
one. And an activation's energy is rarely what dominates a system: section 7
is the reminder that operand supply, not arithmetic, set the limit the last
time we measured.

### 4.6 Or do without an activation device

Two published approaches get nonlinear computation out of optics without a
nonlinear element per neuron, and both bear on this programme.

**Nonlinearity from linear optics.** Encode the input into a scattering medium
more than once and the scattered field becomes a nonlinear function of the
data, although every element is linear. Programmable nonlinear transformations
follow at milliwatt continuous-wave power (Yildirim et al., *Nat. Photonics* 18,
1076, 2024). What it costs is modulators, which a photonic tile has anyway.

**Train the physics as it is.** A deep network built from ultrafast
second-harmonic generation itself — inputs encoded in a pulse's spectrum, no
digital activation functions — classified vowels at 93% when trained through
the real hardware, where training on an accurate digital model of the same
system reached about 40% (Wright et al., *Nature* 601, 549, 2022). That is the
lesson of sections 2.3 and 5.1 from the other direction: models of
phase-matched χ(2) devices are not accurate enough to train against, and
training through the hardware absorbs their imperfections instead of trimming
them out. For a tile with fabricated activation units, training through the
tile is the deployment regime to assume, and S_ACT can emulate it with the
detuning term in place.

---

## 5. What the document does not see: the ADC was a noise reset

Removing the photodetect-digitise-remodulate chain between layers is presented
as pure upside — latency and energy. It also removes the only requantisation
point in the datapath.

In an O-E-O design every activation rounds the signal back onto a digital grid.
Analog error is bounded *per layer* and cannot compound: shot noise, thermal
drift, phase error and crosstalk are all erased at each boundary. Go all-optical
and nothing arrests them. Error accumulates across depth, and because the
activation is nonlinear the accumulation is not even a simple random walk — the
transfer curve's slope amplifies error wherever it is steep, which for a sigmoid
is precisely the operating point.

The document treats precision as a static property ("4–8 bits, analog"). It is
not, once there is no reset. The quantity that matters is not bits per layer but
**how many layers can be chained before accuracy collapses**, and that number is
an architectural constraint: it sets how often an O-E-O reset must be inserted,
which sets the real average energy per activation, which feeds straight back
into the comparison in section 4.

Nobody has this number for any material. We can measure it without photonics.

### 5.1 And every activation unit is its own curve

The measured device adds an error source the O-E-O path never had. Phase
matching is a geometric condition, and the paper measures how geometric: +50 nm
of strip width moves the phase-matching peak by +22 nm, against an acceptance
of about 12 nm FWHM. At a shared pump wavelength, a unit whose strip is 14 nm
wider or narrower than nominal converts at half the efficiency of a nominal
one, and its knee moves with it. The acceptance narrows as 1/L, so the long
units that section 4.4 needs for low energy are proportionally less tolerant.
Film thickness moves the peak too; the paper's phase-matching map is a function
of both.

This error is fixed-pattern, not random. It does not average down over time or
across a batch; it is fabricated into each unit, so a network sees a different
activation function at every neuron. It has to be trimmed per unit — thermally,
at a power that belongs in section 4's budget — or learned around, and which
one is cheaper is an architectural question the S_ACT experiment can answer.

### 5.2 The reset caps the gain

If accuracy forces an O-E-O reset every N activations — the N of section 6.1 —
a layer of the all-optical chain costs E_opt + E_OEO / N, against E_OEO for an
O-E-O layer. The advantage is at most N, however cheap E_opt becomes. For the
compound 4 unit at 100 fs (83.5 fJ, section 4.5):

| N | vs digital O-E-O, 5 pJ | vs analog neuron, 18 fJ |
|---|---|---|
| 1 | 0.98x | 0.18x |
| 4 | 3.7x | 0.20x |
| 10 | 8.6x | 0.21x |
| 100 | 37x | 0.22x |
| ∞ | 60x | 0.22x |

Against a digital path the win is whatever N the network tolerates. Against an
18 fJ analog neuron this unit loses at every N; against today's 172 fJ devices
it wins by 2.1x at most.

E_opt and N are not independent either. A cheaper activation carries fewer
photons, so its shot noise per stage is larger and the chain before a reset is
shorter. At 83.5 fJ that is irrelevant — 650,000 photons, 0.12% — but at the
femtojoule end of section 4.5 it is the wall. The quantity worth measuring is
N as a function of photons per activation.

---

## 6. The experiment to run: an activation stage in the c930 NPU

The c930 NPU has no activation stage at all. C leaves `c_mem` as INT32 or FP32
and the CPU applies the nonlinearity in software. That gap is the opportunity:
an activation unit is a new pipeline stage, and in the FPGA emulator it is a
numerics block, not photonics.

### 6.1 What to build

A new state `S_ACT` between `S_WRITE` and the next output row, carrying:

1. **The transfer function**, as a piecewise-linear or piecewise-quadratic
   lookup over normalised input power, with breakpoints generated by the
   corrected coupled-mode solver. One table per (chi2, L, delta_beta) design
   point, so a diffusion-generated unit can be dropped in as data rather than
   as RTL.
2. **The analog error model already specified for the CPU path**
   ([PTA on the CPU](pta_cpu_integration.md)): signal-dependent shot noise with
   sigma proportional to sqrt(|y|), thermal and TIA noise, weight programming
   error, drift, crosstalk. Driven by the same LFSR the existing testbench uses,
   so RTL and the C reference stay bit-comparable and any result reproduces.
   Plus one term the CPU path does not need: a per-unit phase-matching
   detuning, drawn once per unit from a strip-width distribution and then held
   fixed (section 5.1), scaling that unit's transfer curve. Measured anchor:
   14 nm of width costs half the efficiency.
3. **A configurable requantisation point** — a CSR field saying "insert an
   O-E-O reset every N activations", with N = 1 reproducing today's
   optoelectronic behaviour and N = infinity the all-optical limit.

Everything else already exists. The engine runs a real GEMM through a real DMA
at realistic shapes, and it now does so 2.35x faster than a week ago.

### 6.2 What it measures

Sweep N and network depth; report task accuracy against a bit-exact digital
reference. The output is a curve of accuracy versus number of chained
all-optical activations, per transfer-function shape. From it:

- the maximum optical chain length at a given accuracy target;
- that chain length as a function of photons per activation, swept through the
  shot-noise term, which is the curve section 5.2's bound needs;
- therefore the O-E-O reset frequency;
- therefore the *true* average energy per activation, which is the section-4
  number amortised with a picojoule-scale reset every N stages;
- the strip-width tolerance below which units need no per-unit trim, which
  decides whether the section-4.4 energy carries a tuning-power overhead;
- and, because the transfer curve shape is an input, which activation shapes
  are most robust to accumulation. That is a design constraint the diffusion
  model can be conditioned on, and it is not in the current loss function.

It also gives the all-optical branch a kill criterion, in section 5.2's terms:
unless the measured N makes E_OEO / (E_opt + E_OEO / N) comfortably exceed one
against an *analog* optoelectronic neuron, the activation stays electronic.
Against the 18 fJ neuron of section 4.5, every TPA-QCN unit in this review
fails at every N, and even today's 172 fJ devices are beaten by 2.1x at most,
which is not comfortable. Only the TFLN-class row at 100 fs passes against
18 fJ — 3.0x at N = 4, 12x in the limit — and it still fails against a
nanoscale neuron. The experiment says whether any combination of efficiency,
clock and N that the programme can actually build passes.

### 6.3 Why this is the right first build

It needs no fab, no PDK, no photonics, and no new hardware. It answers a
question nobody in the field has answered. And it closes the "use AI to build
AI" loop in the only direction that can close today: measured architectural
constraints from silicon we control, feeding the conditioning of the generative
model, whose outputs come back as lookup tables for the next measurement. The
Layer 6 fabrication loop cannot close until there is a fabrication run. This
one can close this quarter.

---

## 7. Two internal contradictions worth fixing

**Latency versus weight streaming.** "Latency per layer: 1–10 ps" is optical
propagation through the mesh and nothing else. "Limited on-chip memory —
mitigation: weight streaming from HBM" puts weight reconfiguration on the
critical path, and thermo-optic phase shifters settle in roughly 10 us — six
orders of magnitude above the quoted figure. Either the model is resident and
the picoseconds are real, or it streams and the phase shifters dominate
everything. The architecture code has no weight-load term at all, which is the
same omission we corrected in the c930 NPU this month: the array was never the
constraint, the operand supply was.

**Throughput has no operand supply.** `evaluate_pta` computes throughput as
MACs divided by propagation delay — no modulator bandwidth, no ADC readout, no
memory. That is peak arithmetic in a vacuum. Measured on real RTL, our own
8x8 array looked capable of 64 MAC/cycle and delivered 1.8 (2.9% PE
utilisation); after three fixes it is memory-bound at about 51. The photonic
analogue is that MACs per second is capped by modulator and ADC bandwidth times
channel count, not by time-of-flight. This is a small change to the performance
model and it will move the headline number by orders of magnitude — which is a
reason to make it now rather than after Layer 5.

---

## 8. Smaller notes

- **The evaluation tables carry no provenance.** Placement and scheduler
  results (45 us / 0.9 mJ / 0.15) read as authored rather than emitted: round
  values, no variance, no seeds, despite a reproducibility appendix that fixes
  every seed at 42. Either regenerate them or mark them illustrative. A reader
  who fails to reproduce them will discount Layer 1 too, and Layer 1 deserves
  better.
- **Organic film ageing is missing from the risk register.** It appears once, as
  a packaging note. For a *spontaneously aligned* organic film, relaxation of
  molecular alignment at operating temperature is the historical failure mode —
  it is why poled polymers never shipped. chi(2) decaying over months is a
  programme-ending risk and belongs in the register with an accelerated-ageing
  measurement plan, not in a fabrication table. The waveguide paper supplies
  the first data, encouraging as far as it goes: no degradation over 200 days
  unencapsulated, no optical damage at the pump powers used, and a TCTA-capped
  film keeping 90% of its SHG up to 129 °C. None of that is months of
  operation at temperature under optical load, which is still the measurement
  the register needs — and the watt-scale knees of section 4.4 are that load.
- **Seven layers is too many for zero measured devices.** The serving layer —
  gRPC, HTTP, Kubernetes autoscaling, Prometheus — is roughly 2,000 lines
  solving problems that existing serving stacks already solve, for a chip that
  does not exist. It is the most replaceable code in the repository and it
  dilutes the parts that are novel.

---

## 9. How this connects to what is already built

- **The NPU is the control plane, and the activation stage is the missing
  piece.** [PTA on the CPU](pta_cpu_integration.md) established that the
  weight-stationary dataflow already matches a photonic tile. This document adds
  the thing that tile has never had. Section 6 is where the two meet.
- **The roofline is the missing architecture layer.** The arithmetic-intensity
  analysis that produced our 51 MAC/cycle ceiling is the same analysis the PTA
  performance model needs, with modulator and ADC bandwidth substituted for AXI
  beats.
- **Block-scaled formats are the right numerics and are already argued for.** A
  shared exponent per block is how a limited-SNR analog channel gets dynamic
  range. The document says "4–8 bits (analog)" with no format; connecting it to
  the mxfp8/nvfp4 work costs nothing.
- **Multi-chip scaling and the coherence question are one problem.** The
  core-scaling analysis in the grx930 tree (`c930/doc/c930_core_scaling.md`)
  found the write-through MSI protocol is the real ceiling on the CPU side;
  Layer 5 assumes an optical crossbar. These should not be designed in separate
  documents.

---

## 10. Recommended order

The mainline does not wait on this list.
[PTA on the CPU](pta_cpu_integration.md) keeps the nonlinearity electronic at
the tile boundary, which section 4.5 shows is the right default. What follows
is the all-optical branch, and item 4 is its kill test.

1. ~~**Fix the effective-index model**~~ — **done**, section 2.1.
   `pta_tpaqcn_waveguide_pol.py` replaces it. Phase matching now exists and the
   birefringence window is the new result. ~~Next: check TPA-QCN's measured
   n_o and n_e against that window~~ — **done**, section 2.3: inside the
   window, and a fabricated waveguide phase-matches. The premise holds.
2. **Fix kappa** — add effective mode area, or adopt the normalised-efficiency
   form directly. One function. Then check it against the device: 93 %/W/cm²
   lossless in the published geometry.
3. **Regenerate any Layer 2 or Layer 3 dataset** produced before 1 and 2. It was
   trained on physics that could not phase-match. Calibrate the generator on
   the published device first (n_eff 1.477 at a 1.9 um strip, 230 nm film):
   phase matching is a 10^-3 index condition, and effective-index physics
   misplaces that device's width by 0.9 um.
4. **Build the S_ACT experiment.** No photonics required; answers the deepest
   open question; closes the design loop with hardware we control. Include the
   fixed-pattern detuning of section 5.1 from the start, and sweep photons per
   activation alongside N (section 5.2).
5. **Put an operand supply into the performance model**, then re-derive the
   headline throughput and energy claims — with the pulse length as a
   parameter, and an analog optoelectronic neuron beside the digital O-E-O
   baseline (section 4.5).
6. **Add ageing to the risk register** with a measurement plan.
7. **Cut or clearly label Layer 7**, and regenerate or mark the evaluation
   tables.

---

## 11. What this review does not answer

- ~~Whether TPA-QCN's actual birefringence lands in the window.~~ Answered in
  section 2.3: it does, and the device phase-matches. The indices used there
  are read off a published figure to about ±0.002; tabulated ellipsometry from
  the authors would tighten them, and would be needed to design widths.
- **Whether a tighter geometry changes the energy.** Section 4.4 assumes the
  published strip-loaded mode. Fully etched channels, slot guides and
  organic-loaded silicon nitride, which the authors propose, trade nonlinear
  overlap against mode area; a proper mode solver would settle by how much.
  None plausibly moves 33 pJ to 100 aJ.
- **Whether an all-optical activation can compete at all.** With a 10 ps
  clock the 3 dB design point is 8–33 pJ (section 4.4); with a pulsed clock it
  could reach tens of femtojoules (section 4.5), in the range of an analog
  optoelectronic neuron rather than below it. Against a digital path the
  answer is the chain length from section 6, which nobody has measured.
- **Whether group-velocity matching is possible in these films.** The
  femtosecond rows of section 4.5 need a group-index mismatch about ten times
  smaller than the published devices imply, at the same time as birefringent
  phase matching.
- **What an analog neuron would cost in practice.** Section 4.5's figures are
  C·V² estimates across plausible capacitances and swings, not a measured
  device, and nothing here prices one built alongside a GRX tile.
- **Anything about the diffusion model's quality.** Layers 2 and 3 were read,
  not run. The composite loss is well-posed; whether it trains is untested here.
- **Fabrication.** No judgement is offered on selective-area evaporation,
  shadow masking, or hermetic packaging.
