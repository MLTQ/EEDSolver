# Oracle — Field Theory Reference

## Purpose
This document specifies the governing equations implemented in `solver/fields/formulation.py`.
It is the physics contract for the solver. Any change to the weak forms must be
reflected here with a decision log entry explaining the change and its theoretical basis.

This is the core research asset of the project. Handle with care.

---

## Background: Why Standard EM Discards φ

In standard electromagnetism, the gauge freedom of the vector potential A allows
one to impose the Lorenz gauge (∂μAμ = 0) or Coulomb gauge (∇·A = 0). These
conditions are not physical constraints — they are choices that eliminate redundant
degrees of freedom to simplify computation.

The **Deleted Degrees of Freedom (DDOF)** framework argues that this gauge elimination
discards longitudinal field components that, in an extended electrodynamic theory,
carry physical content. Specifically, the scalar potential φ associated with
∇·A ≠ 0 is proposed to:
1. Couple to a gravitational-like scalar field sector
2. Propagate at potentially non-c velocities in certain media
3. Be detectable via anomalous force and torque effects on test bodies

The EED (Extended Electrodynamics) formulation, associated with the work of
Woodside, Arbab, and others in the tradition of T.T. Brown's empirical observations,
provides field equations for this extended system.

---

## Field Definitions

| Symbol | Name | Type | Space |
|--------|------|------|-------|
| φ | EED scalar potential | Scalar | H¹(Ω) |
| **A** | Magnetic vector potential | Vector | H(curl, Ω) |
| **B** = ∇×**A** | Magnetic flux density | Vector | L²(Ω) |
| **E** = -∇φ_em - ∂**A**/∂t | Electric field | Vector | L²(Ω) |
| **J** | Current density | Vector | L²(Ω) (source) |
| ρ | Charge density | Scalar | L²(Ω) (source) |
| α | EED scalar mass parameter | Scalar | ℝ (input param) |
| β | φ→**A** coupling constant | Scalar | ℝ (input param) |
| γ | **A**→φ coupling constant | Scalar | ℝ (input param) |

Note: φ here is the **EED scalar field**, not the standard EM scalar potential.
In the `scalar_only` and `eed_coupled` formulations, this is the primary
quantity of interest for lab hypothesis testing.

---

## Governing Equations (Strong Form)

### Maxwell Baseline (`maxwell_only`)
Standard magnetostatics (no displacement current, static fields):

```
∇ × (1/μ₀ ∇ × A) = J          in Ω
∇ · A = 0                       (Coulomb gauge)
A × n̂ = 0                      on ∂Ω  (tangential BC)
```

This is the control case. φ does not appear.

### EED Scalar Field (`scalar_only`)
Scalar field driven by divergence of current (longitudinal source):

```
-∇²φ + α²φ = S_φ               in Ω
φ = 0                           on ∂Ω  (Dirichlet)
```

where the source term:
```
S_φ = -c⁻² ∂(∇·A)/∂t ≈ -(1/μ₀ε₀) ∇·J   (magnetostatic limit)
```
(Coefficient note, 2026-07-02: the temporal term carries c⁻² — matching C's
(1/c²)∂φ/∂t — not the c⁻¹ this file previously showed, and the sign here now
matches the Source Term Construction section below. The overall normalization
of S_φ is heuristic in any case; α, β, γ are free parameters that absorb it.)

In the static limit with a prescribed current distribution, S_φ is computed
directly from **J** without solving for **A** first. This is the cheapest
formulation for exploring scalar field topology.

### Full EED Coupled System (`eed_coupled`)
The complete system couples φ and **A**:

```
-∇²φ + α²φ + β ∇·A = S_φ       in Ω
∇ × (1/μ₀ ∇ × A) + γ ∇φ = J   in Ω
```

Boundary conditions:
```
φ = 0                           on ∂Ω
A × n̂ = 0                      on ∂Ω
```

When β = γ = 0: reduces to decoupled Maxwell + isolated scalar.
When α = 0: scalar field is massless (long-range).
When α > 0: scalar field has a characteristic decay length λ = 1/α (Yukawa-like).

**Default parameter values for first build** (adjust based on literature + experiment):
- α = 0.0 (massless, to maximize predicted field extent)
- β = 0.1 (weak coupling, perturbative regime)
- γ = 0.1 (weak coupling, perturbative regime)
- μ₀ = 4π × 10⁻⁷ H/m

### Time-Domain EED (FDTD, `solver-gpu`)

The static weak forms above govern the legacy Python solver. The GPU solver
(`solver-gpu`) evolves a **potential-primary, time-domain** system. The
equations of motion (vacuum, single coupling knob γ; γ=0 = Maxwell/Lorenz):

```
∂²φ/∂t² = c²∇²φ − γ·c²·∂(∇·A)/∂t
∂²A/∂t² = c²∇²A − γ·∇(∂φ/∂t)        ( + c²µ₀J  source, injected separately )
```

**Status of this system — read before citing it (revised 2026-07-02):** this
"coupled form" is a **bespoke Oracle-specific model**, not a published theory.
It was constructed by (a) deleting the −c²∇(∇·A) term from the ungauged
Maxwell A-equation and (b) postulating a second-order φ-equation — at γ=1
equivalent to ∂C/∂t = ∇²φ — in place of the Gauss-law constraint (whose
∂ₜ(∇·A) term carries the *opposite* sign; the φ-equation here is not Gauss's
law "promoted"). Every published EED variant we cite (Woodside, van
Vlaenderen–Waser, Hively–Giakos, Stueckelberg/Aharonov-Bohm electrodynamics)
instead yields the **decoupled** □φ = ρ/ε₀, □A = µ₀J from an action principle.
The coupled form, as far as we can determine, follows from no action, is not
Lorentz covariant (the surviving ∇(∇·A)-shaped terms pick out the lab frame),
and admits superluminal longitudinal propagation (see dispersion below). An
earlier version of this document called the two forms "equally derivable" —
that overstated it. What remains true: the coupled form deliberately *retains*
a dynamical, radiating scalar sector that the decoupled derivation evaporates,
which is the phenomenon this instrument exists to model; it is simulated as an
explicit hypothesis, and experiment adjudicates.

**Longitudinal dispersion (derived 2026-07-02, verified in
`solver-gpu/tests/eed_gamma1_dispersion.rs`):** Fourier analysis of the
longitudinal sector (d ≡ ∇·A) gives

```
φ̈ = −c²k²φ − γc²ḋ        →    (c²k² − ω²)² = γ²ω²c²k²
d̈ = −c²k²d + γk²φ̇        →    ω± = ck·(√(γ²+4) ∓ γ)/2
```

Two non-dispersive branches. At γ=1 they propagate at **(√5−1)/2·c ≈ 0.618c**
and **(√5+1)/2·c ≈ 1.618c** — golden-ratio multiples of c, NOT c. Measured on
a 96³ grid (longitudinal A=∇G packet): γ=0 control front at 1.004c; γ=1 front
at 1.51c (φ·c minus second-order-stencil dispersion). Consequences: (1) any
claim that the longitudinal ∇·A half of C "radiates at c" is wrong — earlier
versions of this file said exactly that, three times; (2) timing/wavelength
predictions for scalar-signal detection must use ω±, not c; (3) the ω₊ branch
is superluminal — the model is acausal in some frame. Transverse modes are
untouched by the coupling and propagate at c as usual.

The EED scalar (the "deleted 7th DOF") is a **diagnostic** of the evolved
potentials:

```
C ≡ ∇·A + (1/c²)·∂φ/∂t            ( = ∂µAµ ; Lorenz gauge sets C=0 )
```

Gravitational (GEM) sector couples to C and A via the Kaluza-Klein map
(`KkDirect`: Φ_g = κ_G·C, A_g = κ_G·A; or `KkPoisson`: ∇²Φ_g = −κ_G·C).

**Two deliberate structural choices in the A-equation** (both physical for the
deleted-DOF thesis, both load-bearing — do not "fix" without re-reading this):

1. **The φ-coupling term ∇(∂φ/∂t) carries NO c².** From c²∇C = c²∇(∇·A) +
   ∇(∂φ/∂t), the temporal piece is c²-free. A spurious c² there gives
   β = dt·c²·γ·k ~ 5×10⁸ at CFL → unconditional NaN (ORC-4eg).

2. **The −c²∇(∇·A) term is intentionally OMITTED.** Restoring it would complete
   A to the textbook curl-curl form (c²∇²A − c²∇(∇·A) = −c²∇×∇×A), under which
   the *longitudinal* ∇·A is non-propagating gauge. But the measured EED scalar
   is **~62% longitudinal ∇·A** (AC open helix; the other ~38% is the temporal
   (1/c²)∂φ/∂t half, same sign, from the γ cross-coupling). Keeping bare c²∇²A
   makes that dominant half **radiate** to a detector — which is the entire
   point of simulating a *physical, propagating* deleted DOF. NOTE (2026-07-02):
   it radiates at the golden-ratio branch speeds 0.618c/1.618c derived above,
   **not at c** as this bullet formerly claimed.

**Stability:** the γ cross-coupling is a skew longitudinal sub-update
([[1,−iβ],[−iβ,1]], |λ|=√(1+β²)>1 — unconditionally unstable if fused). Fixed by
Gauss-Seidel ordering (update phi_vel, then a_vel from the *new* phi_vel →
[[1,−iβ],[−iβ,1−β²]], conditionally stable for β≤2) plus a 0.5·CFL safety
factor. See ORC-4eg.

**Relation to the decoupled reading:** van Vlaenderen's EED gives □φ=ρ/ε₀,
□A=µ₀J (cross-coupling cancels), making C obey □C=µ₀∂µJµ exactly but rendering
the longitudinal sector non-propagating and the temporal half zero (no ρ is
injected). That is the form the published Lagrangian actually yields (see the
status note above); the coupled form is Oracle's own construction. Both cannot
be true; **the experiment decides.** This solver commits to the coupled form
because it *retains* the scalar/gravitational terms that the decoupled
derivation evaporates — the terms this project exists to predict and test.

**CENTRAL CAVEAT — the source model violates charge conservation (promoted
from the 2026-05-29 OPEN item; quantified 2026-07-02):** the solver injects J
only, never ρ. A real circuit conserves 4-current (∂µJµ = ∂ₜρ + ∇·J = 0
identically — charge piles up wherever current dies), and in every published
EED variant C is sourced *only* by ∂µJµ ≠ 0, so a faithful source model would
produce **zero** C through that channel. Oracle's open-tip C therefore rides on
a deliberate idealization: injecting ∇·J ≠ 0 with no compensating ρ.
Quantified in `solver-gpu/tests/closed_circuit_control.rs` (48³, γ=1, 1 GHz
AC helix, identical drive): the bare open helix has a genuine ±1 A current
monopole at each tip (∫∇·J dV = 1.11 A measured); adding return leads that
close the circuit collapses the monopole to 0.12 A rasterization residue, yet
**max|C| only drops from 2.81e-5 to 1.90e-5 — 68% of the C signal survives
closure.** Reading: under the *coupled* dynamics, most of C is generated by
the cross-coupling from any oscillating A (tips or no tips) and would survive
a faithful, charge-conserving circuit model; only ~1/3 of the "tip signature"
is the truncation artifact. The flip side: that surviving 68% exists *because*
of the bespoke coupled EOM — in the decoupled reading a charge-conserving
circuit produces no C at all. Any lab prediction pitched to experimenters must
say which of these two claims it is making, and spatial predictions keyed to
wire terminations (end-cap φ, "toroid confines B but not φ") should be re-run
with leads modeled before hardware is positioned around them.

---

## Weak Forms (as implemented in `formulation.py`)

### `maxwell_only` weak form
Find **A** ∈ H(curl) with **A**×n̂=0 on ∂Ω such that:

```
∫_Ω (1/μ₀) curl(A)·curl(v) dx = ∫_Ω J·v dx    ∀v ∈ H₀(curl)
```

### `scalar_only` weak form
Find φ ∈ H¹₀(Ω) such that:

```
∫_Ω ∇φ·∇ψ dx + α² ∫_Ω φ·ψ dx = ∫_Ω S_φ·ψ dx    ∀ψ ∈ H¹₀(Ω)
```

### `eed_coupled` weak form
Find (φ, **A**) ∈ H¹₀ × H₀(curl) such that for all (ψ, **v**):

```
∫_Ω ∇φ·∇ψ dx
  + α² ∫_Ω φ·ψ dx
  + β  ∫_Ω div(A)·ψ dx
  = ∫_Ω S_φ·ψ dx

∫_Ω (1/μ₀) curl(A)·curl(v) dx
  + γ ∫_Ω ∇φ·v dx
  = ∫_Ω J·v dx
```

This is a **block 2×2 saddle-point system**. Assembled as a single mixed
function space in FEniCSx: `W = CG1 × N1curl`.

---

## Source Term Construction

For a coil carrying current I with wire cross-section Σ and path direction **t̂**:

```
J(x) = I/|Σ| · t̂(x)    for x ∈ coil wire region
J(x) = 0               elsewhere
```

This is implemented by tagging the coil wire as a physical group in Gmsh
and assigning the current density as a piecewise constant function.

For S_φ in the static limit:
```
S_φ = -(1/μ₀ε₀) ∇·J
```

For a solenoid, ∇·J ≈ 0 in the interior and concentrates at the end caps.
This means the EED scalar field will be strongest near the coil terminations
— a testable prediction that distinguishes EED from standard EM.

---

## Testable Predictions

The primary purpose of Oracle is generating spatial predictions of where φ is
largest, so lab sensors can be positioned optimally. Key EED signatures to look for:

1. **End-cap enhancement**: φ should peak near the ends of a solenoid, not at
   the center (unlike B, which is maximum at center).

2. **Geometry dependence of φ/B ratio**: The ratio max(φ)/max(B) should change
   differently with coil geometry than standard EM predicts. Sweeping coil
   radius at fixed current tests this.

3. **α sensitivity**: With α=0 (massless scalar), φ extends far beyond the coil.
   With α>0, it falls off exponentially. Varying α and comparing to sensor
   data constrains the physical coupling.

4. **Toroid suppression**: A toroid confines B but should NOT suppress φ if
   EED is correct (since φ sources from ∇·J at terminations). This is a
   strong discriminating test.

**Caveat on predictions 1 and 4 (2026-07-02):** both are keyed to ∇·J ≠ 0 at
wire *terminations* — but a real energized coil has feed leads completing the
circuit, so ∇·J = 0 along the whole loop and the "termination" signature
depends on where the modeled circuit is truncated. The closed-circuit control
(`closed_circuit_control.rs`; see the Central Caveat above) shows ~68% of the
time-domain C survives circuit closure under the coupled dynamics, so a
leads-included signature exists in that model — but it is weaker and its
spatial pattern should be recomputed with leads before positioning sensors.

---

## References

- Woodside, D.A. (1999). "Uniqueness theorems for classical four-vector fields
  in Euclidean and Minkowski spaces." J. Math. Phys. 40, 4911.
- Arbab, A.I. (2009). "Extended electrodynamics and its consequences."
  Prog. Phys. 3, 1–8.
- The "Deleted Degrees of Freedom" paper (reference to be added by Max —
  include full citation here before implementing `eed_coupled` formulation).
- T.T. Brown experimental literature (for empirical motivation of coil geometries
  to test).

---

## Decision Log

- **2025-05-25** α=0 as default — massless scalar maximizes predicted extent,
  giving the strongest detectable signal. Can be increased if predictions are
  spatially too broad to be useful.
- **2025-05-25** Static (magnetostatic) limit only for v1 — time-domain EED
  (retarded potentials, wave propagation of φ) deferred. Static predictions
  are sufficient for DC coil experiments.
- **2025-05-25** Nédélec elements for A — this is mathematically required for
  the vector potential to be in H(curl). CG elements for A is a common
  mistake that produces spurious solutions. Do not change.
- **2025-05-25** S_φ from ∇·J not from ∇·A — in the static limit these are
  equivalent (up to a factor) but computing from J directly avoids a
  two-stage solve.
- **2026-05-25** α, β, γ are free parameters constrained by experiment, not derived
  from first principles. α in units 1/m (Yukawa decay length λ=1/α). β, γ
  dimensionless. All three exposed as UI sliders for parameter sweeps.
- **2026-05-25** β/γ coupling term structure in eed_coupled implemented as documented.
  Marked TODO: VERIFY AGAINST DDOF PAPER — pending full citation from Max.
- **2026-05-25** Coil types expanded: solenoid, toroid (azimuthal winding),
  toroid_poloidal (poloidal winding — key EED test: confines B but not φ),
  flat_spiral, rodin (Rodin/Marko coil — figure-8 toroid winding).
- **2026-05-29** Time-domain EED (`solver-gpu`) committed to the **coupled,
  longitudinally-propagating** potential form (∂²φ/∂t²=c²∇²φ−γc²∂ₜ(∇·A);
  ∂²A/∂t²=c²∇²A−γ∇(∂ₜφ)+c²µ₀J), NOT van Vlaenderen's decoupled □φ=ρ/ε₀, □A=µ₀J.
  Rationale: the decoupled derivation makes the φ↔A cross-coupling cancel, which
  (with J-only injection, no ρ) zeroes the temporal half of C and makes the
  longitudinal ∇·A non-propagating gauge — evaporating exactly the scalar/grav
  terms this project predicts. Both readings are valid derivations; experiment
  adjudicates. We keep the terms.
- **2026-05-29** A-equation OMITS −c²∇(∇·A) **by design** (not a bug). Adding it
  → curl-curl form → longitudinal ∇·A becomes non-propagating. Measurement (AC
  open helix, 32³, γ=1): C = 1.74e-2 splits 62% longitudinal ∇·A (1.09e-2) +
  38% temporal (1/c²)∂ₜφ (6.56e-3), same sign (constructive). Keeping bare c²∇²A
  lets the dominant 62% radiate. The earlier "missing term" bug report is
  reclassified WAI. Φ_g = κ_G·C is ~62% derivation-robust (survives even the
  decoupled reading) — a useful property for a falsifiable prediction.
- **2026-05-29** A-equation φ-coupling term ∇(∂ₜφ) is c²-free (the temporal part
  of c²∇C carries no c²). A spurious c² there caused the ORC-4eg NaN
  instability. Stability secured via Gauss-Seidel velocity-pass ordering +
  0.5·CFL safety factor.
- **2026-05-29** OPEN (deferred, bead filed): only J is injected, never ρ. So
  C's temporal half is sourced only via the cross-coupling, not by real charge
  accumulation ∂ₜρ=−∇·J at open tips. Faithful ρ injection with exact continuity
  would make ∂µJµ=0 and kill C entirely; the EED prediction needs a deliberate
  current-source idealization (∂µJµ≠0). Resolve when modeling ρ.
- **2026-05-29** GEM sector (Φ_g, A_g) co-evolves INSIDE the EM FDTD loop now
  (ORC-j07): one GEM step after each EM step, against a per-step-refreshed C.
  The old post-hoc pass used a frozen C snapshot → ∂C/∂t wrong by ~n_steps and a
  constant ∇C. The SLW (derivative) channel κ_G·∂C/∂t (→Φ_g), κ_G·∇C (→A_g) only
  fires for time-varying C; static configs stay dark (correct). KkDirect remains
  a post-loop algebraic assignment.
- **2026-05-29** GEM leapfrog carried the SAME ORC-4eg instability (fused skew
  velocity read + spurious c² on ∇(∂Φ_g/∂t)) — latent because it had only ever
  run on static/dark configs. Interleaving under an AC drive exposed it (phi_g
  83% NaN, masked to 0 by the ORC-fwe max-fold). Fixed identically (ORC-21g):
  Gauss-Seidel split vel_gem → vel_gem_phi + vel_gem_a; ∇(∂Φ_g/∂t) is c²-free;
  longitudinal A_g stays propagating (c²∇²A_g, no −c²∇(∇·A_g)). GEM EOM as
  implemented: ∂²Φ_g/∂t²=c²∇²Φ_g − c²∂ₜ(∇·A_g) + κ_G·∂ₜC ; ∂²A_g/∂t²=c²∇²A_g −
  ∇(∂ₜΦ_g) + κ_G·∇C. AC open helix now gives a finite, non-zero Φ_g.
- **2026-07-02** (ORC-0r9) **Longitudinal dispersion derived and measured; the
  "radiates at c" claim was WRONG.** Fourier analysis of the coupled EOM's
  longitudinal sector gives (c²k²−ω²)² = γ²ω²c²k², i.e. ω± = ck(√(γ²+4)∓γ)/2:
  two branches at 0.618c and 1.618c for γ=1 (golden ratio; ω₊ superluminal).
  Verified two ways: eigen-decomposition of the exact discrete Gauss-Seidel
  one-step update matrix (|λ|=1, ω/ck = 0.618/1.617 at kΔx=0.1) and a GPU
  front-speed measurement (`eed_gamma1_dispersion.rs`: γ=0 control 1.004c,
  γ=1 front 1.51c). All "radiates at c" language in this file corrected;
  detector-timing predictions must use ω±. The ω₊ > c branch means the coupled
  model is acausal in some frame — acknowledged as a property of the bespoke
  form, not patched.
- **2026-07-02** (ORC-iqn) **Framing corrected: the coupled form is bespoke.**
  The 2026-05-29 entries call the coupled and decoupled forms "equally
  derivable" — overstated. The decoupled □φ=ρ/ε₀, □A=µ₀J follows from the
  standard EED/Stueckelberg action; the coupled form follows from no action we
  can identify, is not Lorentz covariant, and its φ-equation is a postulate
  (≡ ∂ₜC = ∇²φ at γ=1), not Gauss's law. Simulating it remains a legitimate
  explicit hypothesis; presenting it as standard EED does not. Status note
  added at the top of the time-domain section.
- **2026-07-02** (ORC-mlo) **Closed-circuit control quantified the source
  idealization.** Open helix vs same helix + return leads, identical AC drive
  (48³, γ=1, 1 GHz): tip monopole ∫∇·J dV = 1.11 A → 0.12 A on closure, but
  max|C| = 2.81e-5 → 1.90e-5 (**68% survives**). So under the coupled dynamics
  most of C comes from the γ cross-coupling (any oscillating A), not the ∂µJµ
  tip channel; ~1/3 of the tip signature is the truncated-circuit artifact. In
  the decoupled reading a closed circuit gives C≈0 — so "C detected from a
  closed-loop device" would discriminate the two readings. This also CORRECTS
  the 2026-05-29 "Φ_g = κ_G·C is ~62% derivation-robust" claim: in the
  decoupled reading with conserved sources C≡0 and the residual ∇·A is
  gauge, so nothing survives that reading for closed circuits; the 62/38
  split is a property of the coupled model, not evidence for it. The OPEN
  ρ-injection item from 2026-05-29 is now the Central Caveat section above.
- **2026-06-03** GEM **mass sources** ρ_m, J_m wired (ORC-0tl): ordinary matter
  sources the gravitational sector independently of the EED κ_G channel. Gravity
  ATTRACTS, hence +4πG (vs EM's −1/ε₀). Static/elliptic form (solved directly):
  ∇²Φ_g = 4πG·ρ_m, ∇²A_g = (4πG/c²)·J_m, J_m = ρ_m·v — RHS carries NO c², giving
  Φ_g = −GM/r outside a sphere. Time-domain/leapfrog form: the source added to
  the acceleration is −4πG·c²·ρ_m for Φ_g (and −4πG·J_m for A_g); the c² there
  cancels the wave operator's c²∇². Same physics — the c² only appears where it
  must undo the wave-operator c². Validation: uniform sphere → −GM/r recovered to
  5.5% on 64³ (zero-Dirichlet box; →1% on 128³ + analytic BC). 4πG ≈ 8.385×10⁻¹⁰
  survives f32 directly. Shipped as the static elliptic solve
  (`run_gem_mass_static`, additive PCG, works in any mode); the per-step
  dynamic/radiative leapfrog source is deferred (it only matters for *moving*
  masses radiating gravitationally, not the static background the Woodward/Mach
  testbeds use). Entity: `CoilType::MassSphere` (current-free) + the
  `mass_density_kg_m3` / `mass_velocity_m_s` fields on any entity.
