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
S_φ = -c⁻¹ ∂(∇·A)/∂t ≈ (1/μ₀ε₀) ∇·J   (magnetostatic limit)
```

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
