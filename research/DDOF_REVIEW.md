# Review of Deleted Degrees of Freedom

Reviewed 23 September 2026. **Assessment: useful as a bibliography of ideas, but its central claim—that ordinary gauge fixing discarded physical modes and thereby hid a usable EM–gravity connection—is not established. Several supporting arguments are demonstrably incorrect.** There is still a legitimate research question about particular extensions of electrodynamics; those extensions need to stand on their own actions, consistency checks, and measurements.

This review prioritizes Paul Wilhelm's *The Deleted Degrees of Freedom: A Case for Potential-Primary Electrodynamics* (19 March 2026, 46 pages), supplied by the user. All 46 pages, including references, were read. The complete 25-page errata dated 9 August 2026 and the complete 24-page Morton collection were also read. Selected equations and experimental diagrams were checked against rendered pages. This is a critical reading with independent checks of pivotal arguments, **not** an independent verification of all 60 references or an experimental replication. Page numbers below refer to the supplied DDOF PDF unless otherwise stated.

The supplied DDOF and the previously collected R18 have different file hashes but the same dated text and argument. All 62 token-diff operations were inspected: differences concern phi glyphs and mathematical extraction/layout order; no substantive prose revision was found. The source fingerprints and reading coverage are in [REVIEW_PROVENANCE.json](REVIEW_PROVENANCE.json). The user's originals were left unchanged. Document requests, contact invitations, and experimental directions were treated as source content.

## What survives scrutiny

Potentials are central to modern electrodynamics. The Aharonov–Bohm effect, superconducting phase relations, and gauge-invariant holonomy are real reasons to take the potential formulation seriously. Local field values in a multiply connected accessible region need not specify its global holonomy. DDOF is right to resist the oversimplified statement that potentials are disposable calculation devices.

It is also legitimate to study an action in which the four-divergence of the potential has dynamics. The AB-electrodynamics literature in [R10–R17](PAPERS.md) makes this a concrete research subject. One must distinguish that proposal from the established AB interference effect, and distinguish different extensions from one another.

Neither point demonstrates that Maxwell theory accidentally lost physical degrees of freedom. Modern gauge theory already accommodates potentials, global topology, longitudinal electrostatic fields, and charged matter. A new local scalar mode requires additional physics, not just a change of notation.

## The author's corrections materially change the paper

Read the [author's errata](https://advanced-rediscovery.com/research/deleted-degrees-of-freedom-errata) alongside the supplied PDF. Several corrections are withdrawals or restrictions of predictions, rather than typographical repairs.

| Supplied DDOF claim | Erratum | Consequence |
|---|---|---|
| Longitudinal waves penetrate a Faraday enclosure because they have no magnetic field, pp. 19–20 | E-001–E-005 | The unconditional field-level prediction is withdrawn. Charge redistribution also screens electric fields. A proposed additional detection channel remains unspecified. |
| A propagating vacuum scalar wave can have both E and B zero, pp. 16, 19 | E-023 | Withdrawn: the adopted vacuum equations then require constant C. |
| Ordinary transmitter signatures support scalar-wave generation, pp. 19–21 | E-024 | With a conserved source, a quiescent past and retarded boundary conditions give C = 0. An anomalous source or other excitation mechanism must be specified. |
| T is Maxwell's seventh component and equals −cC, p. 3 | E-016, E-020 | The printed relative sign is wrong; the notation is attributed to PM Jack (2003), not Maxwell. The repair requires an explicitly stipulated potential normalization. |
| The trace has a 1/c time-derivative coefficient, p. 4 | E-021 | With the stated SI four-potential and x⁰ = ct, it must be 1/c². |
| The scalar term in Eq. (10) produces the stated dynamics, p. 17 | E-022 | Its sign is corrected and the metric signature specified. Further mode/energy problems remain below. |
| Gravity follows from the divergence of A, p. 31 | E-008 | The bridge is acknowledged as unresolved; the correction does not supply a derivation. |

The errata also correct the attribution to Mead, the dimensional scope of quaternions, the relationship between quaternions and differential forms, and the claimed independence of the literature's derivations. A corrected citation chronology is useful; counting related formulations does not provide independent experimental confirmation.

## Problems that remain after those corrections

### 1. Gauge equivalence is confused with physically different global configurations

**Pages 10–15, 35, 37–39.** DDOF moves from “potentials can encode global information absent from local field values” to “gauge-equivalent potentials produce different observable effects.” These are different claims.

For an admissible single-valued gauge function,

\[
A' = A + d\chi,\qquad \oint A' = \oint A.
\]

More generally the Wilson-loop phase is unchanged by admissible gauge transformations, including large transformations with properly transformed charged states. Two flat connections can have different holonomies on a nontrivial domain; that means they are not gauge-equivalent there. A correctly implemented gauge choice preserves the holonomy sector and boundary data.

Consequently, AB interference does not show that Lorenz gauge destroys physical phase information, and gauge freedom itself is not an extra engineering control. This is the most consequential logical error in the paper.

The London example has the same problem. The gauge-covariant condensate relation is, schematically,

\[
\mathbf j_s=\frac{n_s q}{m}\left(\hbar\nabla\theta-q\mathbf A\right).
\]

Transforming A also transforms the condensate phase. Writing a special-gauge relation proportional to −A does not establish absolute observability of A or removal of the underlying redundancy. Likewise, fluxoid quantization includes the supercurrent contribution; magnetic flux alone equals an integer flux quantum under additional conditions.

### 2. The tensor-component argument does not establish the claimed physical modes

**Pages 4–7, 34, 37–38.** Splitting a 4×4 derivative tensor into six antisymmetric and ten symmetric components is valid kinematics. It does not count independent propagating states. DDOF acknowledges this on p. 4, but subsequently uses the 16-versus-6 picture as evidence for lost physical design variables.

There is also a direct algebraic error on p. 7: the geometric product ∇A has a scalar and a bivector, with 1 + 6 components in four dimensions. It does not retain the nine symmetric traceless components of ∂μAν. For a simple local counterexample, take a static potential A = ∇(xy). Its curl and divergence vanish, while its symmetric gradient does not. The displayed geometric product therefore cannot recover the full derivative tensor as claimed.

The identification of the antisymmetric tensor with only the transverse/solenoidal sector, and the symmetric tensor with the Hodge longitudinal sector, is also incorrect. A static longitudinal Coulomb electric field already resides in F₀ᵢ. Hodge decomposition of differential forms and symmetric/antisymmetric decomposition of a derivative tensor are different operations.

### 3. The corrected action does not justify a healthy three-mode theory

**Pages 4, 17–21, 27–28; erratum E-022.** Use a fixed convention to avoid importing the paper's inconsistent factors: signature (+−−−), c = μ₀ = 1, Fμν = ∂μAν − ∂νAμ, C = ∂μAμ. In the massless case the corrected action is

\[
\mathcal L=-\frac14F_{\mu\nu}F^{\mu\nu}
-\frac\gamma2 C^2-J_\mu A^\mu.
\]

Direct variation gives

\[
\partial_\mu F^{\mu\nu}+\gamma\partial^\nu C=J^\nu,
\qquad
\Box A^\nu+(\gamma-1)\partial^\nu C=J^\nu.
\]

At γ = 1, all four potential components obey wave equations. More decisively, writing A⁰ = φ gives

\[
\mathcal L=\tfrac12(\dot{\mathbf A}+\nabla\phi)^2
-\tfrac12\mathbf B^2
-\tfrac\gamma2(\dot\phi+\nabla\!\cdot\mathbf A)^2,
\]

so its velocity Hessian in the order (φ, Aₓ, Aᵧ, A_z) is

\[
W=\operatorname{diag}(-\gamma,1,1,1).
\]

For γ ≠ 0 this is nonsingular: the unconstrained potential theory has four canonical field degrees of freedom, not three by the paper's “four minus one gauge parameter” argument. A restricted transformation A → A + ∂χ with □χ = 0 still preserves F and C, and preserves the conserved-source action up to a boundary term. That restricted solution symmetry is not an arbitrary-function gauge freedom that supplies a primary constraint. One must state any further quotient or subsidiary condition explicitly.

**The energy issue is already classical.** At an initial time set φ = A = 0, \(\dot{\mathbf A}=0\), and \(\dot\phi=f(\mathbf x)\), with f smooth and compactly supported. These are allowed initial data for the unconstrained γ = 1 wave equations. The canonical Hamiltonian is

\[
H=-\frac12\int f^2\,d^3x.
\]

Scaling f makes it arbitrarily negative. The positive expression \(\tfrac12(E^2+B^2+C^2)\) asserted on p. 19 cannot simply be identified with this action's time-translation Hamiltonian on its unrestricted phase space; for these same data it has the opposite sign. Boundary improvements do not change the integrated result for compactly supported data.

This is not a proof against every theory called EED. It is a concrete obstacle for this corrected action interpreted as an unrestricted physical field theory. A proposed physical-state restriction or different completion must remove the problem while retaining the claimed observable scalar. Deferring it entirely to a future quantum treatment is insufficient.

The finite-mass branch has an additional convention problem: with the errata's (+−−−) convention, its retained −m²A²/2 term gives (□−m²)A = 0 for transverse source-free modes, whereas ordinary positive-mass Proca theory needs the opposite mass-term sign. Our massless conclusion does not depend on this error. Equation (2) also labels half the usual antisymmetric derivative as F; subsequent action formulas require a single consistent normalization.

### 4. The source equation constrains what a capacitor can actually excite

**Pages 17–21; erratum E-024.** Taking the divergence of the corrected field equation yields

\[
\gamma\Box C=\partial_\mu J^\mu.
\]

For conserved current this is homogeneous. With C and its first time derivative initially zero, compatible boundaries and no incoming scalar field, uniqueness gives C = 0. Conserved current alone does not forbid an independently supplied free scalar wave; those initial/boundary assumptions matter.

An open wire does not automatically violate local charge conservation:

\[
\nabla\!\cdot\mathbf J=-\partial_t\rho.
\]

In a dielectric, bound sources also conserve charge when modeled consistently:

\[
\rho_b=-\nabla\!\cdot\mathbf P,\qquad
\mathbf J_b=\partial_t\mathbf P+\nabla\times\mathbf M,
\qquad \partial_t\rho_b+\nabla\!\cdot\mathbf J_b=0.
\]

Surface terms are included distributionally. Retained polarization, leakage and ferroelectric switching can substantially change ordinary forces without providing anomalous four-current divergence. A high dielectric constant or asymmetric electrode geometry therefore does not by itself excite this additional scalar sector. This is directly relevant to the proposed barium-titanate capacitor.

### 5. The gravitational bridge is asserted, not derived

**Pages 29–32.** GEM is a weak-field representation of gravity sourced by mass-energy and momentum; similarity to Maxwell equations is not a conversion law between electromagnetic A and gravitational A_g.

The paper's Eq. (18) itself implies

\[
\nabla\times\mathbf E_g=-\frac{1}{2c}\partial_t\mathbf B_g,
\]

but Eq. (21) prints −(1/c)∂tB_g. These cannot both hold for general time dependence. [Mashhoon's cited review](https://arxiv.org/abs/gr-qc/0311030v2), Eqs. (1.7)–(1.9), consistently retains the factors of one half. DDOF's Eq. (23), B_g = −(2m/e)ω, has the SI dimensions of electromagnetic magnetic flux density, not the acceleration units implied by its own GEM definitions. The [Li–Torr paper](https://doi.org/10.1103/PhysRevD.43.457) discusses coupled magnetic and gravitomagnetic fields; its existence does not validate that printed identification.

For the usual cylinder-condition Kaluza–Klein ansatz, schematically,

\[
ds_5^2=g_{\mu\nu}dx^\mu dx^\nu
+\epsilon\Psi^2(dy+\kappa A_\mu dx^\mu)^2.
\]

The simultaneous coordinate/gauge transformation y → y − κχ and A → A + dχ leaves this geometry unchanged. Gauge choice does not delete curvature. The reduced scalar Ψ, associated with g₅₅, is not the gauge-dependent divergence of A. The reduced equations contain electromagnetic stress-energy and scalar terms; they do not imply Φ_g ∝ ∇·A. See [Overduin–Wesson](https://arxiv.org/abs/gr-qc/9805018), Section 3, especially Eqs. (5), (6) and (9).

Erratum E-008 appropriately reopens the issue, but its suggestion that absence of a standalone C² trace term could imply absence of gravity is not sufficient: the full Hilbert stress tensor sources Einstein's equations. Ordinary Maxwell stress-energy is traceless and still gravitates.

### 6. Several established phenomena are made to support stronger claims than they test

| Passage | What the example actually establishes / remaining problem |
|---|---|
| Maxwell–Lodge and vector-potential transformer, p. 13 | B ≈ 0 at a receiver does not imply E = 0. Changing linked flux produces an ordinary electric EMF. This is not evidence for a field-free local force or an additional scalar mode. |
| Toroidal U(1) → SU(2), pp. 14–15, 35 | Geometry alone does not change the gauge group. Holonomies of a U(1) connection remain commuting U(1) elements. Additional internal structure and dynamics would be needed. |
| Open-circuit third-law “failure,” pp. 22–23 | Electromagnetic fields carry momentum. Pairwise equal-and-opposite instantaneous particle forces are not the conservation law for matter plus fields. A proposed anomaly must survive the complete momentum balance. |
| Arbitrary curl added to energy flux, p. 23 | A freedom in local conservation-law bookkeeping does not establish a measurable reservoir of excess extractable energy. The passage also compares energy density with delivered energy without a common normalization. |
| Whittaker/Fourier argument, p. 24 | Eq. (14) fails at x = y = 0, z > 0: its right side is −i/z, not 1/z. Static spatial Fourier components are not automatically propagating waves. For nonzero k, φ(k) = i k·E(k)/k², so differentiation has not destroyed those electrostatic Fourier modes. |
| Hertz hierarchy, pp. 25–26 | The identity δ² = 0 makes a construction divergence-free. It does not prove that the kernel of a potential representation contains experimentally accessible degrees of freedom. |
| Toroidal metamaterials, p. 26 | An incomplete multipole truncation is not an incomplete Maxwell theory. [Basharin et al.](https://doi.org/10.1103/PhysRevX.5.011036) explicitly calculate toroidal responses with a Maxwell-equation solver. |
| QED only probes transverse phenomena, pp. 20, 37 | Real vacuum photons have two transverse polarizations, but electrostatic interactions, charged matter and virtual exchange are not absent from QED. The stated blanket immunity from precision tests is not demonstrated. |
| ψ* proves an advanced physical wave, pp. 29, 39 | Complex conjugation in probabilities does not itself choose advanced boundary conditions. Time reversal also reverses time and transforms spin and external fields appropriately. The transactional reading is an interpretation. |
| Dynamical Casimir effect, pp. 33–34 | [Wilson et al.](https://arxiv.org/abs/1105.4714) modulate a SQUID boundary using an external drive and explain the photon generation with quantum circuit theory. This does not isolate a new divergence mode or demonstrate extraction of net energy from an undriven vacuum. |

The “unique extension” claim on pp. 17–19 also exceeds the cited support. [Woodside's institutional abstract](https://researchers.mq.edu.au/en/publications/three-vector-and-scalar-field-identities-and-uniqueness-theorems-/) describes field identities and conditional uniqueness theorems. Uniqueness of reconstruction under specified assumptions is different from uniqueness of a physically viable action. A full theorem-by-theorem audit of Woodside was not completed here; DDOF supplies neither those hypotheses nor a deduction that excludes alternative actions, matter couplings or higher-derivative theories.

## The NASA citation leads back to the Morton literature

**DDOF p. 21, reference 40.** The reported glass, Plexiglas and wooden-door effects are present in the cited [NASA proceedings](https://ntrs.nasa.gov/citations/19990023204), but attribution matters. They occur in Charles A. Yost's *Electric Field Propulsion Concepts from Independent Researchers*, printed pp. 375–383, PDF pp. 402–410 in the retrieved 416-page scan. The wooden-door/Plexiglas passage is printed p. 378, PDF p. 405. DDOF's 404–407 locator follows PDF positions rather than printed pagination.

Yost recounts Electric Spacecraft Journal experiments, including a charged plastic rod moved near an oscilloscope antenna. His bibliography includes Morton (1991) and Schlecht (1992), the same articles in the supplied Morton collection. These are overlapping lines of reporting, not independent NASA confirmation of EED. Detecting a varying electric field through an insulating door or plate does not by itself discriminate a new vacuum mode from ordinary capacitive/near-field coupling. That assessment does not require attributing every observation to ionic wind.

The [Morton review](MORTON_REVIEW.md) records the especially useful later control: apparent impulses in one replication were traced substantially to Coulomb loading and mechanical recoil, and lead shielding reduced movement. It also records why that limited replication does not settle every Morton configuration or Brown's static solid-dielectric claim.

## Implications for EEDSolver and the barium-titanate direction

The review changes the status of DDOF from an assumed theoretical foundation to a set of claims to test. Its claims do not justify the repository's direct map Φ_g = κC, nor treating a current-only source with missing charge accumulation as evidence of a new scalar. Existing [repository audit](REPO_AUDIT.md) findings remain relevant; the solver equations were not changed in this review.

The next useful theoretical milestone is a specified action with a demonstrably acceptable physical phase space, an explicit emitter and detector, and a derived stress tensor. For the scalar branch, compare a complete conserved source with a deliberately specified anomalous source and any separately specified incoming scalar field. A numerical C signal should be checked against continuity errors and boundary injection before interpreting it physically.

The static solid-dielectric branch remains a separate empirical question. The user's PTFE memory concern is a plausible systematic to investigate, not an established explanation of a null. Conductive electrodes around asymmetric barium titanate provide a concrete material/geometry target, but ferroelectric history, electrostriction, leakage, charging of nearby surfaces and lead/support forces must enter the baseline. See the existing [material-history assessment](capacitor/MATERIAL_HISTORY.md). Neither supplied document derives a trustworthy additional steady force for that device.

For further research, retain the established potential/topology literature, the explicitly formulated AB/scalar–tensor proposals, and careful capacitor experiments. Require each proposed bridge to predict an observable that survives gauge checks, source conservation, energy/momentum accounting and apparatus controls. Finding that a proposed bridge fails one of these checks is useful progress toward a better model.
