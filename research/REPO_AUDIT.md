# What to carry forward from Oracle

Research audit dated 22 September 2026, against starting commit `5692f66`.
This pass inspected the field-theory contract, selected source/shader code,
existing issue descriptions, and published sources. It did **not** rerun GPU
experiments or independently reproduce the repository's reported measurements.
No solver equations were changed.

## Assessment

The repository has useful simulation and visualization infrastructure, plus
valuable negative findings already recorded in July. Its current scalar and
gravity outputs nevertheless cannot establish an electromagnetic origin of
gravity: parts of the predicted coupling were put into the equations by hand.
The next research artifact should be a derivation and a small discriminating
benchmark, before a more elaborate device simulation.

“Extended electrodynamics” does not name a single theory. The AB scalar,
massive-vector/Stueckelberg constructions, nonlinear electrodynamics, and the
repo's custom coupled equations need separate definitions and tests. The
[paper library](PAPERS.md) separates those branches.

## Findings tied to files

| Finding | Evidence in this repository | Consequence |
|---|---|---|
| The time-domain coupled model is bespoke. | [FIELD_THEORY.md](../FIELD_THEORY.md), time-domain status note; [fdtd_em.wgsl](../solver-gpu/src/shaders/fdtd_em.wgsl). | Its longitudinal branches are properties of this model, not a published EED prediction established by citation. |
| The AC injection supplies current without the compensating charge evolution. | [inject_j.wgsl](../solver-gpu/src/shaders/inject_j.wgsl); field-theory source caveat; issue ORC-xk8. | A nonzero divergence at a modeled wire end can be a source-truncation artifact. Complete the continuity equation before interpreting C. |
| The direct gravity assignment is postulated. | [gem_kk_direct.wgsl](../solver-gpu/src/shaders/gem_kk_direct.wgsl) sets `phi_g = kappa_g * C` and `a_g = kappa_g * A`. | An output proportional to the chosen coupling is not an independent derivation of that coupling. |
| The direct-map shader's toroid explanation does not follow from its map. | Its comment says nonzero exterior A makes exterior B_g appear, although the implementation uses a constant multiplier. | For constant kappa, curl(A_g) = kappa curl(A) = kappa B. Nonzero A alone does not imply nonzero B_g. |
| The source-sign conventions need reconciliation. | The time-domain contract displays `+ mu0*c^2*J`; [state.rs](../solver-gpu/src/grid/state.rs) computes a negative `source_amp` near its AC injection loop. | Check the current orientation, potential conventions, and all observables before declaring a bug or comparing phases. This audit flags the mismatch; it does not settle it. |
| The older static formulation has a heuristic source normalization. | `S_phi = -(1/mu0/epsilon0) div(J)` and the nearby normalization caveat in FIELD_THEORY.md. | The equality to a time derivative of div(A) is not a derived magnetostatic identity. Re-establish dimensions and a common action for the static and dynamic models. |
| Some historical rationale remains misleading after later corrections. | Earlier decision-log entries call the coupled/decoupled forms equally derivable and describe a derivation-robust fraction; later entries withdraw those claims. | Read the July corrections first. A chronological log is not a clean statement of the current physical assumptions. |

The July dispersion and closed-circuit controls are worth retaining as regression
evidence. A successful code test verifies behavior of the equations implemented;
it does not show that nature follows those equations. The reported 68% surviving
C signal under circuit closure remains a result of the custom coupled dynamics.

## A derivation that decides the source question

The following is an explicit baseline derivation, not a claim that every paper
called EED adopts identical conventions. In flat spacetime use signature
\((+,-,-,-)\), \(x^0=ct\), \(A^\mu=(\phi/c,\mathbf A)\),
\(J^\mu=(c\rho,\mathbf J)\), and
\(\Box=c^{-2}\partial_t^2-\nabla^2\). Set

\[
F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu,
\qquad C=\partial_\mu A^\mu.
\]

For a constant dimensionless coefficient \(\xi\), consider

\[
\mathcal L=-\frac{F_{\mu\nu}F^{\mu\nu}}{4\mu_0}
           -\frac{\xi C^2}{2\mu_0}-J_\mu A^\mu.
\]

Varying the potential and dropping the boundary variation gives

\[
\partial_\mu F^{\mu\nu}+\xi\partial^\nu C=\mu_0J^\nu.
\]

Taking its divergence, the antisymmetric term cancels:

\[
\xi\Box C=\mu_0\partial_\nu J^\nu
          =\mu_0(\partial_t\rho+\nabla\!\cdot\mathbf J).
\]

At \(\xi=1\), the potential equations reduce to
\(\Box A^\nu=\mu_0J^\nu\), or

\[
\Box\phi=\rho/\epsilon_0,\qquad \Box\mathbf A=\mu_0\mathbf J.
\]

These are decoupled equations for the potentials with a very important coupled
**source constraint**. With compatible zero scalar initial data, no incoming
scalar excitation, and conserved four-current, uniqueness gives \(C=0\).
Nonzero homogeneous initial data are a different experiment. In particular,
`div(J) != 0` alone is not evidence for `div_four(J) != 0`: charge can accumulate.
This structure is central to [Modanese's AB formulation](https://arxiv.org/abs/1609.00238).

This also corrects an overstatement in the old documentation: decoupled wave
equations do not by themselves imply that longitudinal potential components
cannot propagate. The physically relevant questions are which combinations are
gauge or constrained, which sources excite them, and how a detector couples.

In standard Maxwell/QED calculations a gauge-fixing term is accompanied by the
appropriate physical-state/constraint treatment. Promoting the term to physical
dynamics is a new theory. For example, the displayed sign produces a negative
coefficient of \((\partial_t A^0)^2\) in the unconstrained kinetic density for
positive \(\xi\). A claimed physical scalar therefore needs an explicit
Hamiltonian/constraint and positive-norm analysis. Lorentz covariance alone
does not establish a healthy extra mode. This is a diagnostic to perform on
each full theory, not a blanket rejection of all scalar extensions. Compare
the different role of the compensating field in [Ruegg and Ruiz-Altaba](https://arxiv.org/abs/hep-th/0304245).

## Why the current gravity map needs a derivation

In a common Kaluza–Klein parametrization, the higher-dimensional line element
has schematic form

\[
ds_5^2=g_{\mu\nu}dx^\mu dx^\nu
 +\sigma^2(dy+\kappa A_\mu dx^\mu)^2,
\]

with signature and conformal factors depending on convention. It introduces a
four-dimensional metric, a vector potential, and a scalar associated with the
extra dimension. Identifying the mixed five-dimensional metric components with
A does **not** identify the four-dimensional Newtonian gravitational potential
with \(C\). The reduced action and matter motion must be derived. See
[Overduin and Wesson](https://arxiv.org/abs/gr-qc/9805018).

There is a direct independent diagnostic. Under a Maxwell gauge transformation
\(A^\mu\mapsto A^\mu+\partial^\mu\chi\), F is unchanged but
\(C\mapsto C+\Box\chi\). The repo's direct rule would then change
\(\Phi_g\mapsto\Phi_g+\kappa_G\Box\chi\). If ordinary gauge-equivalent
inputs imply different measured accelerations, an additional physical gauge
selection, symmetry-breaking action, and consistent detector coupling must be
specified. Calling the rule “Kaluza–Klein” does not supply those ingredients.

Units deserve a separate audit. In SI, A has units of tesla·metre and C of
tesla; \(\Phi_g\) is labeled \(\mathrm{m^2/s^2}\). The gravitational-vector
normalization must also be fixed consistently with its force law. One slider
cannot silently perform two unrelated unit conversions.

For the ordinary GR control use the full source equation

\[
G_{\mu\nu}+\Lambda g_{\mu\nu}
=\frac{8\pi G}{c^4}(T^{\rm matter}_{\mu\nu}+T^{\rm EM}_{\mu\nu}).
\]

The Maxwell stress-energy tensor is traceless in four spacetime dimensions,
but it is not zero. Electromagnetic energy still gravitates. A zero trace
restricts a particular scalar coupling; it does not remove the tensor coupling.
[Geons](https://doi.org/10.1103/PhysRev.97.511) provide a concrete example.

## What a replacement benchmark must establish

These are scientific acceptance criteria for subsequent work, not a claim that
this collection has completed the derivations or experiments.

1. **A named action and conventions.** Write down every field, parameter unit,
   matter coupling, boundary condition, and limiting theory. Derive field
   equations and stress-energy, then count constraints and propagating modes.
   For R15 include the dependence of all coupling/source functions during
   variation, not just the convenient displayed wave equations.
2. **A complete source.** Include leads, returns, displacement effects, and
   charge accumulation, or specify a justified microscopic anomalous current.
   Measure the discrete residual \(\partial_t\rho+\nabla\cdot J\).
   An open subsystem exchanging charge with a reservoir is not, by itself,
   fundamental charge nonconservation. Quantum uncertainty also does not
   automatically invalidate an operator conservation law.
3. **An observable.** Predict a neutral test body's acceleration, differential
   phase, radiation flux, or force on the complete apparatus. Plotting A or C
   alone is insufficient. Include momentum carried by fields, radiation,
   feeds, and supports.
4. **Independent controls.** Recover ordinary Maxwell fields and the relevant
   Einstein–Maxwell weak-field result. Verify gauge-equivalent descriptions
   give the same observables where gauge symmetry applies. Use dimensional
   analysis, analytic solutions, resolution refinement, and boundary-distance
   variation rather than only checking whether a plotted signal is nonzero.
5. **A falsifiable parameter range.** Choose parameters before fitting the
   target signal. Map predictions to composition-dependent free fall,
   charge-conservation tests, or photon-mass bounds only where the model
   actually predicts those observables. Include conventional forces in the
   same apparatus model.

The repo's non-Lorentz-covariant dispersion should also be analyzed on its own
terms. A superluminal branch relative to c is a serious issue for a proposed
Lorentz-invariant completion, but a preferred-frame model does not acquire a
closed causal loop merely by a coordinate boost. Do not replace one unsupported
physical claim with another; state the causal structure and intended domain.

## Citation repairs and immediate research choices

- The matching Arbab title is **2017, Modern Physics Letters B 31, 1750099**
  (R09), not the original 2009 citation.
- The user confirmed DDOF as the motivating title. R18 is the located
  Wilhelm edition; the exact original reading copy has not been compared.
  Its correction ledger is R19. Read the text and corrected claims together.
- The clarified experimental target is Brown’s static solid-dielectric
  effect. R38/R42/R43 directly address that branch; the NASA ionic-wind
  report alone does not settle it. See the [focused assessment](BROWN_STATIC_DIELECTRICS.md).
- For the scalar EED branch, the priority reproduction candidate is R15, with R16 as a possible later
  reduced-solution benchmark. Existence of its solutions is not yet an
  experiment or a proof that the full theory is stable.
- R23/R24 supply an independent standard-theory route: photon–graviton
  conversion in magnetic fields. R26 supplies a mathematical route: reproduce
  a classical double-copy example without treating the correspondence as a
  physical source-conversion mechanism.
- ORC-xk8's old instruction to preserve nonconservation to retain a desired
  signal has been superseded by the conserved-source acceptance criteria. Source realism and a viable theory must determine
  the result; preserving a signal cannot determine the source model.

The scientific target is modest and decisive: either a candidate survives a
consistent source/action/observable comparison and makes a distinct prediction,
or we identify exactly which assumption prevents it. Either result improves
on choosing equations because they retain the phenomenon we hoped to find.
